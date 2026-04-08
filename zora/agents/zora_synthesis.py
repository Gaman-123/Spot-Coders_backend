"""
S5 — Zora Synthesis Agent: Agentic Debate Architecture
=======================================================

BEFORE (v1):
  Single synthesis pass — one LLM call merges all data into a report.
  Problem: no adversarial challenge — biased toward confirming ML findings.

AFTER (v2 — this file):
  Three-agent debate before final synthesis:

  Round 1 — Position Papers:
    • Clinical Strategist  — argues the CLINICAL risk profile
                             (ML model, SHAP features, protein stability,
                              GNN hidden hubs, misfold risk)
    • Financial Auditor    — argues the FINANCIAL burden & feasibility
                             (denial probability, waste estimate,
                              readmission rate, cost-benefit)

  Round 2 — Cross-Examination:
    • Financial Auditor rebutts Clinical Strategist's position
    • Clinical Strategist rebutts Financial Auditor's position

  Round 3 — Final Synthesis:
    • Synthesizer reads both position papers + rebuttals + RAG context
    • Writes the final structured clinical report, acknowledging tensions
    • Labels any unresolved disagreements explicitly

  Benefits:
    ✓ Catches cases where good ML accuracy masks poor financial viability
    ✓ Catches cases where financial fear masks genuine clinical urgency
    ✓ Debate transcript stored in Supabase for auditability
    ✓ Token-efficient: each debater gets only its relevant slice of data
"""

from __future__ import annotations

import json
import time
from crewai import Agent, Task, Crew, Process, LLM
from google import genai as google_genai
from google.genai import types as genai_types
from supabase import create_client

from tools.finance_tool import finance_tool
from tools.safety_vault import run_safety_vault
from services.supabase_service import insert_insight_row
from utils.sse_manager import sse_manager
from utils.config import settings
from models.schemas import SchemaProfile, CleanReport
from datetime import datetime, timezone


# ── LLM helper ────────────────────────────────────────────────────────────────

def _call_llm(
    role: str,
    goal: str,
    backstory: str,
    prompt: str,
    expected_output: str,
    temperature: float = 0.3,
) -> str:
    """Single-agent LLM call with Gemini → Groq fallback."""
    candidates = [
        LLM(model="gemini/gemini-2.0-flash", api_key=settings.GOOGLE_API_KEY, temperature=temperature),
        LLM(model="groq/llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY, temperature=temperature),
    ]
    last_exc = None
    for llm in candidates:
        try:
            agent = Agent(
                role=role, goal=goal, backstory=backstory,
                llm=llm, verbose=False, allow_delegation=False, max_iter=1,
            )
            task = Task(description=prompt, expected_output=expected_output, agent=agent)
            return str(Crew(agents=[agent], tasks=[task],
                            process=Process.sequential, verbose=False).kickoff())
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"LLM call failed for role '{role}': {last_exc}")


# ── RAG citations ─────────────────────────────────────────────────────────────

def _retrieve_rag_citations(run_id: str, query: str, k: int = 5) -> list[dict]:
    try:
        client = google_genai.Client(api_key=settings.GOOGLE_API_KEY)
        resp   = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=[query],
            config=genai_types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768,
            ),
        )
        vec     = resp.embeddings[0].values
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
        result  = supabase.rpc("match_documents", {
            "query_embedding": vec,
            "match_count": k,
            "filter": {"run_id": run_id},
        }).execute()
        return [
            {"chunk_text": r["chunk_text"], "similarity": round(r["similarity"], 4)}
            for r in (result.data or [])
        ]
    except Exception:
        return []


# ── Shared context helpers ─────────────────────────────────────────────────────

def _ml_block(metrics: dict, top_features: dict) -> str:
    top5 = ", ".join(f"{k}={v:.4f}" for k, v in list(top_features.items())[:5])
    return (
        f"Model: {metrics['model']}  AUC: {metrics['auc']}  "
        f"Accuracy: {metrics['accuracy']}  F1: {metrics['f1']}\n"
        f"Top SHAP features: {top5}"
    )

def _protein_block(alphafold: dict, misfold: dict | None) -> str:
    lines = [
        f"Protein: {alphafold['protein_name']} (UniProt: {alphafold.get('uniprot_id','')})",
        f"Stability: {alphafold['stability_score']} ({alphafold['confidence']} confidence)",
    ]
    if misfold and misfold.get("enabled"):
        lines += [
            f"Misfold stuck-score: {misfold.get('stuck_score')} ({misfold.get('energy_state')})",
            f"Aggregation propensity: {misfold.get('aggregation_propensity')}",
        ]
    return "\n".join(lines)

def _finance_block(finance: dict) -> str:
    return (
        f"Denial probability: {finance['denial_probability']*100:.1f}%\n"
        f"Healthcare waste estimate: ${finance['waste_estimate_usd']:,.0f}\n"
        f"Predicted cohort readmission rate: {finance['predicted_readmission_rate']*100:.0f}%"
    )

def _gnn_block(gnn_result: list[dict] | None) -> str:
    if not gnn_result:
        return "No GNN data available."
    hubs = [r["protein"] for r in gnn_result if r.get("is_hidden_hub")]
    top3 = sorted(gnn_result, key=lambda x: x.get("fusion_score", 0), reverse=True)[:3]
    return (
        f"Hidden hubs (low SHAP, high centrality): {', '.join(hubs) or 'none'}\n"
        f"Top fusion-score proteins: {', '.join(r['protein'] for r in top3)}"
    )

def _safety_block(safety: dict) -> str:
    flags = "; ".join(f['message'] for f in safety["safety_flags"]) or "No flags triggered."
    return f"Safety flags: {flags}\nDoctor review required: {safety['doctor_review']}"


# ── The three-round debate ─────────────────────────────────────────────────────

def _run_debate(
    profile: SchemaProfile,
    clean_report: CleanReport,
    metrics: dict,
    top_features: dict,
    alphafold: dict,
    misfold: dict | None,
    finance: dict,
    safety: dict,
    gnn_result: list[dict] | None,
    rag_context: str,
) -> tuple[str, str, str, str, str]:
    """
    Execute all three debate rounds.

    Returns:
        (clinical_position, financial_position,
         clinical_rebuttal, financial_rebuttal,
         final_synthesis)
    """
    ml_blk      = _ml_block(metrics, top_features)
    protein_blk = _protein_block(alphafold, misfold)
    finance_blk = _finance_block(finance)
    gnn_blk     = _gnn_block(gnn_result)
    safety_blk  = _safety_block(safety)
    dataset_hdr = (
        f"DATASET: {profile.filename} | "
        f"{clean_report.rows_after} rows | "
        f"Target: {profile.target_candidate}"
    )

    # ── Round 1a: Clinical Strategist position ────────────────────────────
    clinical_prompt = f"""You are the Clinical Strategist in a risk assessment debate.
Your job: argue ONLY the CLINICAL risk — what the molecular and ML evidence says about patient risk.
Do NOT comment on financial implications — that is the Financial Auditor's role.

DATA:
{dataset_hdr}

ML EVIDENCE:
{ml_blk}

PROTEIN BIOMARKER:
{protein_blk}

GNN PROTEIN NETWORK:
{gnn_blk}

SAFETY FLAGS:
{safety_blk}

RAG EVIDENCE (grounding):
{rag_context or 'No RAG context available.'}

Write your CLINICAL POSITION (3-5 sentences):
- State the overall clinical risk level (low/moderate/high/critical)
- Cite the top 2-3 SHAP features as clinically meaningful biomarkers
- Comment on protein stability and what it means for disease mechanism
- Mention any hidden hub proteins that represent overlooked clinical risk
- Recommend whether immediate clinical action is required and why
""".strip()

    clinical_position = _call_llm(
        role="Clinical Strategist",
        goal="Argue the clinical risk profile from ML, protein, and network evidence.",
        backstory=(
            "Senior clinical ML researcher with 15 years in precision medicine. "
            "You interpret biomarkers, protein stability, and network centrality "
            "as clinical signals. You advocate for patient safety above all."
        ),
        prompt=clinical_prompt,
        expected_output="3-5 sentence clinical position paper stating risk level, biomarker interpretation, and clinical recommendation.",
        temperature=0.3,
    )

    # ── Round 1b: Financial Auditor position ──────────────────────────────
    financial_prompt = f"""You are the Financial Auditor in a risk assessment debate.
Your job: argue ONLY the FINANCIAL and operational burden — cost, denial risk, resource allocation.
Do NOT comment on clinical specifics — that is the Clinical Strategist's role.

DATA:
{dataset_hdr}

ML EVIDENCE (brief):
{ml_blk}

FINANCIAL RISK:
{finance_blk}

SAFETY FLAGS:
{safety_blk}

Write your FINANCIAL POSITION (3-5 sentences):
- State the overall financial risk level (low/moderate/high/critical)
- Comment on the denial probability and what drives it
- State the healthcare waste estimate and whether it is preventable
- Assess whether the cohort readmission rate implies cost-ineffective care
- Recommend specific financial interventions (pre-auth review, care coordination, etc.)
""".strip()

    financial_position = _call_llm(
        role="Financial Auditor",
        goal="Argue the financial burden and economic feasibility of the clinical scenario.",
        backstory=(
            "Healthcare economist and insurance actuary with 20 years at a major payer. "
            "You evaluate clinical decisions through the lens of cost-effectiveness, "
            "denial risk, and resource stewardship. You challenge over-treatment."
        ),
        prompt=financial_prompt,
        expected_output="3-5 sentence financial position paper stating cost risk level, denial analysis, and economic intervention recommendation.",
        temperature=0.3,
    )

    # ── Round 2a: Financial Auditor rebuts Clinical Strategist ────────────
    finance_rebuttal_prompt = f"""You are the Financial Auditor. You have now read the Clinical Strategist's position.
Challenge any points where clinical urgency is proposed WITHOUT corresponding financial justification.

CLINICAL STRATEGIST'S POSITION:
{clinical_position}

YOUR FINANCIAL DATA:
{finance_blk}

Write a SHORT REBUTTAL (2-3 sentences):
- Identify 1 specific claim in the clinical position that is financially unjustified or un-evidenced
- Propose an alternative that balances clinical benefit with financial sustainability
""".strip()

    financial_rebuttal = _call_llm(
        role="Financial Auditor (rebuttal)",
        goal="Challenge financially unjustified clinical claims.",
        backstory="Healthcare economist who challenges over-treatment and unsustainable clinical protocols.",
        prompt=finance_rebuttal_prompt,
        expected_output="2-3 sentence financial rebuttal challenging one clinical claim.",
        temperature=0.2,
    )

    # ── Round 2b: Clinical Strategist rebuts Financial Auditor ────────────
    clinical_rebuttal_prompt = f"""You are the Clinical Strategist. You have now read the Financial Auditor's position.
Challenge any points where financial caution may compromise patient safety.

FINANCIAL AUDITOR'S POSITION:
{financial_position}

YOUR CLINICAL DATA:
{protein_blk}
{gnn_blk}
Safety: {safety_blk}

Write a SHORT REBUTTAL (2-3 sentences):
- Identify 1 specific financial argument that underweights patient safety or clinical urgency
- Propose an approach that preserves both clinical integrity and financial realism
""".strip()

    clinical_rebuttal = _call_llm(
        role="Clinical Strategist (rebuttal)",
        goal="Challenge financially-driven arguments that underweight patient safety.",
        backstory="Clinical ML researcher who defends evidence-based patient safety protocols.",
        prompt=clinical_rebuttal_prompt,
        expected_output="2-3 sentence clinical rebuttal defending patient safety.",
        temperature=0.2,
    )

    # ── Round 3: Synthesizer final verdict ────────────────────────────────
    synthesis_prompt = f"""You are the Chief Healthcare Analytics Synthesizer.
You have just presided over a structured debate between the Clinical Strategist and the Financial Auditor.
Your role: integrate all evidence and debate positions into a final, balanced clinical report.

DATA SUMMARY:
{dataset_hdr}

ML MODEL: {ml_blk}

CLINICAL STRATEGIST'S POSITION:
{clinical_position}

FINANCIAL AUDITOR'S POSITION:
{financial_position}

CLINICAL STRATEGIST'S REBUTTAL (to Finance):
{clinical_rebuttal}

FINANCIAL AUDITOR'S REBUTTAL (to Clinical):
{financial_rebuttal}

PROTEIN: {protein_blk}
FINANCE: {finance_blk}
SAFETY: {safety_blk}
GNN: {gnn_blk}

RAG EVIDENCE:
{rag_context or 'No external evidence available.'}

Write the FINAL SYNTHESIS REPORT (6-9 sentences):
1. Acknowledge the debate — where Clinical Strategist and Financial Auditor AGREED
2. Acknowledge where they DISAGREED and why
3. State the final overall risk verdict (low/moderate/high/critical) — JUSTIFY it
4. Cite the top ML features and what they clinically mean
5. Integrate protein stability + GNN hidden hubs into the interpretation
6. State the financial risk with context
7. List safety flags that require action
8. End with a single, specific, actionable recommendation that balances both perspectives
""".strip()

    final_synthesis = _call_llm(
        role="Chief Healthcare Analytics Synthesizer",
        goal=(
            "Integrate a structured clinical debate into a final balanced report "
            "that weighs clinical urgency against financial sustainability."
        ),
        backstory=(
            "Chief Medical Officer and health economist who has chaired clinical review boards "
            "for 25 years. You synthesize adversarial expert opinions into actionable, "
            "evidence-grounded recommendations that are defensible to both clinicians and payers."
        ),
        prompt=synthesis_prompt,
        expected_output=(
            "6-9 sentence synthesis report covering: debate consensus/disagreements, "
            "final risk verdict with justification, ML + protein + GNN interpretation, "
            "financial risk context, safety flags, and one actionable recommendation."
        ),
        temperature=0.2,
    )

    return clinical_position, financial_position, clinical_rebuttal, financial_rebuttal, final_synthesis


# ── Main agent ────────────────────────────────────────────────────────────────

async def run_synthesis_agent(
    run_id: str,
    profile: SchemaProfile,
    clean_report: CleanReport,
    s4_result: dict,
    gnn_result: list[dict] | None = None,
) -> dict:

    t0 = time.monotonic()

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_synthesis",
        "status": "running",
        "output_summary": "Starting Agentic Debate: Clinical Strategist vs Financial Auditor...",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    automl    = s4_result["automl"]
    alphafold = s4_result["alphafold"]
    misfold   = s4_result.get("misfold")
    metrics   = automl["metrics"]
    top_features = automl.get("top_features", {})

    # ── Finance Tool ─────────────────────────────────────────────────────────
    finance = finance_tool(
        run_id=run_id,
        ml_auc=metrics["auc"],
        stability_score=alphafold["stability_score"],
        rows_after=clean_report.rows_after,
    )

    # ── Safety Vault ─────────────────────────────────────────────────────────
    safety = run_safety_vault(
        ml_auc=metrics["auc"],
        ml_accuracy=metrics["accuracy"],
        stability_score=alphafold["stability_score"],
        denial_probability=finance["denial_probability"],
        waste_estimate_usd=finance["waste_estimate_usd"],
        protein_name=alphafold["protein_name"],
        misfold_summary=misfold,
    )

    # ── RAG Citations ─────────────────────────────────────────────────────────
    rag_query = (
        f"Healthcare dataset {profile.filename} with target {profile.target_candidate}. "
        f"Columns: {', '.join(profile.numeric_columns[:5])}."
    )
    rag_citations = _retrieve_rag_citations(run_id, rag_query, k=5)
    rag_context   = "\n".join(c["chunk_text"] for c in rag_citations[:3])

    # ── Round 1: Announce Clinical Strategist starting ───────────────────────
    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_debate_clinical",
        "status": "running",
        "output_summary": "[Debate R1] Clinical Strategist building clinical risk position...",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    # ── Agentic Debate — all 3 rounds ────────────────────────────────────────
    (
        clinical_position,
        financial_position,
        clinical_rebuttal,
        financial_rebuttal,
        synthesis_text,
    ) = _run_debate(
        profile=profile,
        clean_report=clean_report,
        metrics=metrics,
        top_features=top_features,
        alphafold=alphafold,
        misfold=misfold,
        finance=finance,
        safety=safety,
        gnn_result=gnn_result,
        rag_context=rag_context,
    )

    # SSE: Debate round completions
    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_debate_clinical",
        "status": "completed",
        "output_summary": f"[Debate R1] Clinical position: {clinical_position[:200]}...",
        "data": {"clinical_position": clinical_position},
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_debate_financial",
        "status": "completed",
        "output_summary": f"[Debate R1] Financial position: {financial_position[:200]}...",
        "data": {"financial_position": financial_position},
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_debate_rebuttal",
        "status": "completed",
        "output_summary": (
            f"[Debate R2] Rebuttals complete. "
            f"Clinical rebuttal: {clinical_rebuttal[:120]}... "
            f"Financial rebuttal: {financial_rebuttal[:120]}..."
        ),
        "data": {
            "clinical_rebuttal":  clinical_rebuttal,
            "financial_rebuttal": financial_rebuttal,
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    latency_ms = int((time.monotonic() - t0) * 1000)

    # ── Write to Supabase insights table ─────────────────────────────────────
    debate_transcript = {
        "round_1_clinical":   clinical_position,
        "round_1_financial":  financial_position,
        "round_2_rebuttal_clinical":  clinical_rebuttal,
        "round_2_rebuttal_financial": financial_rebuttal,
    }

    insight_row = {
        "run_id":               run_id,
        "ml_model":             metrics["model"],
        "ml_accuracy":          metrics["accuracy"],
        "ml_auc":               metrics["auc"],
        "top_features":         top_features,
        "stability_score":      alphafold["stability_score"],
        "pdb_link":             alphafold["pdb_link"],
        "protein_name":         alphafold["protein_name"],
        "denial_probability":   finance["denial_probability"],
        "waste_estimate":       finance["waste_estimate_usd"],
        "rag_citations":        rag_citations,
        "synthesis_text":       synthesis_text,
        "safety_flags":         safety["safety_flags"],
        "doctor_review":        safety["doctor_review"],
        "protein_summary_json": misfold if misfold and misfold.get("enabled") else None,
        "gnn_summary_json":     gnn_result,
        "debate_transcript":    debate_transcript,
    }
    inserted_row = insert_insight_row(insight_row)
    insight_id = inserted_row["id"] if inserted_row else None

    # ── Final SSE ─────────────────────────────────────────────────────────────
    misfold_summary_text = ""
    if misfold and misfold.get("enabled"):
        misfold_summary_text = f"Misfold stuck-score: {misfold.get('stuck_score')}. "

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_synthesis",
        "status": "completed",
        "latency_ms": latency_ms,
        "output_summary": (
            f"[Agentic Debate complete — 3 rounds, 5 LLM calls] "
            f"Denial risk: {finance['denial_probability']*100:.0f}%. "
            f"Waste estimate: ${finance['waste_estimate_usd']:,.0f}. "
            f"Doctor review: {safety['doctor_review']}. "
            f"{misfold_summary_text}"
            f"Insight #{insight_id} written with debate transcript."
        ),
        "data": {
            "insight_id":          insight_id,
            "denial_probability":  finance["denial_probability"],
            "waste_estimate_usd":  finance["waste_estimate_usd"],
            "doctor_review":       safety["doctor_review"],
            "safety_flags_count":  safety["rules_triggered"],
            "rag_citations_count": len(rag_citations),
            "debate_rounds":       3,
            "debate_transcript":   debate_transcript,
            "synthesis_preview":   synthesis_text[:300],
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return {
        "insight_id":          insight_id,
        "synthesis_text":      synthesis_text,
        "finance":             finance,
        "safety":              safety,
        "rag_citations":       rag_citations,
        "misfold":             misfold,
        "debate_transcript":   debate_transcript,
    }
