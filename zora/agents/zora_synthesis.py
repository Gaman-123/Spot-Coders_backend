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
    """Single-agent LLM call with Groq primary (higher quota) → Gemini fallback."""
    # PRIMARY is Groq to avoid Gemini 429 quota blocks
    candidates = [
        LLM(model="groq/llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY, temperature=temperature),
        LLM(model="gemini/gemini-2.0-flash", api_key=settings.GOOGLE_API_KEY, temperature=temperature),
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
            if "429" in str(e).lower():
                log.warning(f"LLM 429 for {llm.model}. Attempting next candidate...")
            last_exc = e
            continue
    raise RuntimeError(f"LLM call failed for role '{role}' after all candidates: {last_exc}")


# ── RAG citations ─────────────────────────────────────────────────────────────

from services.embedding_service import get_embedding_async

async def _retrieve_rag_citations(run_id: str, query: str, k: int = 5) -> list[dict]:
    """Retrieve RAG citations using the robust embedding service."""
    try:
        vec = await get_embedding_async(query)
        if not vec:
            log.warning("Could not obtain embedding for RAG query. Proceeding without context.")
            return []
            
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
    except Exception as e:
        log.error(f"RAG citation retrieval failed: {e}")
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
        f"Stability: {alphafold['stability_score']} ({alphafold.get('confidence_plddt', 'unknown')} confidence)",
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

import asyncio
import concurrent.futures

_debate_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def _run_debate(
    profile, clean_report, metrics, top_features,
    alphafold, misfold, finance, safety, gnn_result, rag_context
):
    """
    3-round agentic debate — Rounds 1 & 2 run in parallel threads (2x faster).
    """
    ml_blk      = _ml_block(metrics, top_features)
    protein_blk = _protein_block(alphafold, misfold)
    finance_blk = _finance_block(finance)
    gnn_blk     = _gnn_block(gnn_result)
    safety_blk  = _safety_block(safety)
    dataset_hdr = (
        f"DATASET: {profile.filename} | "
        f"{clean_report.rows_after} rows | Target: {profile.target_candidate}"
    )

    # ── Round 1: Both positions (run in parallel via threads) ─────────────
    def clinical_pos():
        return _call_llm(
            role="Clinical Strategist",
            goal="Argue the clinical risk profile from ML, protein, and network evidence.",
            backstory="Senior clinical ML researcher. You advocate for patient safety above all.",
            prompt=f"""{dataset_hdr}
ML: {ml_blk}
PROTEIN: {protein_blk}
GNN: {gnn_blk}
SAFETY: {safety_blk}
RAG: {rag_context or 'None'}
Write 3-5 sentence CLINICAL POSITION: state risk level, top biomarkers, protein interpretation, clinical action.""",
            expected_output="3-5 sentence clinical position paper.",
            temperature=0.3,
        )

    def financial_pos():
        return _call_llm(
            role="Financial Auditor",
            goal="Argue the financial burden and economic feasibility.",
            backstory="Healthcare economist. You challenge over-treatment and unsustainable protocols.",
            prompt=f"""{dataset_hdr}
ML (brief): {ml_blk}
FINANCE: {finance_blk}
SAFETY: {safety_blk}
Write 3-5 sentence FINANCIAL POSITION: state cost risk, denial analysis, waste preventability, intervention.""",
            expected_output="3-5 sentence financial position paper.",
            temperature=0.3,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_c = pool.submit(clinical_pos)
        fut_f = pool.submit(financial_pos)
        clinical_position  = fut_c.result()
        financial_position = fut_f.result()

    # ── Round 2: Rebuttals (also parallel) ────────────────────────────────
    def finance_rebuttal():
        return _call_llm(
            role="Financial Auditor (rebuttal)",
            goal="Challenge financially unjustified clinical claims.",
            backstory="Healthcare economist who challenges over-treatment.",
            prompt=f"""Clinical Strategist said: {clinical_position}
Your finance data: {finance_blk}
Write 2-sentence FINANCIAL REBUTTAL: challenge one claim, propose sustainable alternative.""",
            expected_output="2-sentence financial rebuttal.",
            temperature=0.2,
        )

    def clinical_rebuttal():
        return _call_llm(
            role="Clinical Strategist (rebuttal)",
            goal="Challenge financially-driven arguments that underweight patient safety.",
            backstory="Clinical ML researcher who defends evidence-based safety.",
            prompt=f"""Financial Auditor said: {financial_position}
Your evidence: {protein_blk} | {gnn_blk}
Write 2-sentence CLINICAL REBUTTAL: defend one safety point, propose balanced approach.""",
            expected_output="2-sentence clinical rebuttal.",
            temperature=0.2,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_cr = pool.submit(clinical_rebuttal)
        fut_fr = pool.submit(finance_rebuttal)
        clinical_rebuttal_txt  = fut_cr.result()
        financial_rebuttal_txt = fut_fr.result()

    # ── Round 3: Final synthesis (single call — sees all 4 positions) ─────
    final_synthesis = _call_llm(
        role="Chief Healthcare Analytics Synthesizer",
        goal="Integrate debate into a final balanced clinical report.",
        backstory="CMO with 25 years chairing clinical review boards. Balances clinical and financial realities.",
        prompt=f"""{dataset_hdr}

CLINICAL POSITION: {clinical_position}
FINANCIAL POSITION: {financial_position}
CLINICAL REBUTTAL: {clinical_rebuttal_txt}
FINANCIAL REBUTTAL: {financial_rebuttal_txt}

PROTEIN: {protein_blk}
FINANCE: {finance_blk}
SAFETY: {safety_blk}
GNN: {gnn_blk}
RAG: {rag_context or 'None'}

Write FINAL SYNTHESIS (6-9 sentences):
1. Where Clinical & Financial AGREED
2. Where they DISAGREED and why
3. Final overall risk verdict (low/moderate/high/critical) — justified
4. Top ML features + clinical meaning
5. Protein stability + GNN hidden hubs
6. Financial risk with context
7. Safety flags requiring action
8. One specific actionable recommendation balancing both views""",
        expected_output="6-9 sentence final synthesis report with risk verdict and actionable recommendation.",
        temperature=0.2,
    )

    return clinical_position, financial_position, clinical_rebuttal_txt, financial_rebuttal_txt, final_synthesis


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
    rag_citations = await _retrieve_rag_citations(run_id, rag_query, k=5)
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
