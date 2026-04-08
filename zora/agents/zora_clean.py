import time
import json
from crewai import Agent, Task, Crew, Process, LLM
from google import genai as google_genai
from google.genai import types as genai_types
from supabase import create_client
from tools.clean_tool import clean_tool
from tools.llm_clean_tool import llm_clean_tool
from services.supabase_service import update_run_status
from utils.sse_manager import sse_manager
from utils.config import settings
from models.schemas import SchemaProfile, CleanReport
from datetime import datetime, timezone

MAX_CRITIC_RETRIES = 3
PASS_THRESHOLD = 7


from services.embedding_service import get_embedding_async

async def _retrieve_schema_context(run_id: str, query: str, k: int = 3) -> str:
    """
    Embed the query using the robust embedding service and retrieve top-k
    schema chunks for this run_id from Supabase pgvector.
    Returns concatenated chunk_text as context string.
    """
    query_vector = await get_embedding_async(query)
    if not query_vector:
        log.warning("Embedding failed for schema context retrieval. Operating without RAG context.")
        return ""

    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
    result = supabase.rpc("match_documents", {
        "query_embedding": query_vector,
        "match_count": k,
        "filter": {"run_id": run_id}
    }).execute()

    if not result.data:
        return ""

    return "\n".join(row["chunk_text"] for row in result.data)


# ── CRITIC LLM ────────────────────────────────────────────────────────────────

def _make_critic_crew(llm: LLM, prompt: str) -> Crew:
    agent = Agent(
        role="Data Quality Critic",
        goal=(
            "Evaluate the quality of a data cleaning operation. "
            "Score from 0 to 10. Score >= 7 means the cleaning is acceptable. "
            "Return ONLY a JSON object with keys: score (int), passed (bool), feedback (str)."
        ),
        backstory=(
            "You are an expert data quality judge. "
            "You verify that cleaning decisions are justified by the schema profile. "
            "You flag hallucinated or unjustified cleaning steps."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=1
    )
    task = Task(
        description=prompt,
        expected_output=(
            'Valid JSON object only. Example: '
            '{"score": 8, "passed": true, '
            '"feedback": "Median imputation correct for cholesterol. 2 dupes removed cleanly."}'
        ),
        agent=agent
    )
    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)


def _critic_kickoff_with_fallback(prompt: str) -> dict:
    """
    Run the critic with llama-3.1-8b-instant primary (as per architecture diagram),
    fall back to llama-3.3-70b-versatile. Returns parsed JSON dict.
    """
    candidates = [
        LLM(model="groq/llama-3.1-8b-instant",
            api_key=settings.GROQ_API_KEY, temperature=0.0),
        LLM(model="groq/llama-3.3-70b-versatile",
            api_key=settings.GROQ_API_KEY, temperature=0.0),
    ]
    last_exc: Exception | None = None
    for llm in candidates:
        try:
            raw = str(_make_critic_crew(llm, prompt).kickoff())
            # Extract JSON from response (strip markdown fences if present)
            raw = raw.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"Critic LLM failed after all fallbacks. Last: {last_exc}")


# ── MAIN AGENT ────────────────────────────────────────────────────────────────

async def run_clean_agent(
    run_id: str,
    profile: SchemaProfile
) -> CleanReport:

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_clean",
        "status": "running",
        "output_summary": "Requesting LLM-guided cleaning script from Groq LLaMA 3.3...",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    t0 = time.monotonic()
    feedback_ctx: str | None = None
    clean_report: CleanReport | None = None
    critic_result: dict = {}

    cleaning_method = "static"  # default
    generated_script = ""        # populated if LLM path succeeds

    for attempt in range(1, MAX_CRITIC_RETRIES + 1):

        # ── Step 1: Try LLM-Guided first, fall back to static ────────────
        if attempt == 1 or (attempt > 1 and cleaning_method == "llm"):
            # On first attempt: try LLM. On retries: if LLM worked before, retry LLM with feedback
            try:
                await sse_manager.publish(run_id, {
                    "type": "agent_update",
                    "agent": "zora_clean",
                    "status": "running",
                    "output_summary": f"[Attempt {attempt}] Groq LLaMA writing custom pandas script from 10-row sample...",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                clean_report, generated_script = llm_clean_tool(
                    run_id=run_id,
                    profile=profile,
                    feedback_ctx=feedback_ctx,
                )
                cleaning_method = "llm"
                await sse_manager.publish(run_id, {
                    "type": "agent_update",
                    "agent": "zora_clean",
                    "status": "running",
                    "output_summary": (
                        f"[LLM] Script executed. "
                        f"{clean_report.rows_before}→{clean_report.rows_after} rows. "
                        f"Nulls imputed: {sum(clean_report.nulls_imputed.values())}. "
                        f"Script preview: {generated_script[:160].strip()!r}"
                    ),
                    "data": {
                        "cleaning_method": "llm_groq",
                        "script_preview": generated_script[:400],
                        "rows_before": clean_report.rows_before,
                        "rows_after": clean_report.rows_after,
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            except Exception as llm_err:
                # LLM path failed — fall back to static
                cleaning_method = "static"
                generated_script = ""
                await sse_manager.publish(run_id, {
                    "type": "agent_update",
                    "agent": "zora_clean",
                    "status": "running",
                    "output_summary": (
                        f"LLM-guided cleaning failed ({type(llm_err).__name__}: {str(llm_err)[:120]}). "
                        "Falling back to static IQR+median pipeline."
                    ),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                clean_report = clean_tool(
                    run_id=run_id,
                    profile=profile,
                    feedback_ctx=feedback_ctx,
                )
        else:
            # If static was used on attempt 1, keep using static on retries
            clean_report = clean_tool(
                run_id=run_id,
                profile=profile,
                feedback_ctx=feedback_ctx,
            )

        # ── Step 2: RAG — retrieve schema context for grounding ───────────────
        rag_query = (
            f"Dataset {profile.filename} schema profile: "
            f"columns, null rates, data types, target column"
        )
        schema_context = await _retrieve_schema_context(run_id, rag_query, k=3)

        # ── Step 3: Build critic prompt ───────────────────────────────────────
        await sse_manager.publish(run_id, {
            "type": "agent_update",
            "agent": "zora_critic",
            "status": "running",
            "output_summary": f"Critic evaluating cleaning quality (attempt {attempt}/{MAX_CRITIC_RETRIES})...",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Only show columns where an action was actually taken
        active_imputation = {
            col: strategy
            for col, strategy in clean_report.imputation_strategy.items()
            if strategy != "none"
        }

        # Append LLM script to critic prompt if available
        script_block = (
            f"\n\nLLM-GENERATED CLEANING SCRIPT (evaluate if logic is medically sound):\n"
            f"```python\n{generated_script[:600]}\n```\n"
            if generated_script else ""
        )

        critic_prompt = f"""
You are evaluating a data cleaning operation. Score it 0-10 (>=7 = PASS).
Cleaning method used: {cleaning_method.upper()} {'(LLM wrote a custom pandas script)' if cleaning_method == 'llm' else '(static IQR+median pipeline)'}

SCHEMA CONTEXT (from RAG):
{schema_context or 'No context available.'}

ACTIONS TAKEN:
- Duplicate rows removed: {clean_report.dupes_removed}
- Same-visit duplicates removed: {clean_report.same_visit_dupes_removed}
- Columns with null imputation: {json.dumps(clean_report.nulls_imputed)} (count imputed per col)
- Imputation method used: {json.dumps(active_imputation)} (only cols that had nulls)
- Invalid values converted to null: {json.dumps(clean_report.invalid_values_converted)}
- Missingness flags added: {json.dumps(clean_report.missingness_flags_added)}
- Outliers removed via IQR rule: {json.dumps(clean_report.outliers_removed)}
- Rows: {clean_report.rows_before} → {clean_report.rows_after}
- Target column skipped (correct): {profile.target_candidate}
- Sample of cleaned data (first row): {json.dumps(clean_report.sample_5_rows[0]) if clean_report.sample_5_rows else 'N/A'}
{script_block if cleaning_method == 'llm' else ''}
{f'PRIOR FEEDBACK (retry {attempt}): {feedback_ctx}' if feedback_ctx else ''}

Score criteria:
1. Median/LLM imputation for numeric null columns = good
2. Mode/LLM imputation for categorical null columns = good
3. Deduplication = always correct
4. Converting implausible vitals/labs to null with flags = good
5. Dropping outliers via IQR or domain-specific logic = good
6. LLM script logic medically sound and not hallucinated = critical
7. Target column untouched = correct

Return ONLY valid JSON: {{"score": <int 0-10>, "passed": <bool>, "feedback": <str>}}
""".strip()

        # ── Step 4: Run critic ────────────────────────────────────────────────
        critic_result = _critic_kickoff_with_fallback(critic_prompt)
        score = int(critic_result.get("score", 0))
        passed = score >= PASS_THRESHOLD

        clean_report.quality_score = score
        clean_report.critic_feedback = critic_result.get("feedback", "")
        clean_report.passed_critic = passed

        await sse_manager.publish(run_id, {
            "type": "agent_update",
            "agent": "zora_critic",
            "status": "completed" if passed else "failed",
            "output_summary": (
                f"Score: {score}/10. {'PASS' if passed else 'FAIL'}. "
                f"{clean_report.critic_feedback}"
            ),
            "data": {
                "score": score,
                "passed": passed,
                "attempt": attempt,
                "feedback": clean_report.critic_feedback
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        if passed:
            break

        # ── Retry: send feedback context back to clean_tool ──────────────────
        if attempt < MAX_CRITIC_RETRIES:
            feedback_ctx = clean_report.critic_feedback

    # ── Step 5: Persist to Supabase ───────────────────────────────────────────
    latency_ms = int((time.monotonic() - t0) * 1000)

    update_run_status(
        run_id,
        status="s3_complete",
        cleaned_rows=clean_report.rows_after,
        quality_score=clean_report.quality_score,
        cleaning_summary=clean_report.model_dump(),
        completed_at=datetime.now(timezone.utc).isoformat()
    )

    # ── Step 6: SSE — clean completed ─────────────────────────────────────────
    rows_delta = clean_report.rows_before - clean_report.rows_after
    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_clean",
        "status": "completed",
        "latency_ms": latency_ms,
        "output_summary": (
            f"[{cleaning_method.upper()}] {clean_report.rows_before}→{clean_report.rows_after} rows. "
            f"{clean_report.dupes_removed + clean_report.same_visit_dupes_removed} dupes. "
            f"{sum(clean_report.nulls_imputed.values())} nulls imputed. "
            f"{sum(clean_report.outliers_removed.values())} IQR outliers removed. "
            f"Quality score: {clean_report.quality_score}/10."
        ),
        "data": {
            "rows_before":            clean_report.rows_before,
            "rows_after":             clean_report.rows_after,
            "dupes_removed":          clean_report.dupes_removed,
            "nulls_imputed":          clean_report.nulls_imputed,
            "invalid_values_converted": clean_report.invalid_values_converted,
            "missingness_flags_added": clean_report.missingness_flags_added,
            "quality_score":          clean_report.quality_score,
            "passed_critic":          clean_report.passed_critic,
            "cleaning_method":        cleaning_method,
            "script_preview":         generated_script[:300] if generated_script else None,
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return clean_report
