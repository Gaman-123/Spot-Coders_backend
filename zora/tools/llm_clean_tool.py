"""
LLM-Guided Data Cleaning Tool
===============================

Unlike static IQR/median pipelines, this tool:

  1. Sends a 10-row sample + full schema profile to LLM (Groq LLaMA 3.3 70B)
  2. LLM analyses the data's specific quirks and writes a tailored pandas
     cleaning script (e.g. custom imputation, domain-specific range clips,
     skewed-distribution handling, date parsing, categorical normalisation).
  3. The script is executed in a restricted sandbox (no builtins, no imports
     beyond pandas/numpy/re — prevents code injection).
  4. Returns a CleanReport identical to the static clean_tool, so the rest of
     the pipeline (Critic loop, Supabase write, SSE) is unchanged.

Fallback:
  If LLM generation fails, script execution raises, or the result drops more
  than 40% of rows, the static clean_tool is used automatically.
"""

from __future__ import annotations

import io
import json
import os
import re
import textwrap
import traceback
from typing import Optional

import numpy as np
import pandas as pd
from groq import Groq

from models.schemas import CleanReport, SchemaProfile
from utils.config import settings


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert clinical data scientist. You will receive a dataset schema \
and a 10-row sample, then write a Python pandas cleaning script tailored \
specifically to this dataset's quirks.

Rules:
- The script MUST operate on a variable called `df` (pandas DataFrame).
- Do NOT import anything. pandas is aliased as `pd`, numpy as `np`, re as `re`.
- Do NOT use exec(), eval(), open(), os, sys, subprocess or any I/O.
- Preserve the target column exactly (do not impute or drop it).
- At the end, assign the cleaned result back to `df`.
- Return ONLY the Python script inside a ```python ... ``` fence. No prose.
- The script should handle: null imputation, duplicate removal, outlier \
  treatment (IQR or domain-specific clipping), categorical normalisation, \
  string cleaning, date parsing, and any other quirks visible in the sample.
- Add a short inline comment per major step explaining WHY.
- Prefer domain-appropriate strategies: e.g. for clinical vitals use \
  physiological plausibility ranges; for free-text columns strip/lowercase; \
  for skewed distributions (e.g. cost, LOS) use median not mean.
"""

_USER_TEMPLATE = """\
DATASET: {filename}
TARGET COLUMN (do not impute/drop): {target}
ROWS: {rows}  COLS: {cols}

SCHEMA:
{schema_json}

SAMPLE (10 rows, all columns):
{sample_csv}

STATISTICS:
{stats_json}

Write the cleaning script now.
"""


# ── Sandbox execution ─────────────────────────────────────────────────────────

_SAFE_GLOBALS: dict = {
    "__builtins__": {
        # Allow only a minimal safe subset of builtins
        "len": len, "range": range, "int": int, "float": float,
        "str": str, "bool": bool, "list": list, "dict": dict,
        "tuple": tuple, "set": set, "print": print,
        "isinstance": isinstance, "zip": zip, "enumerate": enumerate,
        "min": min, "max": max, "abs": abs, "round": round,
        "any": any, "all": all,
    },
    "pd": pd,
    "np": np,
    "re": re,
}


def _execute_script(script: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute `script` in a restricted sandbox.
    `df` is injected as a local variable. Raises on any error.
    """
    local_vars = {"df": df.copy()}
    exec(script, _SAFE_GLOBALS, local_vars)  # noqa: S102 — restricted env
    result = local_vars.get("df")
    if not isinstance(result, pd.DataFrame):
        raise TypeError("Script did not assign a DataFrame back to `df`.")
    return result


# ── Script extraction ─────────────────────────────────────────────────────────

def _extract_script(llm_response: str) -> str:
    """Pull the Python script from ```python ... ``` fences."""
    # Try fenced block first
    match = re.search(r"```python\s*(.*?)```", llm_response, re.DOTALL)
    if match:
        return textwrap.dedent(match.group(1).strip())
    # Fallback: try plain ``` block
    match = re.search(r"```(.*?)```", llm_response, re.DOTALL)
    if match:
        return textwrap.dedent(match.group(1).strip())
    raise ValueError("LLM response contained no fenced Python script.")


# ── Schema summariser ─────────────────────────────────────────────────────────

def _schema_summary(df: pd.DataFrame, profile: SchemaProfile, target_col: str | None) -> str:
    """Build a compact schema JSON string for the LLM prompt."""
    cols = []
    for col in df.columns:
        series = df[col]
        entry: dict = {
            "name": col,
            "dtype": str(series.dtype),
            "null_pct": round(series.isna().mean() * 100, 1),
            "is_target": col == target_col,
        }
        if pd.api.types.is_numeric_dtype(series):
            entry["min"] = round(float(series.min(skipna=True)), 3) if series.notna().any() else None
            entry["max"] = round(float(series.max(skipna=True)), 3) if series.notna().any() else None
            entry["mean"] = round(float(series.mean(skipna=True)), 3) if series.notna().any() else None
            entry["skew"] = round(float(series.skew()), 3) if series.notna().sum() > 3 else None
        else:
            entry["unique"] = int(series.nunique())
            entry["top_values"] = series.value_counts().head(5).index.tolist()
        cols.append(entry)
    return json.dumps(cols, indent=2)


def _stats_summary(df: pd.DataFrame) -> str:
    """Concise describe() as JSON."""
    try:
        desc = df.describe(include="all").fillna("").to_dict()
        # Truncate long lists
        return json.dumps({k: {str(sk): str(sv)[:80] for sk, sv in v.items()} for k, v in desc.items()}, indent=2)[:3000]
    except Exception:
        return "{}"


# ── LLM call ─────────────────────────────────────────────────────────────────

def _generate_cleaning_script(df: pd.DataFrame, profile: SchemaProfile, target_col: str | None) -> str:
    """
    Ask Groq LLaMA 3.3 70B to write a pandas cleaning script.
    Falls back to llama-3.1-8b-instant if rate-limited.
    """
    api_key = settings.GROQ_API_KEY
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.")

    client = Groq(api_key=api_key)

    sample_csv = df.head(10).to_csv(index=False)
    schema_json = _schema_summary(df, profile, target_col)
    stats_json = _stats_summary(df)

    user_content = _USER_TEMPLATE.format(
        filename=profile.filename,
        target=target_col or "unknown",
        rows=len(df),
        cols=len(df.columns),
        schema_json=schema_json,
        sample_csv=sample_csv,
        stats_json=stats_json,
    )

    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    last_exc: Exception | None = None

    for model in models:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            raw = resp.choices[0].message.content or ""
            return _extract_script(raw)
        except Exception as e:
            last_exc = e
            continue

    raise RuntimeError(f"All LLM models failed. Last: {last_exc}")


# ── Diff analysis (for CleanReport) ──────────────────────────────────────────

def _compute_clean_report(
    run_id: str,
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    profile: SchemaProfile,
    target_col: str | None,
    script: str,
) -> CleanReport:
    """
    Compare before/after DataFrames and produce a CleanReport.
    The `llm_script` field is stored in critic_feedback as a preview.
    """
    rows_before = len(df_before)
    rows_after = len(df_after)

    # Null imputation per column
    nulls_imputed: dict[str, int] = {}
    imputation_strategy: dict[str, str] = {}
    for col in df_before.columns:
        if col not in df_after.columns:
            continue
        null_before = int(df_before[col].isna().sum())
        null_after = int(df_after[col].isna().sum())
        imputed = null_before - null_after
        if imputed > 0:
            nulls_imputed[col] = imputed
            # Detect strategy heuristically
            if pd.api.types.is_numeric_dtype(df_after[col]):
                imputation_strategy[col] = "llm_numeric"
            else:
                imputation_strategy[col] = "llm_categorical"
        else:
            imputation_strategy[col] = "none"

    # Row reduction (dedup + outliers combined — script may do both)
    dupes_removed = rows_before - rows_after
    outliers_removed: dict[str, int] = {}

    # New flag columns added by LLM script
    new_cols = [c for c in df_after.columns if c not in df_before.columns]
    missingness_flags = [c for c in new_cols if "flag" in c.lower() or "missing" in c.lower()]

    # Save cleaned CSV
    out_dir = os.path.join(settings.OUTPUT_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    df_after.to_csv(os.path.join(out_dir, "cleaned.csv"), index=False)

    return CleanReport(
        run_id=run_id,
        rows_before=rows_before,
        rows_after=rows_after,
        dupes_removed=dupes_removed,
        same_visit_dupes_removed=0,
        nulls_imputed=nulls_imputed,
        outliers_removed=outliers_removed,
        imputation_strategy=imputation_strategy,
        missingness_flags_added=missingness_flags,
        critic_feedback=f"[LLM-guided] Script generated by Groq LLaMA. Preview:\n{script[:500]}",
        sample_5_rows=df_after.head(5).to_dict(orient="records"),
    )


# ── Main entry point ─────────────────────────────────────────────────────────

def llm_clean_tool(
    run_id: str,
    profile: SchemaProfile,
    feedback_ctx: str | None = None,
) -> tuple[CleanReport, str]:
    """
    LLM-Guided Data Cleaning.

    Args:
        run_id:       Pipeline run ID.
        profile:      SchemaProfile from S1 IngestAgent.
        feedback_ctx: Optional critic feedback from previous attempt
                      (appended to prompt on retry).

    Returns:
        (CleanReport, generated_script_str)

    Raises:
        RuntimeError if LLM fails, script crashes, or result is unsafe (>40% row loss).
    """
    from tools.preprocessing_utils import normalize_column_names, normalize_target_column

    filepath = os.path.join(settings.OUTPUT_DIR, run_id, "ingested.csv")
    df_original = pd.read_csv(filepath)
    df_original.columns = normalize_column_names(df_original.columns.tolist())

    target_col = normalize_target_column(profile.target_candidate, df_original.columns.tolist())

    # Append critic feedback to the prompt if this is a retry
    feedback_suffix = ""
    if feedback_ctx:
        feedback_suffix = (
            f"\n\nPREVIOUS CRITIC FEEDBACK (address these issues):\n{feedback_ctx}"
        )

    # 1. Generate script via LLM
    script = _generate_cleaning_script(df_original, profile, target_col)
    if feedback_suffix:
        # Re-generate with feedback appended (simple approach: inject into prompt)
        # For retries, append feedback and call again
        script = _generate_cleaning_script_with_feedback(
            df_original, profile, target_col, feedback_ctx
        )

    # 2. Execute in sandbox
    df_cleaned = _execute_script(script, df_original)

    # 3. Safety guard: reject if LLM dropped too many rows
    row_loss_pct = (len(df_original) - len(df_cleaned)) / max(len(df_original), 1)
    if row_loss_pct > 0.40:
        raise RuntimeError(
            f"LLM script dropped {row_loss_pct*100:.1f}% of rows — unsafe. "
            "Falling back to static clean_tool."
        )

    # 4. Ensure target column survived
    if target_col and target_col not in df_cleaned.columns:
        raise RuntimeError(
            f"LLM script removed the target column '{target_col}' — unsafe."
        )

    # 5. Build CleanReport
    report = _compute_clean_report(run_id, df_original, df_cleaned, profile, target_col, script)

    return report, script


def _generate_cleaning_script_with_feedback(
    df: pd.DataFrame,
    profile: SchemaProfile,
    target_col: str | None,
    feedback: str,
) -> str:
    """Re-generate script with critic feedback appended to user message."""
    api_key = settings.GROQ_API_KEY
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.")

    client = Groq(api_key=api_key)

    sample_csv = df.head(10).to_csv(index=False)
    schema_json = _schema_summary(df, profile, target_col)

    user_content = _USER_TEMPLATE.format(
        filename=profile.filename,
        target=target_col or "unknown",
        rows=len(df),
        cols=len(df.columns),
        schema_json=schema_json,
        sample_csv=sample_csv,
        stats_json=_stats_summary(df),
    ) + f"\n\nPREVIOUS CRITIC FEEDBACK (fix these issues in your script):\n{feedback}"

    for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            raw = resp.choices[0].message.content or ""
            return _extract_script(raw)
        except Exception:
            continue

    raise RuntimeError("LLM script re-generation with feedback failed.")
