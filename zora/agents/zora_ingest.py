import time
import os
from crewai import Agent, Task, Crew, Process, LLM
from tools.ingest_tool import ingest_tool
from services.supabase_service import update_run_status
from utils.sse_manager import sse_manager
from utils.config import settings
from models.schemas import SchemaProfile
from datetime import datetime, timezone
from datetime import datetime, timezone
import json
import os
from Bio import SeqIO
import pdfplumber  # kept for Tier 3 fallback inside med_ocr_tool
from tools.med_ocr_tool import med_ocr_tool


def _make_crew(llm: LLM, profile_json: str, target_hint: str | None) -> Crew:
    agent = Agent(
        role="Senior Data Engineer",
        goal=(
            "Review the schema profile of the uploaded dataset. "
            "Confirm or correct the target column candidate. "
            "Flag any data quality concerns. "
            "Return a brief validation note as plain text."
        ),
        backstory=(
            "You are meticulous about data quality. "
            "You always verify schema assumptions before "
            "allowing any downstream analysis."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=1
    )
    task = Task(
        description=(
            f"Schema profile:\n{profile_json}\n\n"
            f"Problem description: {target_hint or 'not provided'}\n\n"
            "In 2 sentences: confirm the target column candidate "
            "and flag any obvious data quality issues."
        ),
        expected_output=(
            "2-sentence validation note. "
            "Example: 'Target column confirmed as churn. "
            "income column has 18% null rate — will require imputation.'"
        ),
        agent=agent
    )
    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)


def _kickoff_with_fallback(profile_json: str, target_hint: str | None) -> str:
    """Try Gemini 2.0 Flash first; fall back to Groq llama-3.3-70b on any error."""
    candidates = [
        LLM(model="gemini/gemini-2.0-flash", api_key=settings.GOOGLE_API_KEY, temperature=0.1),
        LLM(model="groq/llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY, temperature=0.1),
    ]
    last_exc: Exception | None = None
    for llm in candidates:
        try:
            return str(_make_crew(llm, profile_json, target_hint).kickoff())
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"All LLMs failed. Last error: {last_exc}")


async def run_ingest_agent(
    run_id: str,
    run_dir: str,
    target_column: str | None
) -> SchemaProfile:

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_ingest",
        "status": "running",
        "output_summary": "Parsing multi-modal dataset (CSV/FASTA/PDF)...",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    t0 = time.monotonic()
    
    fasta_sequences: list[dict] = []
    pdf_chunks: list[dict] = []     # structured MedOCRResult entities
    pdf_ocr_result = None           # MedOCRResult (written back to profile)
    csv_profile = None
    
    # ── Step 1: Scan directory for all file types ──────────────────
    for filename in os.listdir(run_dir):
        fpath = os.path.join(run_dir, filename)
        if not os.path.isfile(fpath): continue
        
        ext = filename.rsplit(".", 1)[-1].lower()
        
        # 1.1: Parse FASTA
        if ext in ("fasta", "fa"):
            try:
                for record in SeqIO.parse(fpath, "fasta"):
                    fasta_sequences.append({"id": record.id, "seq": str(record.seq)})
            except Exception as e:
                print(f"Error parsing FASTA {filename}: {e}")

        # 1.2: Parse PDF — tiered Med-OCR (Textract Medical → Groq LLM → raw text)
        elif ext == "pdf":
            try:
                ocr_result = med_ocr_tool(pdf_path=fpath, run_id=run_id)
                # Store as list-of-dicts for JSON serialisation
                pdf_chunks = [ent.model_dump() for ent in ocr_result.entities]
                # If raw_text fallback fired, store raw chunks as minimal dicts
                if not pdf_chunks:
                    pdf_chunks = [
                        {"text": c, "category": "RAW_TEXT", "confidence": 1.0}
                        for c in ocr_result.raw_text_chunks
                    ]
                # Attach full OCR result to profile
                pdf_ocr_result = ocr_result
                print(
                    f"[ingest] Med-OCR ({ocr_result.extraction_method}): "
                    f"{ocr_result.entity_count} entities | "
                    f"diagnoses={len(ocr_result.diagnoses)} "
                    f"meds={len(ocr_result.medications)} "
                    f"PHI={'YES' if ocr_result.phi_detected else 'no'}"
                )
            except Exception as e:
                print(f"[ingest] Med-OCR failed for {filename}: {e} — using raw pdfplumber")
                try:
                    with pdfplumber.open(fpath) as pdf_raw:
                        for page in pdf_raw.pages:
                            text = page.extract_text()
                            if text:
                                pdf_chunks.extend(
                                    [{"text": p.strip(), "category": "RAW_TEXT", "confidence": 1.0}
                                     for p in text.split("\n\n") if p.strip()]
                                )
                except Exception as inner:
                    print(f"[ingest] pdfplumber also failed: {inner}")
        
        # 1.3: Parse CSV/Main (Polars Ingest Tool)
        elif ext in ("csv", "xlsx", "xls", "json"):
            if csv_profile is None: # Only one main dataset for now
                csv_profile = ingest_tool(
                    filepath=fpath,
                    run_id=run_id,
                    target_column=target_column
                )

    if csv_profile is None:
        # If no CSV was found, create a dummy profile to hold fasta/pdf
        csv_profile = SchemaProfile(
            run_id=run_id, filename="multi_input", rows=0, cols=0, 
            columns=[], numeric_columns=[], categorical_columns=[], 
            datetime_columns=[], target_candidate=None, null_summary={}, 
            duplicate_count=0, memory_mb=0.0
        )

    # Attach multi-modal data
    csv_profile.fasta_sequences = fasta_sequences
    csv_profile.pdf_chunks = pdf_chunks
    if pdf_ocr_result is not None:
        csv_profile.pdf_ocr_result = pdf_ocr_result

    # ── Compact schema for LLM validation (top-10 cols, no binary blobs) ──
    compact_schema = {
        "filename":         csv_profile.filename,
        "rows":             csv_profile.rows,
        "cols":             csv_profile.cols,
        "target_candidate": csv_profile.target_candidate,
        "columns":          csv_profile.columns[:10],
        "numeric_columns":  csv_profile.numeric_columns[:10],
        "categorical_cols": csv_profile.categorical_columns[:10],
        "duplicate_count":  csv_profile.duplicate_count,
        "fasta_count":      len(fasta_sequences),
        "pdf_entity_count": len(pdf_chunks),
        "pdf_diagnoses":    (pdf_ocr_result.diagnoses[:5] if pdf_ocr_result else []),
        "pdf_medications":  (pdf_ocr_result.medications[:5] if pdf_ocr_result else []),
        "pdf_phi_detected": (pdf_ocr_result.phi_detected if pdf_ocr_result else False),
        "pdf_ocr_method":   (pdf_ocr_result.extraction_method if pdf_ocr_result else "none"),
    }

    # Use LLM to validate schema profile — compact JSON to minimise tokens
    validation_note = _kickoff_with_fallback(
        profile_json=json.dumps(compact_schema, indent=2),
        target_hint=target_column
    )

    latency_ms = int((time.monotonic() - t0) * 1000)

    # Update Supabase runs table
    update_run_status(
        run_id,
        status="ingested",
        rows_count=csv_profile.rows,
        cols_count=csv_profile.cols,
        schema_json=csv_profile.model_dump()
    )

    # ── Med-OCR summary for SSE ───────────────────────────────────────────
    ocr_summary = ""
    if pdf_ocr_result and pdf_ocr_result.entity_count > 0:
        ocr_summary = (
            f" | Med-OCR [{pdf_ocr_result.extraction_method}]: "
            f"{pdf_ocr_result.entity_count} entities, "
            f"{len(pdf_ocr_result.diagnoses)} diagnoses, "
            f"{len(pdf_ocr_result.medications)} medications"
            f"{', PHI DETECTED' if pdf_ocr_result.phi_detected else ''}"
        )

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_ingest",
        "status": "completed",
        "latency_ms": latency_ms,
        "output_summary": (
            f"Ingested {csv_profile.rows} rows, "
            f"{len(fasta_sequences)} proteins, "
            f"{len(pdf_chunks)} PDF entities{ocr_summary}. "
            f"{validation_note}"
        ),
        "data": {
            "rows":              csv_profile.rows,
            "proteins":          len(fasta_sequences),
            "pdf_entities":      len(pdf_chunks),
            "pdf_diagnoses":     pdf_ocr_result.diagnoses[:5] if pdf_ocr_result else [],
            "pdf_medications":   pdf_ocr_result.medications[:5] if pdf_ocr_result else [],
            "pdf_phi_detected":  pdf_ocr_result.phi_detected if pdf_ocr_result else False,
            "pdf_ocr_method":    pdf_ocr_result.extraction_method if pdf_ocr_result else "none",
            "target_candidate":  csv_profile.target_candidate,
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return csv_profile
