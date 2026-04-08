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
import pdfplumber


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
    
    fasta_sequences = []
    pdf_chunks = []
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

        # 1.2: Parse PDF
        elif ext == "pdf":
            try:
                with pdfplumber.open(fpath) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            # Simple chunking by paragraph
                            pdf_chunks.extend([p.strip() for p in text.split("\n\n") if p.strip()])
            except Exception as e:
                print(f"Error parsing PDF {filename}: {e}")
        
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

    # Use LLM to validate schema profile — Gemini primary, Groq fallback
    validation_note = _kickoff_with_fallback(
        profile_json=json.dumps(csv_profile.model_dump(), indent=2),
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

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_ingest",
        "status": "completed",
        "latency_ms": latency_ms,
        "output_summary": (
            f"Ingested {csv_profile.rows} rows, {len(fasta_sequences)} proteins, {len(pdf_chunks)} text chunks. "
            f"{validation_note}"
        ),
        "data": {
            "rows": csv_profile.rows,
            "proteins": len(fasta_sequences),
            "pdf_paragraphs": len(pdf_chunks),
            "target_candidate": csv_profile.target_candidate
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return csv_profile
