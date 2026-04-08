import uuid
import asyncio
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import ValidationError
from models.schemas import ProteinContext, RunCreateResponse, RunStatusResponse
from services.supabase_service import (
    create_run_record, get_run, update_run_status
)
from agents.zora_ingest import run_ingest_agent
from agents.zora_embed import run_embed_agent
from agents.zora_clean import run_clean_agent
from agents.zora_feature import run_feature_agent
from agents.zora_automl import run_automl_agent
from agents.zora_gnn import run_gnn_agent
from agents.zora_misfold import run_misfold_agent
from agents.zora_synthesis import run_synthesis_agent
from agents.zora_narrator import run_narrator_agent
from services.de_connector import fetch_latest_dataset
from utils.sse_manager import sse_manager
from utils.config import settings
from utils.logger import get_run_logger
from datetime import datetime, timezone

router = APIRouter()


async def _run_pipeline(
    run_id: str,
    filepath: str,
    target_column: str | None,
    enable_protein_analysis: bool,
    protein_context: ProteinContext | None,
):
    """Background task: S1 ingest → S2 embed → S3 clean → S3.5 feature → S4 automl → S5 synthesis → narrator."""
    log = get_run_logger(run_id)
    try:
        update_run_status(run_id, status="running")

        # S1 — Ingest
        log.info("starting_ingest")
        profile = await run_ingest_agent(
            run_id=run_id,
            run_dir=os.path.dirname(filepath) if os.path.isfile(filepath) else filepath,
            target_column=target_column
        )
        log.info("ingest_complete", rows=profile.rows, cols=profile.cols)

        # S2 — Embed
        log.info("starting_embed")
        vector_count = await run_embed_agent(run_id=run_id, profile=profile)
        log.info("embed_complete", vector_count=vector_count)

        # S3 — Clean + Critic Gate 1
        log.info("starting_clean")
        clean_report = await run_clean_agent(run_id=run_id, profile=profile)
        log.info("clean_complete",
                 rows_after=clean_report.rows_after,
                 quality_score=clean_report.quality_score)

        log.info("starting_feature_engineering")
        feature_report = await run_feature_agent(run_id=run_id, profile=profile)
        log.info(
            "feature_engineering_complete",
            feature_columns=feature_report.feature_columns,
            derived_features=len(feature_report.derived_features_added),
        )

        # S4 — AutoML + AlphaFold + RAG Critic Gate 1
        log.info("starting_automl")
        s4_result = await run_automl_agent(
            run_id=run_id,
            profile=profile,
            explicit_protein_context=protein_context,
        )
        log.info("automl_complete",
                 model=s4_result["automl"]["model_name"],
                 auc=s4_result["automl"]["metrics"]["auc"])

        if enable_protein_analysis:
            log.info("starting_misfold")
            misfold_summary = await run_misfold_agent(
                run_id=run_id,
                s4_result=s4_result,
                enable_protein_analysis=enable_protein_analysis,
            )
            if misfold_summary:
                s4_result["misfold"] = misfold_summary.model_dump()
                log.info("misfold_complete",
                         stuck_score=misfold_summary.stuck_score,
                         energy_state=misfold_summary.energy_state)

        # S4.5 — GNN Network Analysis (New Stage)
        log.info("starting_gnn")
        gnn_result = await run_gnn_agent(
            run_id=run_id,
            automl_result=s4_result
        )
        log.info("gnn_complete", unique_interactors=len(gnn_result))

        # S5 — Synthesis (finance + safety + RAG + LLM)
        log.info("starting_synthesis")
        synthesis_result = await run_synthesis_agent(
            run_id=run_id,
            profile=profile,
            clean_report=clean_report,
            s4_result=s4_result,
            gnn_result=gnn_result,
        )
        log.info("synthesis_complete", insight_id=synthesis_result.get("insight_id"))

        # Narrator — dual voice + G2 Critic Gate 2
        log.info("starting_narrator")
        narrator_result = await run_narrator_agent(
            run_id=run_id,
            profile=profile,
            synthesis_result=synthesis_result,
        )
        log.info("narrator_complete",
                 g2_score=narrator_result["g2_score"],
                 g2_passed=narrator_result["g2_passed"])
        # pipeline_complete SSE emitted by run_narrator_agent

    except Exception as e:
        log.error("pipeline_failed", error=str(e))
        update_run_status(run_id, status="failed")
        await sse_manager.publish(run_id, {
            "type": "error",
            "agent": "pipeline",
            "status": "failed",
            "error_message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        # Send pipeline_complete so SSE stream closes
        await sse_manager.publish(run_id, {
            "type": "pipeline_complete",
            "agent": "pipeline",
            "status": "failed",
            "error_message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    finally:
        pass  # SSE cleanup handled by subscriber after pipeline_complete


@router.post("/run", response_model=RunCreateResponse)
async def create_run(
    file: UploadFile = File(None),
    fasta_file: UploadFile = File(None),
    pdf_file: UploadFile = File(None),
    mode: str = Form("upload"),
    problem_desc: str | None = Form(None),
    target_column: str | None = Form(None),
    enable_protein_analysis: bool = Form(False),
    protein_context_json: str | None = Form(None),
):
    run_id = uuid.uuid4().hex[:12]
    run_dir = os.path.join(settings.UPLOAD_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    final_filepath = ""
    filename = "default_set"

    if mode == "pipeline":
        # Pull from DE pipeline via Supabase
        pipeline_path = fetch_latest_dataset(source_name="default")
        if not pipeline_path:
            raise HTTPException(status_code=404, detail="No dataset found in DE pipeline")
        
        # In reality, we'd copy it to run_dir, but for now we'll just use it
        final_filepath = pipeline_path
        filename = os.path.basename(pipeline_path)
    else:
        # Standard upload flow
        if not file and not fasta_file and not pdf_file:
            raise HTTPException(status_code=400, detail="At least one file must be uploaded in upload mode")
        
        if file:
            filename = file.filename or "upload.csv"
            final_filepath = os.path.join(run_dir, filename)
            content = await file.read()
            with open(final_filepath, "wb") as f:
                f.write(content)
        
        if fasta_file:
            fasta_path = os.path.join(run_dir, fasta_file.filename or "input.fasta")
            content = await fasta_file.read()
            with open(fasta_path, "wb") as f:
                f.write(content)
            if not final_filepath: final_filepath = fasta_path
            
        if pdf_file:
            pdf_path = os.path.join(run_dir, pdf_file.filename or "input.pdf")
            content = await pdf_file.read()
            with open(pdf_path, "wb") as f:
                f.write(content)
            if not final_filepath: final_filepath = pdf_path

    # Validate file type if CSV/JSON
    if mode == "upload" and file:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ("csv", "xlsx", "xls", "json"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported main file type: .{ext}. Accepted: csv, xlsx, xls, json"
            )

    protein_context: ProteinContext | None = None
    if protein_context_json:
        try:
            protein_context = ProteinContext.model_validate_json(
                protein_context_json
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid protein_context_json: {exc.errors()}",
            ) from exc

    # Create Supabase run record
    create_run_record(
        run_id=run_id,
        filename=filename,
        filepath=filepath,
        problem_desc=problem_desc,
        target_column=target_column,
        protein_context_json=(
            protein_context.model_dump(exclude_none=True)
            if protein_context else None
        ),
    )

    # Launch pipeline in background
    asyncio.create_task(
        _run_pipeline(
            run_id,
            final_filepath,
            target_column,
            enable_protein_analysis,
            protein_context,
        )
    )

    return RunCreateResponse(
        run_id=run_id,
        status="queued",
        filename=filename,
        mode=mode
    )


@router.get("/run/{run_id}/status", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunStatusResponse(**run)
