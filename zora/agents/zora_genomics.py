import os
import tarfile
import time
from datetime import datetime, timezone
from models.schemas import GenomicsReport
from utils.sse_manager import sse_manager
from services.supabase_service import update_run_status

GENOMICS_DIR = "data/genomics"

async def run_genomics_agent(run_id: str, automl_result: dict) -> GenomicsReport:
    """
    S4.2 — Zora Genomics Agent: Metadata & Multi-Omic Linking.
    Identifies relevant local GEO datasets (e.g. GSE5281 for Alzheimer's)
    and links genomic sample metadata to the current clinical run.
    """
    t0 = time.monotonic()
    
    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_genomics",
        "status": "running",
        "output_summary": "Searching local genomic archives (GEO/NCBI) for multi-omic context...",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    # 1. Identify Target Diseases/Proteins from AutoML
    found_proteins = list(automl_result.get("automl", {}).get("top_features", {}).keys())
    
    # 2. Match with Local Genomic Data
    # For now, we specifically handle GSE5281 (Alzheimer's) if related proteins (APP) are found
    is_alzheimers = any(p in ["APP", "SNCA", "MAPT"] for p in found_proteins)
    
    report = GenomicsReport(
        run_id=run_id,
        source_tar="none",
        status="skip_no_match"
    )

    tar_path = os.path.join(GENOMICS_DIR, "GSE5281_RAW.tar")
    
    if os.path.exists(tar_path):
        report.source_tar = "GSE5281_RAW.tar"
        try:
            with tarfile.open(tar_path, "r") as tar:
                members = tar.getmembers()
                report.sample_count = len(members)
                
            if is_alzheimers:
                report.genomic_disease_tags = ["Alzheimer's Disease", "Brain Expression Profiling"]
                report.status = "matched"
                summary = f"Matched {report.sample_count} genomic samples from Alzheimer's dataset (GSE5281)."
            else:
                report.status = "loaded_metadata"
                summary = f"Found local GEO dataset {report.source_tar} with {report.sample_count} samples. No direct clinical match."
        except Exception as e:
            report.status = "error"
            summary = f"Error reading genomic archive: {str(e)}"
    else:
        summary = "No local genomic datasets found matching current clinical target."

    latency_ms = int((time.monotonic() - t0) * 1000)
    
    # Update Supabase
    update_run_status(run_id, status="genomics_linked")

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_genomics",
        "status": "completed" if report.status != "error" else "failed",
        "latency_ms": latency_ms,
        "output_summary": summary,
        "data": report.model_dump(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return report
