from typing import List, Dict, Any

PIPELINE_STAGES = [
    {
        "id": "S1",
        "name": "Ingest",
        "subsystem": "ZORA",
        "agent_class": "run_ingest_agent",
        "tools": ["ingest_tool", "fasta_parser", "pdf_parser"],
        "writes_to_supabase": ["runs", "raw_datasets"],
        "sse_event_label": "zora_ingest"
    },
    {
        "id": "S2",
        "name": "Embed",
        "subsystem": "AETHER",
        "agent_class": "run_embed_agent",
        "tools": ["gemini_embedding", "pgvector"],
        "writes_to_supabase": ["documents"],
        "sse_event_label": "zora_embed"
    },
    {
        "id": "S3",
        "name": "Clean",
        "subsystem": "ZORA",
        "agent_class": "run_clean_agent",
        "tools": ["clean_tool"],
        "writes_to_supabase": ["runs"],
        "sse_event_label": "zora_clean"
    },
    {
        "id": "S4",
        "name": "AutoML + Biophysics + GNN",
        "subsystem": "ZORA",
        "agent_class": "run_automl_pipeline", # Or separate agent calls
        "tools": ["alphafold_tool", "sasa_tool", "gnn_agent", "pycaret"],
        "writes_to_supabase": ["gnn_results"],
        "sse_event_label": "zora_automl"
    },
    {
        "id": "Gate1",
        "name": "Hallucination Check",
        "subsystem": "ZORA",
        "agent_class": "run_critic_agent",
        "tools": ["llm_critic"],
        "writes_to_supabase": ["runs"],
        "sse_event_label": "zora_critic"
    },
    {
        "id": "S5",
        "name": "Synthesis",
        "subsystem": "ZORA",
        "agent_class": "run_synthesis_agent",
        "tools": ["risk_tool", "finance_tool", "safety_vault"],
        "writes_to_supabase": ["insights"],
        "sse_event_label": "zora_synthesis"
    },
    {
        "id": "S6",
        "name": "Narrator",
        "subsystem": "ZORA",
        "agent_class": "run_narrator_agent",
        "tools": ["llm_narrator"],
        "writes_to_supabase": ["insights"],
        "sse_event_label": "zora_narrator"
    },
    {
        "id": "Gate2",
        "name": "Quality Check",
        "subsystem": "ZORA",
        "agent_class": "run_critic_agent",
        "tools": ["llm_critic"],
        "writes_to_supabase": ["insights"],
        "sse_event_label": "zora_critic"
    },
    {
        "id": "S7",
        "name": "Delivery",
        "subsystem": "ZORA",
        "agent_class": "run_delivery_agent",
        "tools": ["twilio", "telegram"],
        "writes_to_supabase": ["notifications"],
        "sse_event_label": "zora_delivery"
    }
]

def get_stage_by_id(stage_id: str) -> Dict[str, Any]:
    for stage in PIPELINE_STAGES:
        if stage["id"] == stage_id:
            return stage
    return None
