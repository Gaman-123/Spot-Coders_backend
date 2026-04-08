from typing import Dict, List

# Subsystem Definitions
SYSTEMS = {
    "ZORA": {
        "name": "Zora",
        "description": "Patient-facing multi-agent pipeline",
        "stages": ["ingest", "clean", "automl", "gnn", "synthesis", "narrator"],
        "langsmith_project": "zora-pipeline"
    },
    "AETHER": {
        "name": "Aether",
        "description": "Knowledge and vector memory layer",
        "stages": ["embed", "pgvector", "rag", "keyword_fallback"],
        "langsmith_project": "aether-memory"
    }
}

def get_system_tag(stage_name: str) -> str:
    """
    Returns the subsystem tag (ZORA or AETHER) for a given stage.
    """
    for system_id, config in SYSTEMS.items():
        if stage_name.lower() in [s.lower() for s in config["stages"]]:
            return system_id
    return "ZORA" # Default

def get_langsmith_project(system_id: str) -> str:
    return SYSTEMS.get(system_id, {}).get("langsmith_project", "zora-pipeline")
