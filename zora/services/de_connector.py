from services.supabase_service import get_supabase

def fetch_latest_dataset(source_name: str) -> str | None:
    """
    Queries the raw_datasets table written by the DE pipeline
    and returns the storage_path of the most recently ingested dataset.
    """
    supabase = get_supabase()
    try:
        response = (
            supabase.table("raw_datasets")
            .select("storage_path")
            .eq("source_name", source_name)
            .order("ingested_at", desc=True)
            .limit(1)
            .execute()
        )
        
        if response.data and len(response.data) > 0:
            return response.data[0].get("storage_path")
            
    except Exception as e:
        print(f"Error fetching latest dataset for {source_name}: {e}")
        
    return None
