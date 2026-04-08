import time
import logging
from google import genai as google_genai
from google.genai import types as genai_types
from utils.config import settings

log = logging.getLogger(__name__)

def get_embedding(text: str) -> list[float] | None:
    """
    Get embedding with 429 handling and retries.
    Returns None if all retries fail.
    """
    if not settings.GOOGLE_API_KEY:
        return None
        
    client = google_genai.Client(api_key=settings.GOOGLE_API_KEY)
    
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            resp = client.models.embed_content(
                model="models/embedding-001",
                contents=[text],
                config=genai_types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=768,
                ),
            )
            return resp.embeddings[0].values
        except Exception as e:
            if "429" in str(e):
                if attempt < max_retries:
                    wait = (attempt + 1) * 2
                    log.warning(f"Embedding 429. Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
            log.error(f"Embedding failed: {e}")
            break
            
    return None
