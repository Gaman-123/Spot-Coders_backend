import time
import logging
import asyncio
from typing import Optional
from google import genai as google_genai
from google.genai import types as genai_types
from utils.config import settings

log = logging.getLogger(__name__)

def get_embedding(text: str) -> Optional[list[float]]:
    """
    Synchronous wrapper for embedding with 429 handling and retries.
    Used for legacy tool compatibility.
    """
    # Use a dummy event loop if one isn't running to call the async version
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return None # Should not call sync from inside async loop usually
    except RuntimeError:
        pass
        
    return asyncio.run(get_embedding_async(text))

async def get_embedding_async(text: str) -> Optional[list[float]]:
    """
    Get embedding with 429 handling and retries.
    Returns None if all retries fail.
    """
    if not settings.GOOGLE_API_KEY:
        log.error("GOOGLE_API_KEY not set")
        return None
        
    client = google_genai.Client(api_key=settings.GOOGLE_API_KEY)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Check for non-empty text to avoid API errors
            text_to_embed = text.strip() if text else ""
            if not text_to_embed:
                return [0.0] * 768
                
            resp = client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=[text_to_embed],
                config=genai_types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=768,
                ),
            )
            return resp.embeddings[0].values
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "quota" in err_msg.lower():
                wait = (attempt + 1) * 3
                log.warning(f"Embedding Quota Exceeded (429). Attempt {attempt+1}/{max_retries}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
                continue
            
            log.error(f"Embedding failed with non-quota error: {e}")
            break
            
    log.error(f"Embedding failed after {max_retries} attempts due to quota.")
    return None

async def get_embeddings_batch_async(texts: list[str]) -> list[Optional[list[float]]]:
    """
    Batch process embeddings with 429 handling.
    """
    if not settings.GOOGLE_API_KEY:
        return [None] * len(texts)
        
    client = google_genai.Client(api_key=settings.GOOGLE_API_KEY)
    
    try:
        resp = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=texts,
            config=genai_types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            ),
        )
        return [emb.values for emb in resp.embeddings]
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            log.warning("Batch embedding hit 429. Falling back to serial processing with retries.")
            results = []
            for t in texts:
                results.append(await get_embedding_async(t))
            return results
        log.error(f"Batch embedding failed: {e}")
        return [None] * len(texts)
