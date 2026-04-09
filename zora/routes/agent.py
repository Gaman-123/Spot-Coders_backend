from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from services.supabase_service import get_supabase
from utils.config import settings
from google import genai as google_genai
from google.genai import types as genai_types

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return dot_product / (norm_v1 * norm_v2)

def _get_research_citations(query: str, k: int = 3) -> list[str]:
    """Simple keyword search through local PMC metadata CSV."""
    import os, csv
    csv_path = "data/knowledge/research_metadata.csv"
    if not os.path.exists(csv_path):
        return []
    
    keywords = [w.lower() for w in query.split() if len(w) > 3]
    if not keywords:
        return []
        
    citations = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (row.get("Article Citation", "") + " " + row.get("License", "")).lower()
                if any(kw in text for kw in keywords):
                    citations.append(f"- {row.get('Article Citation')} (PMC ID: {row.get('Accession ID')})")
                    if len(citations) >= k:
                        break
    except Exception:
        pass
    return citations


@router.post("/query")
async def query_agent(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    supabase = get_supabase()
    
    # 1. Scope to a single demo patient
    # Fetch the latest run that has embeddings
    runs_res = supabase.table("runs").select("*").order("created_at", desc=True).limit(20).execute()
    runs = runs_res.data
    
    # Find the first run that has an embedding count > 0 (or fallback to the very first one)
    target_run_id = None
    if not runs:
        raise HTTPException(status_code=404, detail="No clinical runs found to search.")
        
    for run in runs:
        # Assuming the run has some indication it finished, or just pick the latest!
        if run.get("status") in ["queued", "running", "failed"]:
            continue
        target_run_id = run["run_id"]
        break
        
    if not target_run_id:
        target_run_id = runs[0]["run_id"] # Fallback

    # 2. Embed the user's query
    from services.embedding_service import get_embedding_async
    query_embedding = await get_embedding_async(request.query)
    if not query_embedding:
        raise HTTPException(status_code=429, detail="Embedding API quota exceeded. Please try again later.")

    # 3. Fetch documents for this run
    docs_res = supabase.table("documents").select("*").eq("run_id", target_run_id).execute()
    documents = docs_res.data
    
    if not documents:
        # If no documents, we can still ask Gemini to answer based on general knowledge or state ignorance
        top_chunks_text = "No clinical documents found for this patient."
    else:
        # 4. Perform vector similarity search in Python
        scored_docs = []
        for doc in documents:
            doc_emb = doc.get("embedding")
            if not doc_emb:
                continue
            sim = cosine_similarity(query_embedding, doc_emb)
            scored_docs.append((sim, doc))
            
        # Sort by similarity desc
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Take Top-5 chunks
        top_k = 5
        top_chunksText = []
        for sim, doc in scored_docs[:top_k]:
            top_chunksText.append(f"Document Chunk: {doc.get('chunk_text', '')}")
            
        top_chunks_text = "\n\n".join(top_chunksText)
        if not top_chunks_text:
            top_chunks_text = "No document chunks available for this patient."

    # 5. Fetch local research citations for broader context
    research_citations = _get_research_citations(request.query)
    research_context = "\n".join(research_citations) if research_citations else "No relevant research papers found in local PMC index."

    # 6. Query Clinical Hub Agent (Groq primary)
    from crewai import Agent, Task, Crew, Process, LLM
    
    system_prompt = (
        "You are a clinical AI assistant for a physician dashboard explicitly focused on ONE patient's data.\n"
        "Use the provided clinical dataset contexts AND research citations to answer the user's question accurately.\n"
        "If the answer is not in the context, clearly state that you do not know based on the provided documents.\n\n"
        f"--- CLINICAL CONTEXT START ---\n{top_chunks_text}\n--- CLINICAL CONTEXT END ---\n\n"
        f"--- RELEVANT RESEARCH PAPERS ---\n{research_context}\n---------------------------------"
    )

    candidates = [
        LLM(model="groq/llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY, temperature=0.2),
        LLM(model="gemini/gemini-2.0-flash", api_key=settings.GOOGLE_API_KEY, temperature=0.2),
    ]
    
    answer_text = None
    last_err = None
    for llm in candidates:
        try:
            agent = Agent(
                role="Clinical Informatics Specialist",
                goal="Provide concise, evidence-based answers to physician queries.",
                backstory="Expert in interpreting clinical data chunks retrieved via RAG.",
                llm=llm, verbose=False, allow_delegation=False, max_iter=1
            )
            task = Task(description=f"{system_prompt}\n\nQuestion: {request.query}", 
                        expected_output="Direct answer to the clinical question.", 
                        agent=agent)
            answer_text = str(Crew(agents=[agent], tasks=[task]).kickoff())
            break
        except Exception as e:
            last_err = e
            continue
            
    if not answer_text:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer after model rotation: {last_err}")

    return {
        "answer": answer_text,
        "run_id_scoped": target_run_id
    }
