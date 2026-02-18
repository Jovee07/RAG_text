from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.models_api import AskRequest, AskResponse, SourceItem
from src.rag_core import ask
from src.logging_utils import setup_logging

logger=setup_logging()


app=FastAPI(title="tex RAG serv", version="0.1")

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

@app.get("/health")
def health():
    return {"status":"ok"}


@app.post("/ask", response_model=AskResponse)

def ask_endpoint(req:AskRequest):
    logger.info(f"Ask query_len-{len(req.query)} top_k={req.top_k} filters={req.filters}")

    result=ask(req.query, top_k=req.top_k, filter=req.filters)

    sources=[]

    for h in result["sources"]:
        sources.append(SourceItem(
            chunk_id=h["chunk_id"],
            doc_id=h.get("doc_id", ""),
            title=h.get("title"),
            source_type=h.get("source_type"),
            source_path=h.get("source_path"),
            score=h["score"],
        ))

    return AskResponse(
        answer=result["answer"],
        sources=sources,
        latency_ms=result["latency_ms"],
        refused=result["refused"],

    )
