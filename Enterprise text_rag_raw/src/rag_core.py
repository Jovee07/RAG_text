import time
import pickle
import numpy as np
import faiss
from openai import OpenAI
from typing import List, Dict, Any, Optional

from src.config import (
    FAISS_INDEX_PATH, META_PATH, CHUNK_TEXT_PATH, EMBED_MODEL, GEN_MODEL
)

SIMILARITY_THRESHOLD = 0.65
MAX_CHUNKS_PER_DOC=2

client=OpenAI()

def load_assets():
    """Load assets + meta + texts, cache in memory for API speed"""

    global _assets_cache
    if _assets_cache is not None:
        return _assets_cache
    
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    with META_PATH.open('rb') as f:
        metas=pickle.load(f)
    
    with CHUNK_TEXT_PATH.open("rb") as f:
        texts=pickle.load(f)

    
    _asset_cache= (index, metas, texts)

    return _asset_cache

# def load_assets():
#     # Check Redis version
#     redis_version = r.get("assets_version")

#     # Get S3 version (e.g., stored in a small file or metadata)
#     s3_version_obj = s3.get_object(Bucket=bucket, Key="version.txt")
#     s3_version = s3_version_obj["Body"].read().decode()

#     if redis_version == s3_version:
#         # Redis is up-to-date
#         index = pickle.loads(r.get("faiss_index"))
#         metas = pickle.loads(r.get("metas"))
#         texts = pickle.loads(r.get("texts"))
#         return index, metas, texts

#     # Otherwise reload from S3
#     s3.download_file(bucket, "faiss.index", "faiss.index")
#     s3.download_file(bucket, "metas.pkl", "metas.pkl")
#     s3.download_file(bucket, "texts.pkl", "texts.pkl")

#     index = faiss.read_index("faiss.index")
#     with open("metas.pkl", "rb") as f:
#         metas = pickle.load(f)
#     with open("texts.pkl", "rb") as f:
#         texts = pickle.load(f)

#     # Update Redis with new data + version
#     r.set("faiss_index", pickle.dumps(index))
#     r.set("metas", pickle.dumps(metas))
#     r.set("texts", pickle.dumps(texts))
#     r.set("assets_version", s3_version)

#TTl Time-to-Live (TTL) Expiry Method"
#r.setex("faiss_index", 3600, pickle.dumps(index))

#Manual Invalidation
#r.delete("faiss_index", "metas", "texts", "assets_version")




def normalize(v: np.ndarray)->np.ndarray:
    n=np.linalg.norm(v,axis=1, keepdims=True)+1e-12
    return v/n

def embed_query(q:str)-> np.ndarray:
    emb=client.embeddings.create(model=EMBED_MODEL, input=[q])

    v=np.array([emb.data[0].embedding], dtype="float32")

    return normalize(v)

def apply_filters(meta: Dict[str, Any], filters:Optional[Dict[str, any]])->bool:

    """
    Return True if this meta record is allowed.
    For now supports: doc_id, source_type, source
    """

    if not filters:
        return True
    
    for key in ["doc_id","source_type","source"]:
        if key in filters and meta.get(key)!=filters[key]:
            return False
    return True

def retrieve(q: str, top_k:int, filters:Optional[Dict[str, Any]]=None)-> List[Dict[str, Any]]:
    index, metas, texts=load_assets()

    v=embed_query(q)

    k_search=min(max(top_k*4, 10), 60)
    scores, idxs = index.search(v, k_search)

    hits:List[Dict[str, Any]]=[]
    doc_counter:Dict[str, int]={}


    for score, idx in zip(scores[0], idxs[0]):
        score = float(score)

        if score<SIMILARITY_THRESHOLD:
            continue

        m=metas[int(idx)]

        if not apply_filters(m, filters):
            continue
        doc_id=m.get("doc_id", "unknown")

        if doc_counter.get(doc_id, 0) >=MAX_CHUNKS_PER_DOC:
            continue


        chunk_id=m["chunk_id"]

        text=texts.get(chunk_id,"")

        hits.append({
            **m,
            "score":score,
            "text":text,
        })

        doc_counter[doc_id]=doc_counter.get(doc_id,0)+1

        if len(hits)>=top_k:
            break
    return hits

def build_context(hits:List[Dict[str, Any]])->str:
    blocks=[]
    for h in hits:
        blocks.append(
            f"SOURCE_ID:{h['chunk_id']}\n"
            f"DOC:{h['doc_id']}\n"
            f"PATH:{h.get('source_path')}\n"
            f"EXCERPT:\n{h['text']}\n"
        )
    return "\n\n---\n\n".join(blocks)

def generate_answer(q:str, hits:List[Dict[str, Any]])->str:
    context=build_context(hits)

    prompt=f"""
You are a compliance-focused assistant.
Answer ONLY using the provided SOURCES. Do not use outside knowledge.
If the sources do not contain enough information, say: "I don't know based on the provided sources."

QUESTION:
{q}

SOURCES:
{context}
""".strip()

    res=client.responses.create(model=GEN_MODEL, input=prompt)
    return res.output_text


def ask(q:str, top_k:int, filters: Optional[Dict[str, Any]]=None)->Dict[str, Any]:

    """
    orchestrates: retrieve->refusal-> generate
    """

    t0=time.time()

    hits=retrieve(q, top_k=top_k, filters=filters)

    if not hits:

        latency_ms=int((time.time()-t0)*1000)

        return{
            "answer":"i dont know based on the provided soyrces",
            "sources":[],
            "latency_ms":latency_ms,
            "refused":True,
        }
    
    answer=generate_answer(q,hits)

    latency_ms=int((time.time()-t0)*1000)

    return{
            "answer":answer,
            "sources":hits,
            "latency_ms":latency_ms,
            "refused":False,        
    }

