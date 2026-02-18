import json
import pickle
import re
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from src.config import CHUNKS_JSONL, INDEX_DIR, BM25_PATH

TOKEN_RE=re.compile(r"[A-Za-z0-9_]+")

def tokenize(text:str):
    return TOKEN_RE.findall(text.lower())

def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    corpus_tokens=[]
    chunk_ids=[]

    with CHUNKS_JSONL.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="building bm25 corpus"):
            rec=json.loads(line)
            chunk_ids.append(rec["chunk_id"])
            corpus_tokens.append(tokenize(rec["text"]))
    
    bm25=BM25Okapi(corpus_tokens)

    with BM25_PATH.open("wb") as f:
        pickle.dump(
            {
                "bm25":bm25,
                "chunk_ids":chunk_ids,
            },
            f
        )

    print(f"BM25 index saved:{BM25_PATH}")
    print(f"BM25 Documents: {len(chunk_ids)}")

if __name__=="__main__":
    main()