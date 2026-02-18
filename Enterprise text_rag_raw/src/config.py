# folder paths

# file paths

# model names

# chunk sizes

# retrieval settings


import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

RAW_PDFS_DIR=ROOT/"data"/"raw"/"pdfs"
PROCESSED_DIR=ROOT/"data"/"processed"
INDEX_DIR=ROOT/"data"/"index"

PDF_DOCS_JSONL=PROCESSED_DIR/"pdf_docs.jsonl"
CHUNKS_JSONL=PROCESSED_DIR/"chunks.jsonl"

FAISS_INDEX_PATH=INDEX_DIR/"faiss.index"
META_PATH=INDEX_DIR/"meta.pkl"
CHUNK_TEXT_PATH=INDEX_DIR/"chunk_text.pkl"
BM25_PATH=INDEX_DIR/"bm25.pkl"

EMBED_MODEL=os.getenv("EMBED_MDOEL","text-embedding-3-small")
GEN_MODEL=os.getenv("GEN MODEL","gpt-4.1-mini")

MAX_TOKENS=420
OVERLAP_TOKENS=60

BATCH_EMBED=128
TOP_K=6

