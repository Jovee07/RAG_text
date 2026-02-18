import json

from pathlib import Path
import tiktoken
from src.config import PROCESSED_DIR, PDF_DOCS_JSONL, CHUNKS_JSONL, MAX_TOKENS, OVERLAP_TOKENS

ENC = tiktoken.get_encoding("cl100k_base")

# def chunk_tokens(text:str, max_tokens:int, overlap:int):
#     toks=ENC.encode(text)
#     out=[]
#     start=0
#     while start<len(toks):
#         end = min(start+max_tokens,len(toks))
#         out.append(ENC.decode(toks[start:end]))

#         if start==len(toks):
#             break

#         start=max(0,end-overlap)
#     return out

#replace with stream chunk as the above approch requires more RAM for the giant token list hence the below stream chunking will help on the easy processing

def read_jsonl(path:Path):
    if not path.exists():
        return
    with path.open("r",encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def stream_chunk_text(text:str, max_tokens:int, overlap:int):

    """
    Memory safe chunking:
    Split text into small pieces
    tokenize piece by piece
    maintain a rolling  token buffer
    """

    pieces=[p for p in text.split("\n\n") if p.strip()]
    buffer=[]

    for p in pieces:
        ptoks=ENC.encode(p)

        if len(ptoks)>max_tokens*2:
            for line in p.splitlines():
                ltoks=ENC.encode(line)

                buffer.extend(ltoks)

                while len(buffer)>=max_tokens:
                    chunk=buffer[:max_tokens]
                    yield ENC.decode(chunk)

                    buffer = buffer[max_tokens-overlap:]

            continue


        buffer.extend(ptoks)

        while len(buffer)>=max_tokens:
            chunk=buffer[:max_tokens]
            yield ENC.decode(chunk)

            buffer=buffer[max_tokens-overlap:]

    if buffer:
        yield ENC.decode(buffer)

def main():
    PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

    with CHUNKS_JSONL.open("w", encoding="utf-8") as fout:
        for doc in read_jsonl(PDF_DOCS_JSONL):
            chunks=stream_chunk_text(doc["text"],MAX_TOKENS, OVERLAP_TOKENS)
            for i,ch in enumerate(chunks):
                rec={
                    "chunk_id":f'{doc["doc_id"]}"::pdf::{i}',
                    "doc_id":doc["doc_id"],
                    "title":doc["title"],
                    "source":doc["source"],
                    "source_type":doc["source_type"],
                    "source_path":doc.get("source_path",""),
                    "text":ch,
                }
                fout.write(json.dumps(rec, ensure_ascii=False)+"\n")
    print(f"wrote chunks:{CHUNKS_JSONL}")


if __name__=="__main__":
    main()