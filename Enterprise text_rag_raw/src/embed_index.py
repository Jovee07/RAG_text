#  Purpose

# Convert each chunk to embeddings and build searchable index.

# Input:

# chunks.jsonl

# Output:

# faiss.index (vector index)

# meta.pkl (metadata list)

# chunk_text.pkl (chunk_id → chunk text map)

import os
import json
import pickle
import numpy as np
import faiss
from tqdm import tqdm
from openai import OpenAI
from src.config import (INDEX_DIR, CHUNKS_JSONL, FAISS_INDEX_PATH, META_PATH, CHUNK_TEXT_PATH, EMBED_MODEL, BATCH_EMBED)
from dotenv import load_dotenv

load_dotenv()
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
client = OpenAI()


def normalize(mat:np.ndarray)->np.ndarray:
    n=np.linalg.norm(mat,axis=1, keepdims=True)+1e-12
    return mat/n

def main():

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    metas=[]
    texts={}
    vectors=[]

    batch_text=[]
    batch_meta=[]


    with CHUNKS_JSONL.open("r",encoding="utf-8") as f:
        for line in tqdm(f,desc="embedding results"):
            rec=json.loads(line)
            batch_text.append(rec["text"])
            batch_meta.append({k:rec[k] for k in ["chunk_id","doc_id","title","source","source_type","source_path"]})
            texts[rec["chunk_id"]]=rec["text"]


            if len(batch_text)>=BATCH_EMBED:
                emb=client.embeddings.create(model=EMBED_MODEL,input=batch_text)
                arr = np.array([e.embedding for e in emb.data],dtype="float32")

                vectors.append(arr)
                metas.extend(batch_meta)
                batch_text,batch_meta=[],[]
        if batch_text:
            emb=client.embeddings.create(model=EMBED_MODEL, input=batch_text)
            arr=np.array([e.embedding for e in emb.data], dtype="float32")

            vectors.append(arr)
            metas.extend(batch_meta)

    mat=normalize(np.vstack(vectors))

    index=faiss.IndexFlatIP(mat.shape[1])

    index.add(mat)
    
    faiss.write_index(index,str(FAISS_INDEX_PATH))
    with META_PATH.open("wb") as f:
        pickle.dump(metas,f)
    with CHUNK_TEXT_PATH.open("wb") as f:
        pickle.dump(texts,f)

    print("indexed vectors:", index.ntotal)
    print("index saved to",FAISS_INDEX_PATH)

if __name__=="__main__":
    main()



# import os
# import json
# import faiss
# import pickle
# import numpy as np
# from tqdm import tqdm
# from openai import OpenAI

# from src.config import (INDEX_DIR, CHUNKS_JSONL, FAISS_INDEX_PATH, META_PATH, CHUNK_TEXT_PATH, EMBED_MODEL, BATCH_EMBED)

# from dotenv import load_dotenv

# load_dotenv()


# client = OpenAI()

# def normalize(mat:np.ndarray)->np.ndarray:
#     n=np.linalg.norm(mat, axis=1, keepdims=True)+1e-12
#     return mat/n

# def main():
#     INDEX_DIR.mkdir(parents=True, exist_ok=True)
#     metas=[]
#     texts=[]
#     vectors=[]

#     # Vector DB = vectors + metadata + content

#     batch_text=[]
#     batch_meta=[]

#     with CHUNKS_JSONL.open("r",encoding="utf-8") as f:

#         for line in tqdm(f, desc="embeding chunks"):
#             rec = json.loads(line)

#             batch_text.append(rec["text"])
#             batch_meta.append({k:rec[k] for k in[
#                 "chunk_id",
#                 "doc_id",
#                 "title",
#                 "source",
#                 "source_type",
#                 "source_path"
#             ]})

#             texts[rec["chunk_id"]]=rec["text"]

#             if len(batch_text)>=BATCH_EMBED:
#                 emb=client.embeddings.create(
#                     model=EMBED_MODEL,
#                     input=batch_text
#                 )

#                 arr=np.array([e.embedding for e in emb.data], dtype="float32")

#                 vectors.append(arr)
#                 metas.extend(batch_meta)

#         if batch_text:
#             mat=normalize(np.vstack(vectors))
#             emb=client.embeddings.create(
#                 model=EMBED_MODEL,
#                 input=batch_text
#             )

#             arr=np.array([e.embedding for e in emb.data], dtype="float32")

#             vectors.append(arr)
#             metas.extend(batch_meta)

#     index=faiss.IndexFlatIP(mat.shape[1])

#     index.add(mat)

#     faiss.write_index(index, str(FAISS_INDEX_PATH))

#     with META_PATH.open("wb") as f:
#         pickle.dump(metas, f)

#     with CHUNK_TEXT_PATH.open("wb") as f:
#         pickle.dump(texts, f)

#     print("indexed vectors:",index.ntotal)

# if __name__=="__main__":
#     main()

