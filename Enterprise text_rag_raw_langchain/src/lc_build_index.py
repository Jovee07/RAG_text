from json import load
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import ROOT

load_dotenv()

ROOT=Path(__file__).resolve().parents[1]

PDF_DIR=ROOT/"data"/"raw"/"pdfs"

INDEX_DIR=ROOT/"data"/"lc_index"



def main():
    # 1) Load PDFs -> LangChain Documents (each has page_content + metadata)
    loader=PyPDFDirectoryLoader(str(PDF_DIR))
    docs=loader.load()

    # 2) Split -> chunks (LangChain Documents)
    splitter=RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks=splitter.split_documents(docs)

     # 3) Embeddings + FAISS index

    embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
    vs=FAISS.from_documents(chunks, embeddings)

    #4) Save Locally
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_DIR))


    print(f"Loaded pages:{len(docs)}")
    print(f"Chunks created:{len(chunks)}")
    print(f"Saved FAISS Index to:{INDEX_DIR}")

if __name__=="__main__":
    main()


# loader = PyPDFDirectoryLoader(str(PDF_DIR))
# docs=loader.load()

# splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
# chunks=splitter.split_documents(docs)

# embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
# vs=FAISS.from_documents(chunks, embeddings)


# INDEX_DIR.mkdir(parents=True, exist_ok=True)
# vs.save_local(str(INDEX_DIR))

# print(f"Loaded pages:{len(docs)}")
# print(f"Chunks created: {len(chunks)}")
# print(f"Saved faiss indes : {INDEX_DIR}")

# if __name__=="__main__":
#     main()