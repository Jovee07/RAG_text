from pathlib import Path
from tkinter.messagebox import QUESTION
from click import prompt
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from numpy import block
from openai import embeddings

from src.config import ROOT

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR=ROOT/"data"/"lc_index"

SIMILARITY_THRESHOLD=0.65
MAX_CHUNKS_PER_DOC=2
TOP_K=6

def build_sources_text(hits):
    blocks=[]

    for h in hits:
        meta=h["metadata"]

        blocks.append(
            f"SOURCE: {meta.get("source",'')}, page={meta.get('page','')}\n"
            f"SCORE:{h['score']:.4f}\n"
            f"{h['text']}"
        )
    return "\n\n--\n\n".join(blocks)

def main():
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
    vs=FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)

    # retriever = vs.as_retriever(search_kwargs={"k":TOP_K})

    llm=ChatOpenAI(model="gpt-4.1-mini")

    prompt=ChatPromptTemplate.from_messages([
        (
            "system",
            "answer only using the provided sources"
            "if not found, say: 'I dont know based on the provided sources"
        ),
        ("user",
         "Questions:\n{question}\n\nSources:\n{sources}")
    ])

    while True:
        q=input("\n Ask or exit").strip()

        if q.lower()=="exit":
            break

        docs_scores=vs.similarity_search_with_score(q, k=TOP_K*4)

        hits=[]
        doc_counter={}

        for doc, score in docs_scores:
            score=float(score)

            if score < SIMILARITY_THRESHOLD:
                continue

            doc_id=doc.metadata.get("source","unknown_doc")

            if doc_counter.get(doc_id, 0)>=MAX_CHUNKS_PER_DOC:
                continue

            hits.append({
                "metadata":doc.metadata,
                "score":score,
                "text":doc.page_content
            })
            
            doc_counter[doc_id]=doc_counter.get(doc_id,0)+1

            if len(hits)>=TOP_K:
                break

            if not hits:
                print("i dont know based on the provided sources")
                continue

            sources_text=build_sources_text(hits)
            msgs=prompt.format_messages(question=q, sources=sources_text)
            resp=llm.invoke(msgs)
            
            print(resp.content)



        # docs=retriever.invoke(q)

        # if not docs:
        #     print(" I dont know based on the provided sourcecs")
        #     continue

        # sources_text="\n\n----\n\n".join(
        #     f'source={d.metadata.get('source','')}, page={d.metadata.get('page','')}\n{d.page_content}'
        #     for d in docs
        # )

        # msgs=prompt.format_messages(question=q, sources=sources_text)
        # resp=llm.invoke(msgs)
        # print(resp.content)


if __name__=="__main__":
    main()