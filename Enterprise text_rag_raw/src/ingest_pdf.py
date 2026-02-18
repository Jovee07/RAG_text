
# Purpose in pipeline

# Ingestion + normalization

# Input:

# PDFs in data/raw/pdfs/

# Output:

# data/processed/pdf_docs.jsonl (one JSON per PDF)

import json
from pathlib import Path
from pypdf import PdfReader
from tqdm import tqdm
from src.config import RAW_PDFS_DIR, PROCESSED_DIR, PDF_DOCS_JSONL

def extract_pdf_text(pdf_path:Path)->str:
    reader=PdfReader(str(pdf_path))
    parts=[]
    for page in reader.pages:
        txt=page.extract_text()or""
        txt=txt.strip()
        if txt:
            parts.append(txt)
    return "\n\n".join(parts)

def main():
    PROCESSED_DIR.mkdir(parents=True,exist_ok=True)
    RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)

    pdfs=sorted(RAW_PDFS_DIR.glob("*.pdf"))

    if not pdfs:
        raise SystemError(f"No Pdfs were found")
    
    with PDF_DOCS_JSONL.open("w",encoding="utf-8") as f:
        for pdf in tqdm(pdfs,desc="ingestion PDFs"):
            text=extract_pdf_text(pdf)
            if not text.strip():
                continue

            rec={
                "source_type":"pdf",
                "doc_id":pdf.stem,
                "title":pdf.stem,
                "source":"local_pdfs",
                "source_path":str(pdf),
                "text":text,
            }
            f.write(json.dumps(rec,ensure_ascii=False)+"\n")
    print(f"wrote:{PDF_DOCS_JSONL}")

if __name__=="__main__":
    main()