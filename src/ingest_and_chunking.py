
import os
from pypdf import PdfReader
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

#pdf_path = "../data/documents/metformin literature.pdf"


def ingest_and_chunk_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=200,       # Maximum tokens per chunk
        chunk_overlap=40,     # Token overlap between adjacent chunks
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents = splitter.create_documents(
            texts=[text],
            metadatas=[{
                "source": str(pdf_path),
                "page": page_number,
            }],
        )

        chunks.extend(documents)
    
    return chunks,reader.pages