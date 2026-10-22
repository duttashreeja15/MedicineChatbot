
from pathlib import Path
import json
from src.ingest_and_chunking import ingest_and_chunk_pdf
from src.embeddings import embed_documents
from src.retrieval_and_generation import ask_pdf

question = ["What is the mechanism of action of metformin?",
            "What are the common side effects of metformin?",
            "How does metformin affect glucose production in the liver?",
            "What is the role of AMPK in metformin's mechanism of action?",
            "What is the recommended dosage of metformin for a 5-year-old child?",
            "What is the recipe for biryani?"]



def main():
    results = []
    base_dir = Path(__file__).resolve().parent
    pdf_path = base_dir / "data" / "documents" / "metformin literature.pdf"
    print(f"PDF path: {str(pdf_path)}")
    

    print("Step 1: Ingesting & chunking documents...")
    chunks, pages = ingest_and_chunk_pdf(str(pdf_path))
    total_pages = len(pages)
    print(f"Total pages: {total_pages}")
    print(f"Loaded {len(chunks)} chunks")
    
    
    print("Step 2: Embedding documents...")
    chunk_count=embed_documents(chunks)
    print(f"Stored {chunk_count} chunks in chroma_db")
    

    print("Step 3: Retrieve and Generate...")
    
    for q in question:
        print(f"Question: {q}")
        answer,context = ask_pdf(q, top_k=5)
        print(f"Answer: {answer}")
        
        results.append({
            "question": q,
            "answer": answer,
            "context": context,
        })
    
    output_path = base_dir / "evaluation" / "final_output.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, ensure_ascii=False, indent=2)
    
    print("Pipeline completed!")

if __name__ == "__main__":
    main()
