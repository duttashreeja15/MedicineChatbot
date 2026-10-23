import chromadb
from sentence_transformers import SentenceTransformer
import os


def embed_documents(chunks):
    
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [chunk.page_content for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    #print(embeddings.shape)  # (number_of_chunks, 384)

    # Save the database in a local folder
    client = chromadb.PersistentClient(
    path=os.getenv("CHROMA_PATH", "data/chroma_db")
)

    # We supply our own embeddings from all-MiniLM-L6-v2
    collection = client.get_or_create_collection(
        name="metformin_minilm",
        embedding_function=None,
    )

    assert len(chunks) == len(embeddings), "Each chunk needs one embedding."

    # Store in batches to avoid oversized requests
    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))

        collection.upsert(
            ids=[f"metformin_chunk_{i}" for i in range(start, end)],
            embeddings=embeddings[start:end].tolist(),
            documents=[chunk.page_content for chunk in chunks[start:end]],
            metadatas=[chunk.metadata for chunk in chunks[start:end]],
        )

    #print(f"Stored {collection.count()} chunks in chroma_db")
    return collection.count()