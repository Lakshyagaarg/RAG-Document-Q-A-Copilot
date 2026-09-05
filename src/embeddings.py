from pathlib import Path
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from ingestion import load_pdfs, split_documents


# ============================================================
# CONFIGURATION
# ============================================================

PDF_DIRECTORY = Path("data/pdfs")

CHROMA_DIRECTORY = "chroma_db"

COLLECTION_NAME = "technical_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# 1. CREATE EMBEDDING MODEL
# ============================================================

def create_embedding_model():

    print("\n" + "=" * 70)
    print("LOADING EMBEDDING MODEL")
    print("=" * 70)

    print(f"Model: {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    print("Embedding model loaded successfully.")

    return embeddings


# ============================================================
# 2. CREATE VECTOR DATABASE
# ============================================================

def create_vector_database(chunks, embeddings):

    print("\n" + "=" * 70)
    print("CREATING CHROMADB VECTOR DATABASE")
    print("=" * 70)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIRECTORY,
    )

    return vector_store


# ============================================================
# 3. TEST VECTOR SEARCH
# ============================================================

def test_vector_search(vector_store):

    test_query = "What is Retrieval-Augmented Generation?"

    print("\n" + "=" * 70)
    print("TESTING SEMANTIC SEARCH")
    print("=" * 70)

    print(f"Query: {test_query}")

    results = vector_store.similarity_search(
        test_query,
        k=3
    )

    print(f"Retrieved chunks: {len(results)}")

    for i, result in enumerate(results):

        print(f"\n--- Result {i + 1} ---")

        print("Source:")
        print(result.metadata.get("source"))

        print("Page:")
        print(result.metadata.get("page"))

        print("\nContent:")
        print(result.page_content)

        print("-" * 70)


# ============================================================
# 4. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("RAG EMBEDDING + VECTOR DATABASE PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load PDFs
    # Remove old ChromaDB before rebuilding
    # --------------------------------------------------------

    if Path(CHROMA_DIRECTORY).exists():

        print("\nRemoving existing ChromaDB...")

        shutil.rmtree(CHROMA_DIRECTORY)

        print("Old ChromaDB removed.")

    documents = load_pdfs(PDF_DIRECTORY)

    print("\nTotal pages loaded:", len(documents))

    # --------------------------------------------------------
    # Split documents using Phase 1 function
    # --------------------------------------------------------

    chunks = split_documents(documents)

    print("Total chunks created:", len(chunks))

    # --------------------------------------------------------
    # Create embedding model
    # --------------------------------------------------------

    embeddings = create_embedding_model()

    # --------------------------------------------------------
    # Store chunks + embeddings in ChromaDB
    # --------------------------------------------------------

    vector_store = create_vector_database(
        chunks,
        embeddings
    )

    # --------------------------------------------------------
    # Verify database
    # --------------------------------------------------------

    collection = vector_store._collection

    print("\n" + "=" * 70)
    print("VECTOR DATABASE CREATED SUCCESSFULLY")
    print("=" * 70)

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored vectors: {collection.count()}")
    print(
        f"Database directory: "
        f"{Path(CHROMA_DIRECTORY).resolve()}"
    )

    # --------------------------------------------------------
    # Test semantic search
    # --------------------------------------------------------

    test_vector_search(vector_store)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()