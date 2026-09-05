from pathlib import Path
import sys


# Allow Python to import files from the src directory.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))


from retrieval import (
    create_embedding_model,
    load_vector_database,
    retrieve_documents,
)


# Test questions and expected source documents.
TEST_CASES = [
    (
        "What is Retrieval-Augmented Generation?",
        "01_rag_fundamentals.pdf",
    ),
    (
        "What are the main components of RAG architecture?",
        "02_rag_architecture.pdf",
    ),
    (
        "How can Python be used to build a RAG system?",
        "03_python_rag_engineering.pdf",
    ),
    (
        "What is the purpose of text chunking in a RAG pipeline?",
        "05_text_chunking.pdf",
    ),
    (
        "Why is chunk overlap used when splitting documents?",
        "08_chunk_overlap.pdf",
    ),
    (
        "What are embeddings in a RAG system?",
        "10_embeddings.pdf",
    ),
    (
        "What is cosine similarity used for in semantic search?",
        "14_cosine_similarity.pdf",
    ),
    (
        "What is a vector database?",
        "15_vector_databases.pdf",
    ),
    (
        "What is ChromaDB used for?",
        "16_chromadb.pdf",
    ),
    (
        "Why is vector database persistence useful?",
        "18_persistent_vector_stores.pdf",
    ),
    (
        "What is similarity search?",
        "19_similarity_search.pdf",
    ),
    (
        "How does top-k retrieval work?",
        "20_top_k_retrieval.pdf",
    ),
    (
        "What is hybrid retrieval?",
        "22_hybrid_retrieval.pdf",
    ),
    (
        "What does hallucination control combine?",
        "27_hallucination_control.pdf",
    ),
    (
        "What is retrieval recall?",
        "31_retrieval_recall.pdf",
    ),
]


# Questions that should not be answerable from the knowledge base.
UNSUPPORTED_CASES = [
    "What is the capital of France?",
    "Who was the first person to walk on the Moon?",
    "What is the boiling point of water?",
    "Who invented the telephone?",
    "What is the population of India?",
]


def get_retrieved_sources(results):
    """
    Extract unique PDF filenames while
    preserving retrieval order.
    """

    sources = []

    for document, _ in results:
        source = Path(
            document.metadata.get(
                "source",
                "",
            )
        ).name

        if source not in sources:
            sources.append(source)

    return sources


def calculate_reciprocal_rank(
    expected_source,
    retrieved_sources,
):
    """
    Calculate the reciprocal rank of the
    expected source.
    """

    for rank, source in enumerate(
        retrieved_sources,
        start=1,
    ):
        if source == expected_source:
            return 1 / rank

    return 0.0


def evaluate_unsupported_questions(
    vector_store,
):
    """
    Evaluate questions that should not be
    answerable from the knowledge base.
    """

    print("\n" + "=" * 70)
    print("UNSUPPORTED QUESTION EVALUATION")
    print("=" * 70)

    for question in UNSUPPORTED_CASES:

        results = retrieve_documents(
            query=question,
            vector_store=vector_store,
            top_k=3,
        )

        print("\n" + "-" * 70)
        print(f"Question: {question}")

        for rank, (
            document,
            distance,
        ) in enumerate(
            results,
            start=1,
        ):
            source = Path(
                document.metadata.get(
                    "source",
                    "",
                )
            ).name

            print(
                f"Rank {rank}: "
                f"{source} | "
                f"Distance: {distance:.4f}"
            )


def main():
    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    # Load the embedding model.
    embeddings = create_embedding_model()

    # Load the existing ChromaDB.
    vector_store = load_vector_database(
        embeddings
    )

    top_1_correct = 0
    top_3_correct = 0
    reciprocal_ranks = []

    for question, expected_source in TEST_CASES:

        results = retrieve_documents(
            query=question,
            vector_store=vector_store,
            top_k=3,
        )

        retrieved_sources = get_retrieved_sources(
            results
        )

        reciprocal_rank = (
            calculate_reciprocal_rank(
                expected_source,
                retrieved_sources,
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank
        )

        if (
            retrieved_sources
            and retrieved_sources[0]
            == expected_source
        ):
            top_1_correct += 1

        if expected_source in retrieved_sources:
            top_3_correct += 1

        print("\n" + "-" * 70)
        print(f"Question: {question}")
        print(f"Expected: {expected_source}")
        print(
            f"Retrieved: {retrieved_sources}"
        )

        if (
            retrieved_sources
            and retrieved_sources[0]
            == expected_source
        ):
            print("Top-1: PASS")
        else:
            print("Top-1: FAIL")

        if expected_source in retrieved_sources:
            print("Top-3: PASS")
        else:
            print("Top-3: FAIL")

        print(
            f"Reciprocal Rank: "
            f"{reciprocal_rank:.2f}"
        )

    total = len(TEST_CASES)

    top_1_accuracy = (
        top_1_correct / total
    ) * 100

    top_3_accuracy = (
        top_3_correct / total
    ) * 100

    mean_reciprocal_rank = (
        sum(reciprocal_ranks) / total
    )

    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    print(
        f"Top-1 Accuracy: "
        f"{top_1_correct}/{total} "
        f"= {top_1_accuracy:.1f}%"
    )

    print(
        f"Top-3 Accuracy: "
        f"{top_3_correct}/{total} "
        f"= {top_3_accuracy:.1f}%"
    )

    print(
        f"Mean Reciprocal Rank (MRR): "
        f"{mean_reciprocal_rank:.2f}"
    )

    # Evaluate unsupported questions.
    evaluate_unsupported_questions(
        vector_store
    )

    print("=" * 70)


if __name__ == "__main__":
    main()