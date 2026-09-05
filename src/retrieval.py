from collections import Counter
import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ChromaDB configuration
CHROMA_DIRECTORY = "chroma_db"
COLLECTION_NAME = "technical_documents"

# Same embedding model used during indexing
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Number of final chunks sent to the generation stage
TOP_K = 3

# Retrieve more candidates before ranking
CANDIDATE_MULTIPLIER = 5

# Maximum number of chunks from one source
MAX_CHUNKS_PER_SOURCE = 2


def create_embedding_model():
    """
    Create the embedding model used for semantic search.
    """

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    print("Embedding model loaded.")

    return embeddings


def load_vector_database(embeddings):
    """
    Load the existing ChromaDB vector database.
    """

    print("Loading existing ChromaDB...")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIRECTORY,
        embedding_function=embeddings,
    )

    print("ChromaDB loaded successfully.")

    return vector_store


def get_query_words(query):
    """
    Extract meaningful words from the query.

    Very short/common words are ignored because they
    provide little information for retrieval.
    """

    stop_words = {
        "what",
        "is",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "to",
        "for",
        "does",
        "do",
        "why",
        "how",
        "can",
        "and",
        "or",
        "are",
        "this",
        "that",
    }

    words = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        query.lower(),
    )

    return [
        word
        for word in words
        if word not in stop_words
    ]


def calculate_keyword_score(query, document):
    """
    Calculate lexical relevance between the query
    and a document chunk.

    A higher score means more query words are present
    in the document chunk.
    """

    query_words = get_query_words(query)

    if not query_words:
        return 0.0

    document_text = (
        document.page_content.lower()
    )

    matches = sum(
        word in document_text
        for word in query_words
    )

    return matches / len(query_words)


def calculate_phrase_score(query, document):
    """
    Give an additional score when multiple important
    query words occur close together in the chunk.

    This helps identify precise answers instead of
    generic chunks containing only some query words.
    """

    query_words = get_query_words(query)

    if len(query_words) < 2:
        return 0.0

    document_text = (
        document.page_content.lower()
    )

    # Check whether the important query words
    # appear together in the same text window.
    for i in range(
        len(query_words) - 1
    ):
        first_word = query_words[i]
        second_word = query_words[i + 1]

        pattern = (
            rf"\b{re.escape(first_word)}\b"
            rf".{{0,80}}"
            rf"\b{re.escape(second_word)}\b"
        )

        if re.search(
            pattern,
            document_text,
        ):
            return 0.20

    return 0.0


def calculate_combined_score(
    query,
    document,
    distance,
):
    """
    Combine semantic similarity with lexical relevance.

    Lower semantic distance is better.

    Keyword and phrase scores provide additional
    signals for precise questions.
    """

    keyword_score = calculate_keyword_score(
        query,
        document,
    )

    phrase_score = calculate_phrase_score(
        query,
        document,
    )

    # Convert similarity signals into a single
    # ranking score where lower is better.
    combined_score = (
        distance
        - keyword_score
        - phrase_score
    )

    return combined_score


def retrieve_documents(
    query: str,
    vector_store,
    top_k: int = TOP_K,
):
    """
    Retrieve relevant document chunks.

    Steps:
    1. Retrieve semantic candidates.
    2. Calculate lexical relevance.
    3. Calculate phrase relevance.
    4. Rank candidates using the combined score.
    5. Remove duplicate chunks.
    6. Apply source diversity.
    7. Return the final top-k chunks.
    """

    # Retrieve a larger candidate set first.
    candidate_count = (
        top_k * CANDIDATE_MULTIPLIER
    )

    candidate_results = (
        vector_store.similarity_search_with_score(
            query,
            k=candidate_count,
        )
    )

    ranked_results = []

    for document, distance in candidate_results:

        # Calculate the final ranking score.
        combined_score = calculate_combined_score(
            query,
            document,
            distance,
        )

        ranked_results.append(
            (
                document,
                distance,
                combined_score,
            )
        )

    # Lower combined score means better relevance.
    ranked_results.sort(
        key=lambda item: item[2]
    )

    results = []

    # Track duplicate chunks.
    seen_chunks = set()

    # Track number of chunks selected per source.
    source_counts = Counter()

    for (
        document,
        distance,
        combined_score,
    ) in ranked_results:

        # Remove exact duplicate chunks.
        chunk_key = (
            document.page_content.strip()
        )

        if chunk_key in seen_chunks:
            continue

        # Identify the source PDF.
        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        # Limit repeated chunks from one source.
        if (
            source_counts[source]
            >= MAX_CHUNKS_PER_SOURCE
        ):
            continue

        # Add the chunk to the final results.
        results.append(
            (
                document,
                distance,
            )
        )

        seen_chunks.add(chunk_key)

        source_counts[source] += 1

        # Stop after collecting top-k chunks.
        if len(results) >= top_k:
            break

    return results


def display_results(query, results):
    """
    Display retrieval results for debugging.
    """

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    print(f"\nQuery: {query}")
    print(
        f"Retrieved chunks: {len(results)}"
    )

    for i, (
        document,
        distance,
    ) in enumerate(results):

        print(
            f"\n--- Result {i + 1} ---"
        )

        print(
            f"Distance: {distance:.4f}"
        )

        print("Source:")
        print(
            document.metadata.get(
                "source",
                "Unknown source",
            )
        )

        print("Page:")
        print(
            document.metadata.get(
                "page",
                "Unknown page",
            )
        )

        print("\nContent:")
        print(
            document.page_content
        )

        print("-" * 70)


def run_tests(vector_store):
    """
    Run predefined retrieval tests.
    """

    test_queries = [
        "What is Retrieval-Augmented Generation?",
        "What is the purpose of text chunking in a RAG pipeline?",
        "Why is vector database persistence useful?",
        "What does hallucination control combine?",
        "What is the capital of France?",
    ]

    for query in test_queries:

        results = retrieve_documents(
            query=query,
            vector_store=vector_store,
            top_k=TOP_K,
        )

        display_results(
            query=query,
            results=results,
        )


def main():
    """
    Load ChromaDB and run retrieval tests.
    """

    print("=" * 70)
    print("RAG RETRIEVAL PIPELINE")
    print("=" * 70)

    # Load the same embedding model used during indexing.
    embeddings = create_embedding_model()

    # Load the existing vector database.
    vector_store = load_vector_database(
        embeddings
    )

    # Check the number of stored vectors.
    collection = vector_store._collection

    print(
        f"\nVectors available in database: "
        f"{collection.count()}"
    )

    # Run retrieval tests.
    run_tests(vector_store)


if __name__ == "__main__":
    main()