import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

from retrieval import (
    create_embedding_model,
    load_vector_database,
    retrieve_documents,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.6-flash"

TOP_K = 3


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 1. CREATE GEMINI CLIENT
# ============================================================

def create_gemini_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found.\n"
            "Please add it to your .env file."
        )

    client = genai.Client(
        api_key=api_key
    )

    return client


# ============================================================
# 2. BUILD CONTEXT FROM RETRIEVED DOCUMENTS
# ============================================================

def build_context(documents):

    context_parts = []

    for i, document in enumerate(documents):

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page",
            "Unknown page"
        )

        context_parts.append(
            f"""
SOURCE {i + 1}
File: {source}
Page: {page}

Content:
{document.page_content}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# 3. CREATE RAG PROMPT
# ============================================================

def create_rag_prompt(context, question):

    return f"""
You are a technical document Q&A assistant.

Answer the user's question using ONLY the information provided in the context.

Rules:
1. Answer directly without introductory phrases.
2. Do not start with phrases such as:
   - "Based on the provided documents"
   - "Based on the context"
   - "According to the documents"
   - "The provided documents state"
3. Do not mention that you were given context or documents unless necessary to explain that the information is unavailable.
4. Do not use outside knowledge.
5. If the answer cannot be found in the context, say exactly:
   "The provided documents do not contain enough information to answer this question."
6. Keep the answer concise and technically accurate.

Context:
{context}

Question:
{question}

Answer directly:
"""


# ============================================================
# 4. GENERATE ANSWER USING GEMINI
# ============================================================

def generate_answer(client, prompt):

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=prompt,
    )

    return interaction.output_text


# ============================================================
# 5. COMPLETE RAG PIPELINE
# ============================================================

def answer_question(question, vector_store, client):
    results = retrieve_documents(
        query=question,
        vector_store=vector_store,
        top_k=TOP_K,
    )

    documents = [document for document, _ in results]

    context = build_context(documents)
    prompt = create_rag_prompt(context=context, question=question)
    answer = generate_answer(client=client, prompt=prompt)

    return answer, documents


# ============================================================
# 6. MAIN TEST
# ============================================================

def main():

    print("=" * 70)
    print("RAG GENERATION PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Create embedding model
    # --------------------------------------------------------

    embeddings = create_embedding_model()

    # --------------------------------------------------------
    # Load ChromaDB
    # --------------------------------------------------------

    vector_store = load_vector_database(
        embeddings
    )

    # --------------------------------------------------------
    # Create Gemini client
    # --------------------------------------------------------

    client = create_gemini_client()

    # --------------------------------------------------------
    # Test question
    # --------------------------------------------------------

    question = (
    "What is Retrieval-Augmented Generation?"
    )

    print("\nQuestion:")
    print(question)

    # --------------------------------------------------------
    # Run RAG
    # --------------------------------------------------------

    answer, documents = answer_question(
        question=question,
        vector_store=vector_store,
        client=client,
    )

    # --------------------------------------------------------
    # Display answer
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GEMINI ANSWER")
    print("=" * 70)

    print(answer)

    # --------------------------------------------------------
    # Display sources
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SOURCES USED")
    print("=" * 70)

    for i, document in enumerate(documents):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        source_name = Path(source).name
        page_number = page + 1 if isinstance(page, int) else page

        print(f"{i + 1}. {source_name} (Page {page_number})")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()