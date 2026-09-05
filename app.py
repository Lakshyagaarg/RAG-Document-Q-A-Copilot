import sys
from pathlib import Path
import hashlib
import tempfile

import streamlit as st
from langchain_chroma import Chroma


# Add src directory to Python path
SRC_DIR = Path("src")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from retrieval import (
    create_embedding_model,
    load_vector_database,
    retrieve_documents,
)

from ingestion import process_uploaded_pdf

from generation import (
    create_gemini_client,
    build_context,
    create_rag_prompt,
    generate_answer,
)


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="RAG Document Copilot",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------
# Application header
# ---------------------------------------------------------

st.title("📚 RAG Document Copilot")

st.write(
    "Ask questions from technical documents using "
    "Retrieval-Augmented Generation."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    st.write("**Embedding Model**")
    st.code("all-MiniLM-L6-v2")

    st.write("**Vector Database**")
    st.code("ChromaDB")

    st.write("**Retrieval**")
    st.code("Top 3 chunks")

    st.write("**LLM**")
    st.code("Gemini 3.6 Flash")


# ---------------------------------------------------------
# Load RAG components
# ---------------------------------------------------------

@st.cache_resource
def load_rag_components():
    embeddings = create_embedding_model()

    vector_store = load_vector_database(
        embeddings
    )

    gemini_client = create_gemini_client()

    return (
        embeddings,
        vector_store,
        gemini_client,
    )


try:
    with st.spinner("Loading RAG system..."):
        (
            embeddings,
            vector_store,
            gemini_client,
        ) = load_rag_components()

    st.success("RAG system loaded successfully.")

except Exception as e:
    st.error(
        f"Failed to initialize RAG system:\n\n{e}"
    )
    st.stop()


# ---------------------------------------------------------
# Uploaded PDF vector store
# ---------------------------------------------------------

@st.cache_resource
def create_uploaded_vector_store(
    pdf_bytes,
    filename,
    _embeddings,
):
    file_hash = hashlib.sha256(
        pdf_bytes
    ).hexdigest()[:12]

    collection_name = (
        f"uploaded_pdf_{file_hash}"
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as temp_file:

        temp_file.write(pdf_bytes)

        temp_pdf_path = Path(
            temp_file.name
        )

    try:
        documents, chunks = process_uploaded_pdf(
            temp_pdf_path
        )

        if not chunks:
            raise ValueError(
                "No extractable text was found "
                "in this PDF. Please upload a "
                "text-based PDF."
            )

        # Store original filename in metadata.
        for chunk in chunks:
            chunk.metadata["source"] = filename

        uploaded_vector_store = (
            Chroma.from_documents(
                documents=chunks,
                embedding=_embeddings,
                collection_name=collection_name,
            )
        )

        return (
            uploaded_vector_store,
            len(documents),
            len(chunks),
        )

    finally:
        temp_pdf_path.unlink(
            missing_ok=True
        )


# ---------------------------------------------------------
# Knowledge base statistics
# ---------------------------------------------------------

pdf_directory = Path("data/pdfs")

pdf_files = sorted(
    pdf_directory.glob("*.pdf")
)

document_count = len(pdf_files)

collection = vector_store._collection

vector_count = collection.count()


# Calculate unique indexed pages from ChromaDB metadata.
metadata = collection.get(
    include=["metadatas"]
).get("metadatas", [])

indexed_pages = {
    (
        item.get("source"),
        item.get("page"),
    )
    for item in metadata
    if item.get("source") is not None
}

page_count = len(indexed_pages)


# ---------------------------------------------------------
# Document topic mapping
# ---------------------------------------------------------

TOPICS = [
    {
        "icon": "🧠",
        "name": "RAG Fundamentals",
        "documents": [
            "01_rag_fundamentals.pdf",
            "02_rag_architecture.pdf",
            "03_python_rag_engineering.pdf",
            "24_context_construction.pdf",
        ],
    },
    {
        "icon": "📄",
        "name": "Document Processing",
        "documents": [
            "05_text_chunking.pdf",
            "06_recursive_chunking.pdf",
            "08_chunk_overlap.pdf",
            "52_document_ingestion.pdf",
            "53_pdf_processing.pdf",
        ],
    },
    {
        "icon": "🔢",
        "name": "Embeddings & Vector Databases",
        "documents": [
            "10_embeddings.pdf",
            "11_sentence_transformers.pdf",
            "12_embedding_dimensions.pdf",
            "13_embedding_normalization.pdf",
            "14_cosine_similarity.pdf",
            "15_vector_databases.pdf",
            "16_chromadb.pdf",
            "17_faiss.pdf",
            "18_persistent_vector_stores.pdf",
            "51_embeddings_vector_databases.pdf",
        ],
    },
    {
        "icon": "🔍",
        "name": "Retrieval",
        "documents": [
            "09_document_metadata.pdf",
            "19_similarity_search.pdf",
            "20_top_k_retrieval.pdf",
            "21_retrieval_thresholds.pdf",
            "22_hybrid_retrieval.pdf",
            "23_reranking.pdf",
            "31_retrieval_recall.pdf",
            "35_query_rewriting.pdf",
            "36_multi_query_retrieval.pdf",
            "38_metadata_filtering.pdf",
        ],
    },
    {
        "icon": "⚡",
        "name": "RAG Optimization",
        "documents": [
            "04_rag_evaluation_cases.pdf",
            "07_chunk_size_optimization.pdf",
            "25_grounded_generation.pdf",
            "26_prompt_engineering_for_rag.pdf",
            "27_hallucination_control.pdf",
            "28_citation_design.pdf",
            "29_answer_abstention.pdf",
            "30_rag_evaluation.pdf",
            "32_answer_faithfulness.pdf",
            "33_evaluation_datasets.pdf",
            "34_regression_testing.pdf",
            "42_rag_observability.pdf",
            "43_caching_in_rag.pdf",
            "44_latency_optimization.pdf",
            "48_rag_error_handling.pdf",
        ],
    },
    {
        "icon": "🚀",
        "name": "Advanced RAG",
        "documents": [
            "37_conversation_aware_rag.pdf",
            "39_access_control_in_rag.pdf",
            "40_prompt_injection_in_rag.pdf",
            "41_pii_protection.pdf",
            "45_local_rag_deployment.pdf",
            "46_streamlit_rag_applications.pdf",
            "47_environment_configuration.pdf",
            "49_rag_production_checklist.pdf",
            "50_future_rag_improvements.pdf",
        ],
    },
]


# ---------------------------------------------------------
# Create topic-to-document lookup
# ---------------------------------------------------------

topic_documents = {
    topic["name"]: topic["documents"]
    for topic in TOPICS
}


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "kb_answer" not in st.session_state:
    st.session_state.kb_answer = None

if "kb_sources" not in st.session_state:
    st.session_state.kb_sources = []

if "upload_answer" not in st.session_state:
    st.session_state.upload_answer = None

if "upload_sources" not in st.session_state:
    st.session_state.upload_sources = []

if "upload_hash" not in st.session_state:
    st.session_state.upload_hash = None

if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = None


# ---------------------------------------------------------
# Main tabs
# ---------------------------------------------------------

knowledge_tab, upload_tab = st.tabs(
    [
        "📚 Technical RAG Knowledge Base",
        "📤 Upload PDF",
    ]
)


# =========================================================
# KNOWLEDGE BASE TAB
# =========================================================

with knowledge_tab:

    st.subheader(
        "📚 Technical RAG Knowledge Base"
    )

    st.write(
        "Explore the topics covered by the "
        "technical document knowledge base."
    )


    # -----------------------------------------------------
    # Knowledge base statistics
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📄 Documents",
        document_count,
    )

    col2.metric(
        "📑 Pages",
        page_count,
    )

    col3.metric(
        "🧩 Indexed Chunks",
        vector_count,
    )

    col4.metric(
        "🔎 Top-K",
        3,
    )


    # -----------------------------------------------------
    # Topics
    # -----------------------------------------------------

    st.divider()

    st.subheader("📚 Topics Covered")

    st.write(
        "Select a topic to view the related documents."
    )


    for row_start in range(
        0,
        len(TOPICS),
        2,
    ):

        col_left, col_right = st.columns(2)

        row_topics = TOPICS[
            row_start:row_start + 2
        ]


        for column, topic in zip(
            [col_left, col_right],
            row_topics,
        ):

            topic_name = topic["name"]
            icon = topic["icon"]

            pdf_list = topic_documents[
                topic_name
            ]


            with column:

                st.write(
                    f"{icon} **{topic_name}**"
                )

                st.caption(
                    f"{len(pdf_list)} document(s)"
                )


                if st.button(
                    "View Documents →",
                    key=f"topic_{topic_name}",
                    use_container_width=True,
                ):

                    if (
                        st.session_state.selected_topic
                        == topic_name
                    ):
                        st.session_state.selected_topic = None

                    else:
                        st.session_state.selected_topic = (
                            topic_name
                        )

                    st.rerun()


        st.write("")


    # -----------------------------------------------------
    # Selected topic documents
    # -----------------------------------------------------

    selected_topic = (
        st.session_state.selected_topic
    )


    if selected_topic:

        st.divider()

        selected_icon = next(
            topic["icon"]
            for topic in TOPICS
            if topic["name"] == selected_topic
        )

        st.subheader(
            f"{selected_icon} {selected_topic}"
        )

        st.caption(
            "Documents included in this topic"
        )

        selected_documents = (
            topic_documents[selected_topic]
        )


        if selected_documents:

            for pdf_name in selected_documents:
                st.write(
                    f"📄 {pdf_name}"
                )

        else:

            st.info(
                "No documents are currently "
                "assigned to this topic."
            )


    # -----------------------------------------------------
    # Knowledge base Q&A
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "💬 Ask the Knowledge Base"
    )


    with st.form(
        "knowledge_question_form"
    ):

        knowledge_question = st.text_input(
            "Enter your question:",
            placeholder=(
                "e.g. What is Retrieval-Augmented Generation?"
            ),
        )

        knowledge_submitted = (
            st.form_submit_button(
                "🔍 Ask Copilot",
                type="primary",
            )
        )


    if knowledge_submitted:

        # Clear the previous answer.
        st.session_state.kb_answer = None
        st.session_state.kb_sources = []

        if not knowledge_question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                # Retrieve relevant chunks.
                try:

                    results = retrieve_documents(
                        query=knowledge_question,
                        vector_store=vector_store,
                        top_k=3,
                    )

                except Exception as e:

                    st.error(
                        "⚠️ Unable to search the "
                        f"documents: {e}"
                    )

                    st.stop()


                if not results:

                    st.warning(
                        "No relevant information was "
                        "found in the knowledge base."
                    )

                    st.stop()


                # Extract documents.
                documents = [
                    document
                    for document, _ in results
                ]


                # Build context.
                context = build_context(
                    documents
                )


                # Create grounded prompt.
                prompt = create_rag_prompt(
                    context=context,
                    question=knowledge_question,
                )


                # Generate answer.
                try:

                    answer = generate_answer(
                        client=gemini_client,
                        prompt=prompt,
                    )

                    st.session_state.kb_answer = (
                        answer
                    )

                    st.session_state.kb_sources = (
                        results
                    )


                except Exception as e:

                    error_message = str(e)


                    if (
                        "429" in error_message
                        or "quota"
                        in error_message.lower()
                        or "rate limit"
                        in error_message.lower()
                    ):

                        st.error(
                            "⚠️ Gemini API quota exceeded. "
                            "Your current Gemini Free Tier limit has been reached. "
                            "Please try again after the quota resets or check your Gemini API billing and limits."
                        )


                    elif (
                        "connection"
                        in error_message.lower()
                        or "timeout"
                        in error_message.lower()
                        or "timed out"
                        in error_message.lower()
                    ):

                        st.error(
                            "⚠️ Unable to connect to "
                            "the Gemini API. "
                            "Please check your internet "
                            "connection and try again."
                        )


                    else:

                        st.error(
                            "⚠️ Unable to generate an "
                            "answer right now. "
                            "Please try again later."
                        )

                    st.stop()


    # -----------------------------------------------------
    # Display knowledge base answer
    # -----------------------------------------------------

    if st.session_state.kb_answer:

        st.subheader("🤖 Answer")

        with st.container(border=True):

            st.write(
                st.session_state.kb_answer
            )

        if st.button(
            "🗑️ Clear Answer",
            key="clear_kb_answer",
        ):

            st.session_state.kb_answer = None
            st.session_state.kb_sources = []

            st.rerun()


    # -----------------------------------------------------
    # Display knowledge base sources
    # -----------------------------------------------------

    if (
        st.session_state.kb_sources
        and st.session_state.kb_answer
        != "The provided documents do not contain enough information to answer this question."
    ):

        st.subheader("📚 Sources")

        seen_sources = set()

        source_number = 1


        for (
            document,
            distance,
        ) in st.session_state.kb_sources:

            source = document.metadata.get(
                "source",
                "Unknown source",
            )

            page = document.metadata.get(
                "page",
                "Unknown page",
            )


            page_number = (
                page + 1
                if isinstance(page, int)
                else page
            )


            source_key = (
                source,
                page,
            )


            if source_key in seen_sources:
                continue


            seen_sources.add(
                source_key
            )


            with st.expander(
                f"Source {source_number}: "
                f"{Path(source).name} "
                f"— Page {page_number}"
            ):

                st.caption(
                    f"Retrieval distance: "
                    f"{distance:.4f}"
                )

                st.write(
                    document.page_content
                )


            source_number += 1


# =========================================================
# UPLOAD PDF TAB
# =========================================================

with upload_tab:

    st.subheader(
        "📤 Upload Your PDF"
    )

    st.write(
        "Upload a technical PDF and ask questions "
        "specifically about that document."
    )


    # -----------------------------------------------------
    # PDF uploader
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="pdf_uploader",
    )


    current_upload_hash = None


    if uploaded_file:

        current_upload_hash = (
            hashlib.sha256(
                uploaded_file.getvalue()
            ).hexdigest()[:12]
        )


    # Clear previous upload answer
    # when a different file is selected.

    if (
        current_upload_hash
        != st.session_state.upload_hash
    ):

        st.session_state.upload_answer = None

        st.session_state.upload_sources = []

        st.session_state.upload_hash = (
            current_upload_hash
        )


    uploaded_vector_store = None

    uploaded_page_count = 0

    uploaded_chunk_count = 0


    # -----------------------------------------------------
    # Process uploaded PDF
    # -----------------------------------------------------

    if uploaded_file:

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )


        pdf_bytes = uploaded_file.getvalue()


        if not pdf_bytes:

            st.error(
                "The uploaded PDF is empty. "
                "Please upload a valid PDF."
            )

            st.stop()


        try:

            with st.spinner(
                "Processing PDF and creating embeddings..."
            ):

                (
                    uploaded_vector_store,
                    uploaded_page_count,
                    uploaded_chunk_count,
                ) = create_uploaded_vector_store(
                    pdf_bytes,
                    uploaded_file.name,
                    embeddings,
                )


            col1, col2 = st.columns(2)


            col1.metric(
                "📑 Pages",
                uploaded_page_count,
            )


            col2.metric(
                "🧩 Chunks",
                uploaded_chunk_count,
            )


            st.info(
                f"🔎 Questions will be answered "
                f"using **{uploaded_file.name}** only."
            )


        except Exception as e:

            st.error(
                "Failed to process uploaded PDF:\n\n"
                f"{e}"
            )

            st.stop()


    # -----------------------------------------------------
    # Upload PDF Q&A
    # -----------------------------------------------------

    st.subheader(
        "💬 Ask About Your PDF"
    )


    with st.form(
        "uploaded_question_form"
    ):

        uploaded_question = st.text_input(
            "Enter your question:",
            placeholder=(
                "e.g. What is the main concept "
                "explained in this document?"
            ),
        )


        uploaded_submitted = (
            st.form_submit_button(
                "🔍 Ask Copilot",
                type="primary",
            )
        )


    if uploaded_submitted:

        # Clear the previous answer.
        st.session_state.upload_answer = None
        st.session_state.upload_sources = []

        if not uploaded_file:

            st.warning(
                "Please upload a PDF first."
            )


        elif not uploaded_question.strip():

            st.warning(
                "Please enter a question."
            )


        elif uploaded_vector_store is None:

            st.error(
                "The uploaded PDF could not be processed."
            )


        else:

            with st.spinner(
                "Searching the uploaded PDF and generating answer..."
            ):

                # Retrieve chunks.
                try:

                    results = retrieve_documents(
                        query=uploaded_question,
                        vector_store=uploaded_vector_store,
                        top_k=3,
                    )

                except Exception as e:

                    st.error(
                        "⚠️ Unable to search the "
                        f"uploaded PDF: {e}"
                    )

                    st.stop()


                if not results:

                    st.warning(
                        "No relevant information was "
                        "found in the uploaded PDF."
                    )

                    st.stop()


                # Extract documents.
                documents = [
                    document
                    for document, _ in results
                ]


                # Build context.
                context = build_context(
                    documents
                )


                # Create grounded prompt.
                prompt = create_rag_prompt(
                    context=context,
                    question=uploaded_question,
                )


                # Generate answer.
                try:

                    answer = generate_answer(
                        client=gemini_client,
                        prompt=prompt,
                    )


                    st.session_state.upload_answer = (
                        answer
                    )

                    st.session_state.upload_sources = (
                        results
                    )


                except Exception as e:

                    error_message = str(e)


                    if (
                        "429" in error_message
                        or "quota"
                        in error_message.lower()
                        or "rate limit"
                        in error_message.lower()
                    ):

                        st.error(
                            "⚠️ Gemini API quota exceeded. "
                            "Your current Gemini Free Tier limit has been reached. "
                            "Please try again after the quota resets or check your Gemini API billing and limits."
                        )


                    elif (
                        "connection"
                        in error_message.lower()
                        or "timeout"
                        in error_message.lower()
                        or "timed out"
                        in error_message.lower()
                    ):

                        st.error(
                            "⚠️ Unable to connect to "
                            "the Gemini API. "
                            "Please check your internet "
                            "connection and try again."
                        )


                    else:

                        st.error(
                            "⚠️ Unable to generate an "
                            "answer right now. "
                            "Please try again later."
                        )

                    st.stop()


    # -----------------------------------------------------
    # Display uploaded PDF answer
    # -----------------------------------------------------

    if st.session_state.upload_answer:

        st.subheader("🤖 Answer")

        with st.container(border=True):

            st.write(
                st.session_state.upload_answer
            )

        if st.button(
            "🗑️ Clear Answer",
            key="clear_upload_answer",
        ):

            st.session_state.upload_answer = None
            st.session_state.upload_sources = []

            st.rerun()


    # -----------------------------------------------------
    # Display uploaded PDF sources
    # -----------------------------------------------------

    if (
        st.session_state.upload_sources
        and st.session_state.upload_answer
        != "The provided documents do not contain enough information to answer this question."
    ):

        st.subheader("📚 Sources")

        seen_sources = set()

        source_number = 1


        for (
            document,
            distance,
        ) in st.session_state.upload_sources:

            source = document.metadata.get(
                "source",
                "Unknown source",
            )

            page = document.metadata.get(
                "page",
                "Unknown page",
            )


            page_number = (
                page + 1
                if isinstance(page, int)
                else page
            )


            source_key = (
                source,
                page,
            )


            if source_key in seen_sources:
                continue


            seen_sources.add(
                source_key
            )


            with st.expander(
                f"Source {source_number}: "
                f"{Path(source).name} "
                f"— Page {page_number}"
            ):

                st.caption(
                    f"Retrieval distance: "
                    f"{distance:.4f}"
                )

                st.write(
                    document.page_content
                )


            source_number += 1