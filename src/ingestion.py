from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Dataset location
PDF_DIRECTORY = Path("data/pdfs")

# Chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def clean_document_text(text):
    """
    Keep the original document text unchanged.
    """

    return text


def load_pdfs(pdf_directory: Path):
    """
    Load all PDF files from the dataset directory.
    """

    documents = []

    pdf_files = sorted(
        pdf_directory.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: "
            f"{pdf_directory.resolve()}"
        )

    print(
        f"Found {len(pdf_files)} PDF files.\n"
    )

    for pdf_path in pdf_files:
        print(
            f"Loading: {pdf_path.name}"
        )

        loader = PyPDFLoader(
            str(pdf_path)
        )

        pdf_documents = loader.load()

        # Preserve original PDF content.
        for document in pdf_documents:
            document.page_content = clean_document_text(
                document.page_content
            )

        documents.extend(pdf_documents)

        print(
            f"  Pages loaded: "
            f"{len(pdf_documents)}"
        )

    return documents


def split_documents(documents):
    """
    Split documents into smaller chunks
    for embedding and retrieval.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_documents(
        documents
    )

    return chunks


def process_uploaded_pdf(pdf_path):
    """
    Load and split a user-uploaded PDF.
    """

    loader = PyPDFLoader(
        str(pdf_path)
    )

    documents = loader.load()

    # Preserve original PDF content.
    for document in documents:
        document.page_content = clean_document_text(
            document.page_content
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_documents(
        documents
    )

    return documents, chunks