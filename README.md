<div align="center">

# 📚 RAG Document Q&A Copilot

### Ask • Retrieve • Understand • Answer

**A technical document question-answering system powered by Retrieval-Augmented Generation (RAG).**

Built with **Python · LangChain · ChromaDB · Sentence Transformers · Google Gemini · Streamlit**

<br>

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Gemini](https://img.shields.io/badge/Google%20Gemini-LLM-orange)

</div>

---

## 🌟 Overview

**RAG Document Q&A Copilot** is a document-grounded AI application that allows users to ask questions about technical PDF documents.

Instead of relying only on the language model's internal knowledge, the system first **retrieves relevant information from the documents** and then provides that context to Gemini to generate a grounded answer.

It supports both a **pre-built technical knowledge base** and **user-uploaded PDFs**.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📄 **PDF Ingestion** | Load and process technical PDF documents |
| ✂️ **Smart Chunking** | Split documents using recursive character-based chunking |
| 🧠 **Embeddings** | Generate semantic embeddings with Sentence Transformers |
| 🗄️ **Vector Database** | Store and search embeddings using ChromaDB |
| 🔎 **Intelligent Retrieval** | Combine semantic, keyword, and phrase relevance |
| 🎯 **Top-K Retrieval** | Select the most relevant document chunks |
| 🤖 **Grounded Generation** | Generate answers using retrieved context |
| 🛡️ **Hallucination Control** | Restrict answers to available document information |
| 🚫 **Answer Abstention** | Avoid guessing when information is unavailable |
| 📤 **PDF Upload** | Ask questions about an uploaded document |
| 📌 **Source Tracking** | Display source PDF and page information |
| 🖥️ **Streamlit UI** | Simple interactive web interface |

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   📄 PDF Documents  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   PDF Text Loader   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   ✂️ Text Chunking  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 🧠 Sentence         │
                    │    Transformers     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   🗄️ ChromaDB       │
                    └──────────┬──────────┘
                               │
                               │
                         👤 User Query
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 🔎 Retrieval +      │
                    │    Ranking          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 🎯 Top-K Context    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 🤖 Google Gemini    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 💬 Grounded Answer  │
                    │    + Sources        │
                    └─────────────────────┘
```

---

## 🔄 RAG Pipeline

```text
LOAD → CHUNK → EMBED → STORE → RETRIEVE → RANK → GENERATE → ANSWER
```

### How it works

1. **Load** — PDFs are loaded using `PyPDFLoader`.
2. **Chunk** — Text is split using `RecursiveCharacterTextSplitter`.
3. **Embed** — Chunks are converted into embeddings using `all-MiniLM-L6-v2`.
4. **Store** — Embeddings are stored in ChromaDB.
5. **Retrieve** — Relevant chunks are retrieved for the user's question.
6. **Rank** — Semantic, keyword, and phrase relevance improve ranking.
7. **Generate** — Retrieved context is provided to Gemini.
8. **Answer** — Gemini generates a grounded response.
9. **Abstain** — The system avoids guessing when the required information is unavailable.

---

## 📚 Knowledge Base

The technical knowledge base covers topics including:

`RAG Fundamentals` · `RAG Architecture` · `Chunking` · `Embeddings` · `Vector Databases` · `ChromaDB` · `FAISS` · `Similarity Search` · `Top-K Retrieval` · `Hybrid Retrieval` · `Reranking` · `Prompt Engineering` · `Hallucination Control` · `RAG Evaluation` · `Query Rewriting` · `Metadata Filtering` · `Prompt Injection` · `PII Protection` · `Observability` · `Caching` · `Deployment`

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| 🐍 **Python** | Core development |
| 🔗 **LangChain** | RAG pipeline and document processing |
| 📑 **PyPDFLoader** | PDF text extraction |
| ✂️ **RecursiveCharacterTextSplitter** | Document chunking |
| 🧠 **Sentence Transformers** | Text embeddings |
| 🗄️ **ChromaDB** | Vector storage and similarity search |
| 🤖 **Google Gemini** | Grounded answer generation |
| 🎈 **Streamlit** | Web application interface |

---

## 📁 Project Structure

```text
RAG-Doc-Copilot/
│
├── 📂 data/
│   └── 📂 pdfs/
│       └── *.pdf
│
├── 📂 src/
│   ├── app.py
│   ├── ingestion.py
│   ├── embeddings.py
│   ├── retrieval.py
│   └── generation.py
│
├── 📂 tests/
│   └── test_retrieval.py
│
├── 📂 chroma_db/
│
├── .env
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone <your-repository-url>
cd RAG-Doc-Copilot
```

### 2️⃣ Create a virtual environment

```bash
python -m venv rag_env
```

### 3️⃣ Activate the environment — Windows

```powershell
.\rag_env\Scripts\Activate.ps1
```

### 4️⃣ Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 🔐 Gemini API Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

> ⚠️ **Important:** Never commit your `.env` file or API key to GitHub.

---

## 🗄️ Build the Vector Database

From the project root:

```powershell
cd src
python embeddings.py
```

This will:

- Load the PDF knowledge base
- Create document chunks
- Generate embeddings
- Store vectors in ChromaDB
- Run a semantic-search test

---

## 🔎 Run Retrieval Tests

From the project root:

```powershell
python -m pytest tests/test_retrieval.py
```

Or run the retrieval module directly:

```powershell
cd src
python retrieval.py
```

---

## ▶️ Run the Application

From the project root:

```powershell
python -m streamlit run src/app.py
```

The application provides:

**🧠 Knowledge Base Q&A** · **📤 PDF Upload** · **💬 Uploaded-Document Q&A** · **📚 Sources** · **📊 Statistics**

---

## 💡 Example Questions

```text
What is Retrieval-Augmented Generation?

What is the purpose of text chunking in a RAG pipeline?

Why is vector database persistence useful?

What does hallucination control combine?

What is the purpose of Top-K retrieval?
```

For questions outside the available document knowledge, the application returns:

```text
The provided documents do not contain enough information to answer this question.
```

---

## 📊 Retrieval Evaluation

The current evaluation uses **15 supported questions** and **5 unsupported questions**.

| Metric | Result |
|---|---:|
| 🎯 Top-1 Accuracy | **86.7%** |
| 🔎 Top-3 Accuracy | **100%** |
| 📈 Mean Reciprocal Rank | **0.92** |

> These metrics evaluate the **retrieval component**, not the quality of Gemini's generated answers.

---

## 🔮 Future Improvements

- 🔀 Hybrid BM25 + vector retrieval
- 🏆 Cross-encoder reranking
- 🏷️ Advanced metadata filtering
- 💬 Conversation-aware retrieval
- 📊 Expanded evaluation datasets
- ⏱️ Retrieval and generation latency monitoring

---

## 👤 Author

<div align="center">

### **Lakshya Garg**

💻 **RAG · Generative AI · Python**

*Building practical AI systems with retrieval, reasoning, and grounded generation.*

</div>

---

<div align="center">

⭐ **If you find this project useful, consider giving it a star!** ⭐

</div>
