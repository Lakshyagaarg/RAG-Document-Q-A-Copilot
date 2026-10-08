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

Instead of relying only on the language model's internal knowledge, the system first **retrieves relevant information from the documents** and then provides that context to Gemini to generate a grounded answer. When the documents don't contain enough evidence, it **abstains instead of guessing**.

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
| 🎯 **Two-Stage Retrieval** | Retrieve a broad candidate set, rerank, then select the top 3 chunks |
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
                    │ 🎯 Top-3 Context    │
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
5. **Retrieve** — Up to 15 semantically similar chunks are retrieved for the user's question.
6. **Rank** — Keyword and phrase-based scoring rerank the candidates, exact duplicates are removed, and repeated chunks from the same source are limited. The top 3 chunks are kept.
7. **Generate** — The retrieved context is provided to Gemini.
8. **Answer** — Gemini generates a response grounded in that context.
9. **Abstain** — If the available evidence is insufficient, the system returns a predefined response instead of guessing.

---

## 🧩 Design Decisions

**Why combine semantic and keyword-based retrieval?**
Semantic search captures meaning, even when wording differs. Technical queries, however, often depend on exact terms and phrases. A custom ranking step blends semantic relevance with keyword and phrase scoring to balance conceptual relevance with exact-term matching.

**Why retrieve 15 chunks but use only 3?**
Retrieving too few chunks risks missing relevant information, while passing too many adds unnecessary context. A two-stage process, a broad candidate set followed by reranking, keeps the context focused before it reaches the LLM.

**How are unsupported questions handled?**
An LLM may answer from its general training even when the documents don't support it. The generation prompt instructs the model to rely on the retrieved context and return a fixed response when the evidence is insufficient. This is an explicit abstention mechanism, but it does not eliminate hallucinations entirely.

**Why evaluate retrieval separately?**
Measuring retrieval on its own, apart from generation, helps isolate where failures actually occur in the pipeline.

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
RAG-Document-Q-A-Copilot/
│
├── 📂 data/
│   └── 📂 pdfs/
│       └── *.pdf
│
├── app.py
├── src/
│   ├── ingestion.py
│   ├── embeddings.py
│   ├── retrieval.py
│   └── generation.py
|
├── 📂 tests/
│   └── test_retrieval.py
│
├── 📂 chroma_db/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Lakshyagaarg/RAG-Document-Q-A-Copilot.git
cd RAG-Document-Q-A-Copilot
```

### 2️⃣ Create a virtual environment

```bash
python -m venv rag_env
```

### 3️⃣ Activate the environment

**Windows (PowerShell)**

```powershell
.\rag_env\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source rag_env/bin/activate
```

### 4️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Gemini API Configuration

Copy the example environment file and add your own key:

```bash
cp .env.example .env
```

On Windows PowerShell, use `copy .env.example .env`.

Then edit `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

> ⚠️ **Important:** Never commit your `.env` file or API key to GitHub. Make sure `.env` and `chroma_db/` are listed in `.gitignore`.

---

## 🗄️ Build the Vector Database

From the project root:

```bash
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

```bash
python -m pytest tests/test_retrieval.py
```

Or run the retrieval module directly:

```bash
cd src
python retrieval.py
```

---

## ▶️ Run the Application

From the project root:

```bash
python -m streamlit run app.py
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

### ⚠️ Limitations

- The evaluation set is small (20 questions), so results do not guarantee performance on unseen queries or documents.
- The custom reranking has not yet been compared against a semantic-only baseline, so its exact impact is not measured.
- Abstention reduces unsupported answers but does not eliminate hallucinations entirely.

---

## 🔮 Future Improvements

- 🔀 Replace custom keyword scoring with BM25
- 🏆 Add cross-encoder reranking
- 🆚 Compare against a semantic-only retrieval baseline
- 📊 Expand the evaluation dataset
- 🏷️ Advanced metadata filtering
- 💬 Conversation-aware retrieval
- ⏱️ Retrieval and generation latency monitoring

---

## 👤 Author

### **Lakshya Garg**

Final-year B.Tech (AI & ML) student building practical AI systems with retrieval, reasoning, and grounded generation.

- 🐙 GitHub: https://github.com/Lakshyagaarg
- 💼 LinkedIn: https://www.linkedin.com/in/lakshya-garg-a43b672b3/
- 📧 Email: lakshya.garg5785@gmail.com

📬 Currently open to AI/ML internship and entry-level opportunities.
