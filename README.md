# RAG Multi-Document Chatbot

A **Retrieval-Augmented Generation (RAG) chatbot** that allows users to ask questions about multiple PDF documents.
The chatbot retrieves relevant information from documents using **ChromaDB vector search** and generates answers using **TinyLlama**.

---

## 🚀 Features

* 📄 Load and process **multiple PDF documents**
* ✂️ Automatic **text chunking**
* 🔎 **Semantic search** using vector embeddings
* 🧠 **Conversational memory** for follow-up questions
* 🤖 **TinyLlama LLM** for answer generation
* ⚡ **ChromaDB vector database**
* 💬 Interactive **command-line chatbot**

---

## 🏗 Architecture

User Question
↓
Retriever (ChromaDB)
↓
Relevant Document Chunks
↓
TinyLlama LLM
↓
Generated Answer

---

## 📂 Project Structure

```
rag-multi-document-chatbot/
│
├── app.py                 # Main chatbot application
├── pdfs/                  # Folder containing PDF documents
├── vectordb/              # ChromaDB vector database (auto-generated)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```
git clone https://github.com/punsarawickramarathna/rag-multi-document-chatbot.git
cd rag-multi-document-chatbot
```

### 2️⃣ Create virtual environment

```
python -m venv venv
```

Activate environment:

**Windows**

```
venv\Scripts\activate
```

**Mac/Linux**

```
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Chatbot

```
streamlit run app.py
```

Example:

```
You: What is Artificial Intelligence?

Bot: Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.
```

---

## 📚 Technologies Used

* **LangChain**
* **ChromaDB**
* **Sentence Transformers**
* **TinyLlama LLM**
* **HuggingFace Transformers**
* **Python**

---

## 🧠 How It Works

1. PDF documents are loaded using `PyPDFLoader`
2. Documents are split into smaller chunks
3. Chunks are converted into embeddings using **Sentence Transformers**
4. Embeddings are stored in **ChromaDB**
5. When a user asks a question:

   * Relevant chunks are retrieved
   * The LLM generates an answer using the retrieved context

---

## 🎯 Future Improvements

* Web UI using **Streamlit**
* Upload PDFs dynamically
* Deploy as an API using **FastAPI**
* Add support for **larger LLMs**

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

## 👨‍💻 Author

Developed by **Punsara Wickramarathna**

GitHub:
https://github.com/punsarawickramarathna
