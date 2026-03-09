from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os


# -----------------------------
# Load PDF Documents
# -----------------------------
def load_documents(pdf_folder):

    documents = []

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())

    return documents


# -----------------------------
# Split Documents into Chunks
# -----------------------------
def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    return chunks


# -----------------------------
# Create / Load Vector Database
# -----------------------------
def create_vector_db(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load existing DB if available
    if os.path.exists("vectordb"):
        vectordb = Chroma(
            persist_directory="vectordb",
            embedding_function=embeddings,
            collection_name="pdf_collection"
        )
        return vectordb

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory="vectordb",
        collection_name="pdf_collection"
    )

    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectordb.add_documents(batch)

    vectordb.persist()

    return vectordb


# -----------------------------
# Load TinyLlama LLM
# -----------------------------
def load_llm():

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.3,
        repetition_penalty=1.2
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return llm


# -----------------------------
# Create RAG Chain
# -----------------------------
def create_chain(vectordb):

    llm = load_llm()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    prompt_template = """
You are an AI assistant that answers questions using the provided context.

Rules:
- Use ONLY the given context
- If the answer is not in the context, say "I don't know"
- Keep the answer short and clear.

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 2, "fetch_k": 5}
        ),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return chain


# -----------------------------
# Main Chat Loop
# -----------------------------
def main():

    pdf_folder = "pdfs"

    print("Loading documents...")
    documents = load_documents(pdf_folder)

    print("Splitting documents...")
    chunks = split_documents(documents)

    print("Creating / Loading Vector DB...")
    vectordb = create_vector_db(chunks)

    print("Creating RAG chain...")
    chain = create_chain(vectordb)

    print("\nRAG Chatbot Ready!")
    print("Type 'exit' to quit\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            break

        result = chain({"question": question})

        answer = result["answer"]

        print("\nBot:", answer)
        print()


if __name__ == "__main__":
    main()