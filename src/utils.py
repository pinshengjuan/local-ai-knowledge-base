# src/utils.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
def load_config():
    return {
        "OLLAMA_HOST": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "FILES_DIR": "./documents",
        "MODEL_NAME": os.getenv("MODEL_NAME"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL"),
        "CHUNK_SIZE": 500,
        "CHUNK_OVERLAP": 250,
        "RETRIEVER_K": 5
    }

# Custom prompt for penalty-related questions
def get_prompt():
    prompt_template = """
    You are an expert in interpreting legal documents. Use the following context from the documents to answer the question. If the context contains information about fines or penalties, extract the specific amount or details. If no specific amount is found, say so and provide any relevant information available.

    Context: {context}

    Question: {question}

    Answer:
    """
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# --- Document Processing ---
@st.cache_resource
def load_knowledge_base(files_dir, chunk_size, chunk_overlap, embedding_model):
    loader = PyPDFDirectoryLoader(files_dir)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# --- LLM and Chain Setup ---
@st.cache_resource
def init_llm(ollama_host, model_name):
    return ChatOllama(base_url=ollama_host, model=model_name)

def setup_qa_chain(llm, vector_store, prompt, retriever_k):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": retriever_k}),
        chain_type_kwargs={"prompt": prompt}
    )

# --- Query Handling ---
def process_query(query, llm, qa_chain, use_knowledge_base):
    if not query:
        st.warning("Please enter a question.")
        return None
    with st.spinner("Thinking..."):
        if use_knowledge_base:
            rewritten_query = llm.invoke("Rewrite this question clearly: " + query).content
            result_dict = qa_chain({"query": rewritten_query})
            return result_dict["result"]
        else:
            return llm.invoke(query).content