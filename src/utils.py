# src/utils.py
import os
import io
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
from PIL import Image, ImageDraw
from pdf2image import convert_from_path

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
def load_config():
    return {
        "OLLAMA_HOST": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "FILES_DIR": os.getenv("FILES_DIR"),
        "MODEL_NAME": os.getenv("MODEL_NAME"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL"),
        "CHUNK_SIZE": 500,
        "CHUNK_OVERLAP": 250,
        "RETRIEVER_K": 5,
        "PROMPT_INSTRUCTION": os.getenv("PROMPT_INSTRUCTION")
    }

# Custom prompt for penalty-related questions
def get_prompt(prompt_instruction):
    prompt_template = f"""
{prompt_instruction}

Context: {{context}}

Question: {{question}}

Answer:
"""
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def update_knowledge_base(vector_store, new_documents, embedding_model):
    embeddings = OllamaEmbeddings(model=embedding_model)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
    chunks = text_splitter.split_documents(new_documents)
    vector_store.add_documents(chunks)  # Incrementally add to existing FAISS index
    return vector_store

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
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

def generate_pdf_snapshot(file_path, page_number):
    try:
        # Convert PDF page to image
        image = convert_from_path(pdf_path=file_path, first_page=page_number, last_page=page_number, fmt="png", dpi=600)
        img_byte_arr = io.BytesIO()
        image[0].save(img_byte_arr, format="PNG")
        return img_byte_arr.getvalue()
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
    except Exception as e:
        raise Exception(f"Error generating PDF snapshot: {str(e)}")

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