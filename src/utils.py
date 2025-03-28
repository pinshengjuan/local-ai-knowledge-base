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
    """
    Load configuration parameters from environment variables.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
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
    """
    Generate a custom prompt for penalty-related questions.

    Args:
        prompt_instruction (str): The instruction for the prompt.
    
    Returns:
        PromptTemplate: The prompt template for the QA chain.
    """
    prompt_template = f"""
{prompt_instruction}

Context: {{context}}

Question: {{question}}

Answer:
"""
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def update_knowledge_base(vector_store, new_documents, embedding_model):
    """
    Update the knowledge base with new documents.

    Args:
        vector_store (FAISS): The FAISS vector store instance.
        new_documents (list): List of new documents to add.
        embedding_model (str): The name of the embedding model.

    Returns:
        FAISS: The updated FAISS vector store instance.
    """
    embeddings = OllamaEmbeddings(model=embedding_model)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
    chunks = text_splitter.split_documents(new_documents)
    vector_store.add_documents(chunks)  # Incrementally add to existing FAISS index
    return vector_store

# --- Document Processing ---
@st.cache_resource
def load_knowledge_base(files_dir, chunk_size, chunk_overlap, embedding_model):
    """
    Load the knowledge base from a directory of PDF files.

    Args:
        files_dir (str): The directory containing the PDF files.
        chunk_size (int): The size of each text chunk.
        chunk_overlap (int): The overlap between text chunks.
        embedding_model (str): The name of the embedding model.
    
    Returns:
        FAISS: The FAISS vector store instance.
    """
    # Load documents from PDF files
    loader = PyPDFDirectoryLoader(files_dir)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    # Generate embeddings and create FAISS vector store
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# --- LLM and Chain Setup ---
@st.cache_resource
def init_llm(ollama_host, model_name):
    """
    Initialize the ChatOllama instance.

    Args:
        ollama_host (str): The host URL for the Ollama server.
        model_name (str): The name of the model to use.
    
    Returns:
        ChatOllama: The ChatOllama instance.
    """
    return ChatOllama(base_url=ollama_host, model=model_name)

def setup_qa_chain(llm, vector_store, prompt, retriever_k):
    """
    Setup the QA chain with the LLM, vector store, and prompt.
    
    Args:
        llm (ChatOllama): The LLM instance.
        vector_store (FAISS): The vector store instance.
        prompt (PromptTemplate): The prompt template for the QA chain.
        retriever_k (int): The number of retriever results to return.
    
    Returns:
        RetrievalQA: The QA chain instance
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": retriever_k}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

def generate_pdf_snapshot(file_path, page_number):
    """
    Generate a snapshot of a PDF page.

    Args:
        file_path (str): The path to the PDF file.
        page_number (int): The page number to snapshot
    
    Returns:
        bytes: The image bytes of the
    """
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
    """
    Process a query using the LLM or the knowledge base QA chain.

    Args:
        query (str): The question to ask.
        llm (ChatOllama): The LLM instance.
        qa_chain (RetrievalQA): The QA chain instance.
        use_knowledge_base (bool): Whether to use the knowledge base for answering.
    
    Returns:
        dict: A dictionary containing the answer, source documents, and snapshots.
    """
    if not query:
        st.warning("Please enter a question.")
        return None
    with st.spinner("Thinking..."):
        if use_knowledge_base:
            rewritten_query = llm.invoke("Rewrite this question clearly: " + query).content
            result_dict = qa_chain({"query": rewritten_query})
            # Generate snapshots for each source document
            snapshots = []
            for doc in result_dict["source_documents"]:
                file_path = doc.metadata.get("source")
                page_number = doc.metadata.get("page")
                if file_path and page_number:
                    snapshot = generate_pdf_snapshot(file_path, page_number+1) #+1 because page number is 1-indexed
                    if snapshot:
                        logger.info(f"Snapshot for {file_path}, page {page_number}: {snapshot is not None}")
                        snapshots.append({
                            "file_path": file_path,
                            "page_number": page_number+1, #+1 because page number is 1-indexed
                            "snapshot": snapshot
                        })
            return {
                "answer": result_dict["result"],
                "source_documents": result_dict["source_documents"],
                "snapshots": snapshots
            }
        else:
            return {"answer": llm.invoke(query).content, "source_documents": [], "snapshots": []}