# src/app.py
import os
import streamlit as st
from utils import load_config, get_prompt, update_knowledge_base, load_knowledge_base, init_llm, setup_qa_chain, process_query
from ui import display_mode, display_relevant, display_conversation, toggle_knowledge_base, render_input_form

# --- Main Application ---
def main():
    """
    Main Streamlit application.

    This function initializes the application, loads the configuration,
    and sets up the LLM and knowledge base. It then displays the UI elements
    and handles the conversation flow.
    """
    st.title("Local AI Knowledge Base")
    
    # Load configuration
    config = load_config()

    # Initialize session state
    if "conversation" not in st.session_state: # To store conversation history
        # st.session_state.conversation is a list of tuples, where each tuple contains:
        # (
        #   'prompt': str,  # The user's query or input
        #   'answer': str,  # The AI's response
        #   [
        #       Document(metadata={'source': str, 'page': int}, page_content=str),  # List of source documents
        #       Document(metadata={'source': str, 'page': int}, page_content=str),  # List of source documents
        #       ...
        #   ],
        #   [
        #       {
        #           'file_path': str,  # Path to the document
        #           'page_number': int,  # Page number in the document
        #           'snapshot': str  # Snapshot identifier
        #       },
        #       {
        #           'file_path': str,  # Path to the document
        #           'page_number': int,  # Page number in the document
        #           'snapshot': str  # Snapshot identifier
        #       },
        #       ...
        #   ]
        # ),
        # ...
        st.session_state.conversation = []
    if "input_key" not in st.session_state: # To manage the input form key
        st.session_state.input_key = 0
    if "use_knowledge_base" not in st.session_state: # To toggle between LLM and knowledge base
        st.session_state.use_knowledge_base = True
    if "sidebar_latest_result" not in st.session_state:  # To store the latest result for sidebar display
        st.session_state.sidebar_latest_result = None

    # Load LLM and setup chain
    llm = init_llm(config["OLLAMA_HOST"], config["MODEL_NAME"])
    if st.session_state.use_knowledge_base:
        with st.spinner("Loading knowledge base..."):
            vector_store = load_knowledge_base(
                config["FILES_DIR"], 
                config["CHUNK_SIZE"], 
                config["CHUNK_OVERLAP"], 
                config["EMBEDDING_MODEL"]
            )
            prompt = get_prompt(config["PROMPT_INSTRUCTION"])
            qa_chain = setup_qa_chain(llm, vector_store, prompt, config["RETRIEVER_K"])
        mode_text = "Using Knowledge Base"
    else:
        qa_chain = llm
        mode_text = "Direct LLM Query"

    # Create a sidebar container to manage content
    sidebar_container = st.sidebar.empty()

    # # File upload section
    # uploaded_files = st.file_uploader("Upload new documents (PDFs)", type=["pdf"], accept_multiple_files=True)
    # if uploaded_files:
    #     with st.spinner("Updating knowledge base..."):
    #         new_docs = [UnstructuredFileLoader(file).load()[0] for file in uploaded_files]
    #         st.session_state.vector_store = update_knowledge_base(st.session_state.vector_store, new_docs, config["EMBEDDING_MODEL"])
    #         qa_chain = setup_qa_chain(llm, st.session_state.vector_store, prompt, config["RETRIEVER_K"])
    #     st.success("Knowledge base updated!")

    # Display UI elements
    display_mode(mode_text)
    display_conversation(st.session_state.conversation)
    toggle_knowledge_base(st.session_state.use_knowledge_base)

    # Display relevant document sections in the sidebar
    if st.session_state.sidebar_latest_result:
        display_relevant([st.session_state.sidebar_latest_result], sidebar_container)
    else:
        with sidebar_container:
            st.write("Relevant Document Sections")

    # Render input form in the main flow
    query, submitted = render_input_form(st.session_state.input_key)
    if submitted and query:
        # Clear the sidebar immediately when a new question is submitted
        with sidebar_container:
            st.write("Relevant Document Sections")
            st.write("Processing new query...")
        result = process_query(query, llm, qa_chain, st.session_state.use_knowledge_base)
        if result:
            current_conversation_entry = (query, result["answer"], result["source_documents"], result["snapshots"])
            st.session_state.conversation.append(current_conversation_entry)

            # Generate PDF snapshot if using knowledge base
            st.session_state.sidebar_latest_result = current_conversation_entry
            st.session_state.input_key += 1
            st.rerun()

if __name__ == "__main__":
    main()