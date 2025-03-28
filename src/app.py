# src/app.py
import os
import streamlit as st
from utils import load_config, get_prompt, load_knowledge_base, init_llm, setup_qa_chain, process_query
from ui import display_mode, display_conversation, toggle_knowledge_base, render_input_form

# --- Main Application ---
def main():
    st.title("Local AI Knowledge Base")
    
    # Load configuration
    config = load_config()

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0
    if "use_knowledge_base" not in st.session_state:
        st.session_state.use_knowledge_base = True

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

    # Display UI elements
    display_mode(mode_text)
    display_conversation(st.session_state.conversation)
    toggle_knowledge_base(st.session_state.use_knowledge_base)

    # Render input form in the main flow
    query, submitted = render_input_form(st.session_state.input_key, st.session_state.use_knowledge_base)
    if submitted and query:
        result = process_query(query, llm, qa_chain, st.session_state.use_knowledge_base)
        if result:
            st.session_state.conversation.append((query, result))
            st.session_state.input_key += 1
            st.rerun()

if __name__ == "__main__":
    main()