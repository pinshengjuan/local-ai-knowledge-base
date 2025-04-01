# src/ui.py
import streamlit as st

# --- UI Components ---
def display_mode(mode_text):
    """
    Display the current mode of the application.

    Args:
        mode_text (str): The text to display for the current mode.
    """
    st.write(f"**Mode:** {mode_text}")

def display_relevant(latest_result, container):
    """
    Display the relevant document sections in the sidebar.

    Args:
        latest_result (list): List of tuples containing query, answer, source_docs, and snapshots.
        container (st.container): Streamlit container to display the relevant sections.
    """
    with container:
        st.write("Relevant Document Sections")
        if not latest_result:
            st.write("No relevant sections yet.")
            return

    for item in latest_result:
        # Unpack the tuple: ignore prompt and answer, take source_docs and snapshots
        _, _, source_docs, snapshots = item  # Use _ for unused variables
        if snapshots:
            for doc, snap in zip(source_docs, snapshots or []):
                if snap and snap["snapshot"]:
                    source_info = f"{doc.metadata.get('source', 'Unknown source')} (Page {doc.metadata.get('page', 'N/A')+1})" #+1 because page number is 1-indexed
                    content_snippet = f"{doc.page_content[:300]}..."
                    content = f'{source_info}:\n{content_snippet}'
                    st.sidebar.write(content)
                    st.sidebar.image(
                        snap["snapshot"],
                        caption=f"Page {snap['page_number']}, {snap['file_path']} ",
                        use_column_width=True
                    )
                else:
                    st.warning(f"Snapshot unavailable for {snap['file_path']} (Page {snap['page_number']})")

def display_conversation(conversation):
    """
    Display the conversation history in a chat-like format.

    Args:
        conversation (list): List of tuples containing query, answer, source_docs, and snapshots.
    """
    # Custom CSS for alignment and styling
    st.markdown(
        """
        <style>
        .chat-container {
            overflow-y: auto;
            padding: 10px;
        }
        .question {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 10px;
        }
        .answer {
            display: flex;
            justify-content: flex-start;
            margin-bottom: 10px;
        }
        .message-box {
            background-color: #f0f0f0; /* Light gray for message boxes */
            color: black;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            max-width: 60%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display conversation in chat-like format
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for item in conversation:
            query, answer, _, _ = item  # Use _ for unused variables
            # Align question to the right
            query_html = f'<div class="question"><div class="message-box"> {query}'
            st.markdown(query_html, unsafe_allow_html=True)

            # Align answer to the left
            answer_html = f'<div class="answer"><div class="message-box"> {answer}'
            st.markdown(answer_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def toggle_knowledge_base(use_knowledge_base):
    """
    Toggle to use the knowledge base for answering questions.

    Args:
        use_knowledge_base (bool): Whether to use the knowledge base for answering.
    """
    st.session_state.use_knowledge_base = st.toggle(
        "Use Knowledge Base",
        value=use_knowledge_base,
        help="Toggle to use your documents or query the LLM directly."
    )

def render_input_form(input_key):
    """
    Render the input form for asking questions.

    Args:
        input_key (int): The key to identify the input form.
    
    Returns:
        tuple: A tuple containing the query and the submission status.
    """
    with st.form(key="query_form", clear_on_submit=True):
        query = st.text_input(
            "Ask a question:",
            key=f"query_{input_key}",
            placeholder="Type your question here...",
            label_visibility="visible"
        )
        submitted = st.form_submit_button("Ask")
    return query, submitted