# Local AI Knowledge Base Application

This is a Streamlit-based web application that allows users to interact with a local AI knowledge base powered by Ollama. The application supports querying a set of documents using a retrieval-augmented generation (RAG) approach or direct queries to a large language model (LLM). It includes features like conversation history, document snapshot generation, and a toggle to switch between knowledge base and direct LLM queries.

## Features
- **Document-Based QA**: Query a collection of PDF documents stored in a specified directory using a RAG pipeline.
- **Direct LLM Queries**: Ask questions directly to the LLM without using the knowledge base.
- **Conversation History**: View past queries and responses in a chat-like interface.
- **Document Snapshots**: Generate and display snapshots of relevant PDF pages in the sidebar.
- **Customizable Configuration**: Configure the application using environment variables (e.g., model name, embedding model, and prompt instructions).
- **Streamlit Interface**: User-friendly web interface for seamless interaction.

## Prerequisites
- **Docker**: Required to run the application using Docker Compose.
- **PDF Documents**: Place PDF files in the `./documents` directory for the knowledge base.
- **Hardware**: A machine with sufficient resources to run Ollama and the Streamlit app. For GPU acceleration, ensure compatibility with the `ollama/ollama:rocm` image.

## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up Environment Variables**
   Create a `.env` file in the `./src` directory based on the provided example:
   ```plaintext
   MODEL_NAME=llama3.2
   EMBEDDING_MODEL=nomic-embed-text
   PROMPT_INSTRUCTION="You are an expert in -. You are given a question and a set of documents. Your task is to provide a concise and accurate answer based on the information in the documents. If the answer is not present in the documents, say so. You should not make any assumptions or provide any information that is not in the documents. You should also not provide any information that is not relevant to the question. You should not provide any information that is not in the documents. Use the following context from the documents to answer the question. If you don't know, say so and provide any relevant information available."
   ```

3. **Prepare Documents**
   Place your PDF documents in the `./documents` directory. These will be indexed and used by the knowledge base.

4. **Build and Run with Docker Compose**
   ```bash
   docker-compose up --build
   ```
   This command starts two services:
   - **Ollama**: Runs the LLM server on port `11434`.
   - **Streamlit App**: Runs the web interface on port `8501`.

5. **Access the Application**
   Open your browser and navigate to `http://localhost:8501` to use the application.

## Usage
1. **Toggle Knowledge Base**: Use the toggle in the UI to switch between querying the knowledge base or directly querying the LLM.
2. **Ask a Question**: Enter your question in the text input field and click "Ask".
3. **View Conversation History**: The main interface displays the conversation history in a chat-like format.
4. **Inspect Relevant Documents**: When using the knowledge base, relevant document sections and snapshots of PDF pages are displayed in the sidebar.

## Project Structure
- **docker-compose.yml**: Defines the Docker services for Ollama and the Streamlit app.
- **src/Dockerfile**: Dockerfile for building the Streamlit application.
- **src/requirements.txt**: Python dependencies for the application.
- **src/app.py**: Main Streamlit application script.
- **src/ui.py**: Contains functions for rendering the UI components.
- **src/utils.py**: Utility functions for configuration, document processing, and query handling.
- **src/.env**: Environment variables for configuration.
- **documents/**: Directory for storing PDF documents to be indexed.

## Configuration
The application is configured via environment variables in the `.env` file:
- `MODEL_NAME`: The Ollama model to use (e.g., `llama3.2`).
- `EMBEDDING_MODEL`: The embedding model for document indexing (e.g., `nomic-embed-text`).
- `PROMPT_INSTRUCTION`: Instructions for the LLM to ensure accurate and relevant responses.
- `OLLAMA_HOST`: The URL of the Ollama server (default: `http://ollama:11434`).
- `FILES_DIR`: Directory containing the PDF documents (default: `./documents`).

Additional configurations in `utils.py`:
- `CHUNK_SIZE`: Size of text chunks for document splitting (default: 500).
- `CHUNK_OVERLAP`: Overlap between text chunks (default: 250).
- `RETRIEVER_K`: Number of documents to retrieve for each query (default: 5).

## Dependencies
- **Streamlit**: For the web interface.
- **LangChain**: For document processing, embeddings, and RAG pipeline.
- **Ollama**: For running the local LLM and embeddings.
- **FAISS**: For vector storage and retrieval.
- **PyPDF**: For loading PDF documents.
- **pdf2image**: For generating PDF page snapshots.
- **Pillow**: For image processing.

See `requirements.txt` for detailed dependency versions.

## Notes
- Ensure the `./documents` directory exists and contains valid PDF files before starting the application.
- The `ollama/ollama:rocm` image is used for GPU acceleration. If you don't have a compatible GPU, consider using the `ollama/ollama` image instead.
- Snapshots are generated for each relevant document page and displayed in the sidebar. Ensure `poppler-utils` is installed (included in the Dockerfile).

## Troubleshooting
- **Ollama Connection Issues**: Verify that the Ollama service is running and accessible at `http://localhost:11434` (or the configured `OLLAMA_HOST`).
- **PDF Snapshot Errors**: Ensure `poppler-utils` is installed and PDF files are valid. Check logs for specific errors.
- **Performance Issues**: Adjust `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `RETRIEVER_K` in `utils.py` to optimize performance.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any bugs or feature requests.

## License
This project is licensed under the GNU General Public License (GPL) Version 2. See the `LICENSE` file for details.