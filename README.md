# RAG Pipeline with LLM for Unstructured Data

This project implements a Retrieval Augmented Generation (RAG) pipeline with a Large Language Model (LLM) in a Streamlit application. It allows users to upload various unstructured documents (PDF, CSV, Excel), ask questions based on their content, and leverage web search for answers not found in the documents. Additionally, it offers features for generating Multiple Choice Questions (MCQs) from the document content.

## Features

- **Multi-Document Upload**: Upload PDF, CSV, and Excel files.
- **Intelligent Q&A (RAG)**: Ask questions about the content of uploaded documents. The system uses a hybrid retrieval approach combining dense (Qdrant) and sparse (BM25) methods, followed by re-ranking to fetch the most relevant information.
- **Web Search Fallback**: Automatically performs a web search using Tavily if the answer is not found or insufficiently covered in the uploaded documents.
- **MCQ Generation**: Generate multiple-choice questions based on the content of the uploaded documents.
- **Gemini LLM**: Utilizes Google's `gemini-1.5-flash` model for query expansion, answer generation, and MCQ creation.
- **Qdrant Vector Database**: Stores document embeddings for efficient semantic search.
- **BM25 Keyword Search**: Augments semantic search with keyword-based retrieval for improved relevance.

## Prerequisites

Before running the application, ensure you have the following installed and configured:

1. **Python 3.8+**
2. **Ollama**: For running local LLMs (e.g., Llama 2, Mistral). Download from [ollama.com](https://ollama.com/).
3. **API Keys**:
   * **Google Gemini API Key**: Obtain from [ai.google.dev](https://ai.google.dev/).
   * **Tavily API Key**: Obtain from [tavily.com](https://tavily.com/).

## Installation

1. **Clone the repository** (if you haven't already):

   ```bash
   git clone <your-repository-link>
   cd Rag_structured_unstructured
   ```
2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
4. **Set Environment Variables**:
   Create a `.env` file in the `Rag_structured_unstructured` directory with your API keys and Qdrant URL (if using a remote Qdrant instance; otherwise, default is fine):

   ```
   QDRANT_URL="<your_qdrant_url>" # e.g., "http://localhost:6333" if running locally
   QDRANT_API_KEY="<your_qdrant_api_key>"
   GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
   TAVILY_API_KEY="YOUR_TAVILY_API_KEY"
   ```

   *Replace `<your_qdrant_url>`, `<your_qdrant_api_key>`, `YOUR_GEMINI_API_KEY`, and `YOUR_TAVILY_API_KEY` with your actual values.*

   *If you are running Qdrant locally without an API key, you can omit `QDRANT_URL` and `QDRANT_API_KEY` or set them appropriately. The application is configured to use an in-memory Qdrant client if no URL/key is provided.*
5. **Download an Ollama Model**:
   Run the following command in your terminal to download a model (e.g., Llama 2):

   ```bash
   ollama run llama2
   ```

   You can replace `llama2` with other models like `mistral` if you prefer.

## Usage

1. **Start the Streamlit application**:
   Navigate to the `Rag_structured_unstructured` directory in your terminal and run:

   ```bash
   streamlit run app.py
   ```
2. **Access the Application**: The application will open in your web browser, typically at `http://localhost:8501`.
3. **Upload Documents**: Use the file uploader in the sidebar to upload PDF, CSV, or Excel files. The application will process these documents and create embeddings.
4. **Ask Questions**: Go to the "Document Q&A" tab and enter your questions related to the uploaded documents. The RAG pipeline will provide answers, leveraging web search if needed.
5. **Generate MCQs**: Switch to the "MCQ Generation" tab to generate multiple-choice questions based on the combined content of your uploaded documents.

## Project Structure

```
Rag_structured_unstructured/
├── .env                 # Environment variables (API keys, etc.)
├── app.py               # Main Streamlit application code
├── requirements.txt     # Python dependencies
└── README.md            # Project README file
```

## Troubleshooting

- **API Key Errors**: Ensure your `.env` file is correctly configured with valid API keys for Gemini and Tavily.
- **Ollama Not Running**: Make sure Ollama is running and you have downloaded a model (e.g., `llama2`). Check the Ollama service status.
- **Qdrant Connection**: Verify that your `QDRANT_URL` and `QDRANT_API_KEY` in the `.env` file are correct if you are using a remote Qdrant instance. If running locally, ensure Qdrant is accessible.
- **File Processing Errors**: Check the Streamlit logs for specific errors during document upload and processing.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
