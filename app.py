import os
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, PayloadSchemaType, CreateFieldIndex
from qdrant_client.models import PointStruct 
from sentence_transformers import SentenceTransformer 
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile
import shutil
import numpy as np
import re
import hashlib
import uuid
from rank_bm25 import BM25Okapi
from unstructured.partition.auto import partition
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader
import pandas as pd # Needed for CSV/Excel data handling potentially
from tavily import TavilyClient # For web search

load_dotenv()


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize Qdrant client
try:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
except Exception as e:
    st.error(f"Failed to initialize Qdrant client: {str(e)}")
    st.stop()

# Initialize sentence transformer model for embeddings
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Failed to initialize SentenceTransformer: {str(e)}")
    st.stop()

# Configure Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Failed to initialize Gemini API: {str(e)}")
    st.stop()

# Initialize Tavily client for web search
try:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Tavily client: {str(e)}")
    st.stop()

# Constants
COLLECTION_NAME = "rag_collection_unstructured"
VECTOR_SIZE = 384
CHUNK_SIZE = 1000
MIN_SIMILARITY_SCORE = 0.62  # Only accept chunks at/above this score

# ===== Core Functions =====
def generate_file_hash(file_path):
    """Generate a unique hash for the file based on its content"""
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
        return hashlib.md5(file_content).hexdigest()
    except Exception as e:
        st.error(f"Error generating file hash: {str(e)}")
        return None

def extract_text_from_file(uploaded_file):
    """Extract text from various file types"""
    try:
        # Create a temporary directory for uploaded files
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        text_content = ""

        if file_extension == ".pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                text_content += page.extract_text() or ""
        elif file_extension == ".csv":
            df = pd.read_csv(file_path)
            text_content = df.to_string()
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
            text_content = df.to_string()
        else:
            # Use unstructured for other types or as a fallback
            elements = partition(filename=file_path)
            text_content = "\n".join([str(el) for el in elements])
        
        shutil.rmtree(temp_dir) # Clean up temp directory
        return text_content, os.path.basename(file_path) # Return filename for context
    except Exception as e:
        st.error(f"Error extracting text from {uploaded_file.name}: {str(e)}")
        return None, None

def chunk_text(text):
    """Split text into manageable chunks"""
    if not text:
        return []
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE) if text[i:i + CHUNK_SIZE].strip()]

def tokenize_text(text):
    """Tokenize text for BM25 processing."""
    return text.lower().split()

def get_embeddings(chunks):
    """Convert text chunks into embeddings"""
    try:
        return model.encode(chunks, show_progress_bar=False)
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None

def is_valid_embeddings(embeddings):
    """Check if embeddings are valid and not empty"""
    if embeddings is None:
        return False
    if isinstance(embeddings, np.ndarray):
        return embeddings.size > 0
    if hasattr(embeddings, '__len__'):
        return len(embeddings) > 0
    return False

def setup_collection():
    """Create Qdrant collection if it doesn't exist"""
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_exists = any(col.name == COLLECTION_NAME for col in collections.collections)
        
        if not collection_exists:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            st.info(f"Created new collection: {COLLECTION_NAME}")
        else:
            st.info(f"Using existing collection: {COLLECTION_NAME}")
        
        # Create index for file_hash field if it doesn't exist
        try:
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="file_hash",
                field_schema=PayloadSchemaType.KEYWORD
            )
        except Exception:
            # Index might already exist, which is fine
            pass
        
        return True
    except Exception as e:
        st.error(f"Error setting up collection: {str(e)}")
        return False

def check_file_embeddings_exist(file_hash):
    """Check if embeddings for a file already exist in Qdrant"""
    try:
        result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="file_hash",
                        match=MatchValue(value=file_hash)
                    )
                ]
            ),
            limit=1
        )
        return len(result[0]) > 0
    except Exception as e:
        st.warning(f"Error checking existing embeddings: {str(e)}")
        return False

def get_existing_embeddings(file_hash):
    """Retrieve existing chunks for a file from Qdrant (no vectors needed)."""
    try:
        result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="file_hash",
                        match=MatchValue(value=file_hash)
                    )
                ]
            ),
            with_vectors=False,
            limit=10000  # Large limit to get all chunks
        )
        
        if result[0]:
            # Sort by id to maintain chunk order
            sorted_points = sorted(result[0], key=lambda x: x.id)
            chunks = [point.payload.get("text", "") for point in sorted_points]
            return chunks, None
        return [], None
    except Exception as e:
        st.error(f"Error retrieving existing embeddings: {str(e)}")
        return [], None

def clear_all_embeddings():
    try:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[])  # empty filter ⇒ match all points
            )
        )
        print("🗑️ Cleared all embeddings and payloads from collection")
    except Exception as e:
        print(f"Error clearing embeddings: {e}")

def load_all_chunks_for_bm25():
    """Loads all chunks from Qdrant to build a global BM25 model."""
    try:
        scroll_result, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100000,  # Adjust limit as needed for total number of chunks
            with_payload=True,
            with_vectors=False
        )
        all_chunks = [point.payload.get("text", "") for point in scroll_result if point.payload.get("text")]
        if all_chunks:
            tokenized_all_chunks = [tokenize_text(chunk) for chunk in all_chunks]
            st.session_state.bm25_model = BM25Okapi(tokenized_all_chunks)
            st.session_state.all_chunks = all_chunks # Store original chunks for retrieval
            st.info("BM25 model initialized with all existing chunks.")
        else:
            st.info("No chunks found in Qdrant for BM25 initialization.")
    except Exception as e:
        st.error(f"Error loading all chunks for BM25: {str(e)}")

def recreate_collection_with_schema():
    """Recreate collection with proper payload schema"""
    try:
        # Delete existing collection
        try:
            qdrant_client.delete_collection(COLLECTION_NAME)
        except:
            pass  # Collection might not exist
        
        # Create new collection
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        
        # Create index for file_hash field
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="file_hash",
            field_schema=PayloadSchemaType.KEYWORD
        )

        # Validate index works by performing a filtered scroll (limit 1)
        try:
            qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_hash",
                            match=MatchValue(value="_validation_check_")
                        )
                    ]
                ),
                with_vectors=False,
                limit=1
            )
        except Exception as validation_error:
            st.error(f"Validation after recreation failed. The index may not be active yet: {validation_error}")
            return False

        st.success("Collection recreated and file_hash index validated.")
        load_all_chunks_for_bm25()
        return True
    except Exception as e:
        st.error(f"Error recreating collection: {str(e)}")
        return False

def upload_to_qdrant(chunks, embeddings, file_hash):
    """Upload data to Qdrant with file hash for identification"""
    if not chunks or not is_valid_embeddings(embeddings):
        return False
    
    try:
        # Ensure vectors are plain Python lists of floats and provide stable string ids
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_list = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
            point_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_hash}-{i}")
            points.append(
                PointStruct(
                    id=str(point_id),
                    vector=vector_list,
                    payload={"text": chunk, "file_hash": file_hash}
                )
            )
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        return True
    except Exception as e:
        st.error(f"Error uploading to Qdrant: {str(e)}")
        return False

def search_chunks(query_text):
    """Retrieve most relevant chunks from all available PDFs using hybrid search."""
    try:
        # --- Dense Retrieval (Qdrant) ---
        query_vector = model.encode([query_text])[0]
        qdrant_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=20, # Get more candidates for re-ranking
        )
        dense_hits = {hit.payload.get("text", ""): hit.score for hit in qdrant_results}

        # --- Sparse Retrieval (BM25) ---
        if 'bm25_model' in st.session_state and 'all_chunks' in st.session_state:
            bm25_model = st.session_state.bm25_model
            all_chunks = st.session_state.all_chunks
            tokenized_query = tokenize_text(query_text)
            bm25_scores = bm25_model.get_scores(tokenized_query)
            
            # Pair scores with original chunks
            sparse_hits = {}
            for i, score in enumerate(bm25_scores):
                original_chunk = all_chunks[i]
                if score > 0: # Only consider chunks with a positive BM25 score
                    sparse_hits[original_chunk] = score
        else:
            sparse_hits = {}

        # --- Hybrid Scoring ---
        combined_scores = {}
        alpha = 0.5 # Weight for dense vs sparse

        all_chunks = set(dense_hits.keys()).union(set(sparse_hits.keys()))

        if not all_chunks: # If no chunks from either, return empty
            return []
        
        # Normalize scores (simple min-max for now, can be improved)
        max_dense_score = max(dense_hits.values()) if dense_hits else 1.0
        max_sparse_score = max(sparse_hits.values()) if sparse_hits else 1.0

        for chunk in all_chunks:
            dense_score = dense_hits.get(chunk, 0.0) / max_dense_score
            sparse_score = sparse_hits.get(chunk, 0.0) / max_sparse_score
            combined_scores[chunk] = (alpha * dense_score) + ((1 - alpha) * sparse_score)
        
        # Sort and return top chunks
        sorted_chunks = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        return [chunk for chunk, score in sorted_chunks[:20]]

    except Exception as e:
        st.error(f"Error performing hybrid search: {str(e)}")
        return []

def re_rank_chunks(query_text, candidate_chunks, top_k=5):
    """Re-ranks candidate chunks based on semantic similarity to the query."""
    if not candidate_chunks:
        return []

    # Encode query and candidate chunks
    query_embedding = model.encode([query_text])[0]
    chunk_embeddings = model.encode(candidate_chunks)

    # Calculate cosine similarity between query and each chunk
    query_embedding_norm = np.linalg.norm(query_embedding)
    chunk_embeddings_norm = np.linalg.norm(chunk_embeddings, axis=1)

    # Avoid division by zero for zero vectors
    if query_embedding_norm == 0:
        return []
    chunk_embeddings_norm[chunk_embeddings_norm == 0] = 1e-12 # Small epsilon to avoid div by zero

    similarities = np.dot(chunk_embeddings, query_embedding) / (chunk_embeddings_norm * query_embedding_norm)

    # Pair chunks with their similarity scores
    scored_chunks = list(zip(candidate_chunks, similarities))

    # Sort by similarity score in descending order
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Return top_k re-ranked chunks
    return [chunk for chunk, score in scored_chunks[:top_k]]

def expand_query(query, num_terms=3):
    """Expands the user query with additional terms using Gemini."""
    prompt = (
        f"Generate {num_terms} related terms or short phrases for the following query. "
        f"Output only the terms, separated by commas, no other text.\n"
        f"Query: {query}\n"
        f"Related terms:"
    )
    try:
        response = gemini_model.generate_content(prompt)
        expanded_terms = [term.strip() for term in response.text.strip().split(',') if term.strip()]
        return " ".join(expanded_terms) # Return as a space-separated string
    except Exception as e:
        st.warning(f"Error expanding query: {e}")
        return ""

def generate_answer_from_gemini(query, context, web_search_results=None):
    """Use Gemini to answer the question, optionally incorporating web search results."""
    if not query:
        return "Please provide a question."
    
    full_context = ""
    if context:
        full_context += f"Document Context:\n{context}\n\n"
    if web_search_results:
        full_context += f"Web Search Results:\n{web_search_results}\n\n"

    if not full_context:
        return "No relevant information found in documents or web search to answer the question."
    
    marks_match = re.search(r'(\d+)\s*marks', query, re.IGNORECASE)
    extr_instruction = ""
    if marks_match:
        marks = int(marks_match.group(1))
        if marks >= 8:
            extr_instruction = (
                "This is a high-mark question. Write a long, detailed, and well-structured answer. "
                "Include introduction, step-by-step explanation, algorithm, math, advantages, disadvantages, applications, and examples."
            )
        elif marks >= 5:
            extr_instruction = (
                "This is a medium-mark question. Write a moderately detailed answer with explanation, steps, and at least one example."
            )
        else:
            extr_instruction = (
                "This is a short-mark question. Write a concise answer."
            )

    prompt = (
        f"Context:\n{full_context}\n\n"
        f"Question: {query}\n\n"
        f"Instructions: {extr_instruction}\n\n"
        f"Answer:"
    )
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error from Gemini: {str(e)}"

def generate_mcqs_from_context(context, num_questions=5):
    """
    Use the LLM to generate MCQs from the context.
    """
    prompt = (
        f"Generate {num_questions} multiple choice questions (MCQs) from the following context. "
        f"For each question, use this format:\n"
        f"Q: <question text>\n"
        f"A) <option 1>\n"
        f"B) <option 2>\n"
        f"C) <option 3>\n"
        f"D) <option 4>\n"
        f"Answer: <A/B/C/D>\n\n"
        f"Context:\n{context}\n"
    )
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating MCQs: {e}")
        return None

def parse_mcqs(mcq_text):
    """
    Parse the MCQ text into a list of dicts.
    """
    questions = []
    blocks = re.split(r'\n(?=Q: )', mcq_text)
    for block in blocks:
        q = {}
        lines = block.strip().split('\n')
        if len(lines) < 6:
            continue
        q['question'] = lines[0][3:].strip()
        q['options'] = [line[3:].strip() for line in lines[1:5]]
        answer_line = lines[5]
        q['answer'] = answer_line.split(':')[-1].strip()
        questions.append(q)
    return questions

def generate_flowchart_from_context(context, topic):
    """
    Use the LLM to generate a flowchart description in DOT language.
    """
    prompt = (
        f"Given the following context from a textbook or manual, generate a flowchart in Graphviz DOT format "
        f"for the topic: '{topic}'. Only output the DOT code, nothing else.\n\n"
        f"Context:\n{context}\n"
    )
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating flowchart: {e}")
        return None

def main():
    st.set_page_config(layout="wide")
    st.title("📄 Document Question Answering System with RAG")

    # Sidebar for additional options
    with st.sidebar:
        st.header("🔧 Options")
        if st.button("🗑 Clear All Embeddings", help="This will delete all stored embeddings from Qdrant"):
            if clear_all_embeddings():
                st.session_state.processed_files = {}  # Stores hashes of processed files
                st.session_state.current_file_name = None
                st.session_state.context_chunks = []
                st.session_state.texts = {}  # stores text content by file hash
                st.session_state.chunks_by_file = {}  # stores chunks by file hash
                st.session_state.bm25_model = None
                st.session_state.all_chunks = []
                st.session_state.mcq_questions = []
                st.session_state.user_mcq_answers = []
                st.session_state.mcq_submitted = False
                st.session_state.last_uploaded_filenames = []
                st.rerun()

        if st.button("🔄 Recreate Collection", help="Recreate collection with proper schema (fixes index errors)"):
            if recreate_collection_with_schema():
                st.session_state.processed_files = {}  # Stores hashes of processed files
                st.session_state.current_file_name = None
                st.session_state.context_chunks = []
                st.session_state.texts = {}  # stores text content by file hash
                st.session_state.chunks_by_file = {}  # stores chunks by file hash
                st.session_state.bm25_model = None
                st.session_state.all_chunks = []
                st.session_state.mcq_questions = []
                st.session_state.user_mcq_answers = []
                st.session_state.mcq_submitted = False
                st.session_state.last_uploaded_filenames = []
                st.rerun()

        st.header("ℹ Information")
        st.info("""
        *How it works:*
        - Each uploaded document (PDF, CSV, Excel) gets a unique hash based on its content
        - Embeddings are stored with this hash as a key
        - If you upload the same document again, existing embeddings are reused
        - This saves time and computational resources
        """)

    # Session state initialization
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {} # Stores hashes of processed files
    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = None
    if 'context_chunks' not in st.session_state:
        st.session_state.context_chunks = []
    if 'texts' not in st.session_state:
        st.session_state.texts = {} # stores text content by file hash
    if 'chunks_by_file' not in st.session_state:
        st.session_state.chunks_by_file = {} # stores chunks by file hash
    if 'bm25_model' not in st.session_state:
        st.session_state.bm25_model = None
    if 'all_chunks' not in st.session_state:
        st.session_state.all_chunks = []
    if 'mcq_questions' not in st.session_state:
        st.session_state.mcq_questions = []
    if 'user_mcq_answers' not in st.session_state:
        st.session_state.user_mcq_answers = []
    if 'mcq_submitted' not in st.session_state:
        st.session_state.mcq_submitted = False
    if 'last_uploaded_filenames' not in st.session_state:
        st.session_state.last_uploaded_filenames = [] # To track all uploaded files

    # Initialize Qdrant and BM25 on startup
    if setup_collection():
        load_all_chunks_for_bm25()

    # Main file uploader
    uploaded_files = st.file_uploader("Upload PDF, CSV, or Excel files", type=["pdf", "csv", "xlsx"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files.values(): # Check by name for display
                # Save to a temporary location for hashing and processing
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.read())

                    file_hash = generate_file_hash(temp_file_path)

                    if file_hash not in st.session_state.processed_files:
                        st.info(f"⏳ Processing {uploaded_file.name}...")

                        if check_file_embeddings_exist(file_hash):
                            st.info(f"📚 Found existing embeddings for {uploaded_file.name}. Loading from Qdrant...")
                            chunks, _ = get_existing_embeddings(file_hash)
                            if chunks:
                                st.session_state.chunks_by_file[file_hash] = chunks
                                st.session_state.processed_files[file_hash] = uploaded_file.name
                                st.session_state.last_uploaded_filenames.append(uploaded_file.name)
                                st.success(f"✅ Embeddings for {uploaded_file.name} loaded from existing data.")
                            else:
                                st.warning(f"Existing embeddings found for {uploaded_file.name} but couldn't retrieve. Reprocessing...")
                                # Fall through to normal processing
                                pass

                        if file_hash not in st.session_state.processed_files: # If not loaded from existing, process
                            text_content, _ = extract_text_from_file(uploaded_file) # extract_text_from_file handles temp file now

                            if text_content:
                                st.session_state.texts[file_hash] = text_content
                                chunks = chunk_text(text_content)
                                st.session_state.chunks_by_file[file_hash] = chunks
                                embeddings = get_embeddings(chunks)

                                if chunks and is_valid_embeddings(embeddings) and setup_collection():
                                    if upload_to_qdrant(chunks, embeddings, file_hash):
                                        st.session_state.processed_files[file_hash] = uploaded_file.name
                                        st.session_state.last_uploaded_filenames.append(uploaded_file.name)
                                        st.success(f"✅ {uploaded_file.name} processed and indexed.")
                                    else:
                                        st.error(f"Failed to upload embeddings for {uploaded_file.name} to Qdrant.")
                                elif not (chunks and is_valid_embeddings(embeddings)):
                                    st.error(f"Failed to process {uploaded_file.name}.")
                            else:
                                st.error(f"No text extracted from {uploaded_file.name}.")

        # After processing all uploaded files, reload BM25 with all chunks
        if st.session_state.last_uploaded_filenames:
            load_all_chunks_for_bm25()

    # Check if any files are processed
    if not st.session_state.processed_files:
        st.warning("Please upload documents to get started.")
    else:
        st.info(f"Currently processed documents: {', '.join(st.session_state.processed_files.values())}")

    # Rest of the UI remains largely the same, but now operates on a collection of documents
    qna_tab, mcq_tab = st.tabs(["Document Q&A", "MCQ Generation"]) 

    with qna_tab:
        st.header("Ask a Question from the Documents")
        question = st.text_input("Ask a question based on the uploaded document content:")
        if question and st.session_state.processed_files:
            st.info("✨ Expanding query...")
            expanded_terms = expand_query(question)
            expanded_query_text = f"{question} {expanded_terms}".strip()

            st.info("🔍 Searching for relevant information...")
            candidate_chunks = search_chunks(expanded_query_text)

            re_ranked_chunks = re_rank_chunks(expanded_query_text, candidate_chunks, top_k=5)

            st.session_state.context_chunks = re_ranked_chunks
            context = " ".join(re_ranked_chunks)
            web_search_results = None

            # Implement logic to trigger web search if document context is insufficient
            if not context or len(context.strip()) < 100: # Simple heuristic: if context is empty or very short
                st.info("🌐 Document context insufficient. Performing web search...")
                try:
                    search_response = tavily_client.search(query=expanded_query_text)
                    if search_response and search_response['results']:
                        web_search_results = "\n".join([f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}\n" for r in search_response['results']])
                        st.success("✅ Web search completed and results will be used.")
                    else:
                        st.warning("⚠️ Web search performed but no relevant results found.")
                        web_search_results = None
                except Exception as e:
                    st.error(f"❌ Error during web search: {e}")
                    web_search_results = None
            else:
                st.info("✅ Sufficient document context found. Skipping web search.")
                web_search_results = None # Ensure it's None if not explicitly performed

            st.info("🤖 Generating answer from Gemini...")
            answer = generate_answer_from_gemini(question, context, web_search_results)
            st.subheader("🧠 Answer:")
            st.write(answer)

    with mcq_tab:
        st.header("Generate MCQs from Document Concepts")
        num_questions = st.slider("Number of MCQs", 1, 10, 5)

        if st.button("Generate MCQs") and st.session_state.processed_files:
            # Combine all processed text for MCQ generation
            combined_text_for_mcq = " ".join(st.session_state.texts.values())
            if combined_text_for_mcq:
                mcq_text = generate_mcqs_from_context(combined_text_for_mcq, num_questions)
                if mcq_text:
                    questions = parse_mcqs(mcq_text)
                    if questions:
                        st.session_state.mcq_questions = questions
                        st.session_state.user_mcq_answers = [None] * len(questions)
                        st.session_state.mcq_submitted = False
                    else:
                        st.warning("No MCQs could be parsed.")
                else:
                    st.warning("No MCQs could be generated.")
            else:
                st.warning("No document text available to generate MCQs from.")

        if 'mcq_questions' in st.session_state and st.session_state.mcq_questions:
            questions = st.session_state.mcq_questions
            for i, q in enumerate(questions):
                st.markdown(f"Q{i+1}: {q['question']}")
                st.session_state.user_mcq_answers[i] = st.radio(
                    "Select your answer:",
                    options=['A', 'B', 'C', 'D'],
                    key=f"mcq_{i}"
                )
                st.markdown(f"A) {q['options'][0]}")
                st.markdown(f"B) {q['options'][1]}")
                st.markdown(f"C) {q['options'][2]}")
                st.markdown(f"D) {q['options'][3]}")
                st.markdown('---')

            if st.button("Submit Quiz"):
                st.session_state.mcq_submitted = True

            if st.session_state.get('mcq_submitted', False):
                score = 0
                questions = st.session_state.mcq_questions
                user_answers = st.session_state.user_mcq_answers
                for i, q in enumerate(questions):
                    is_correct = user_answers[i] == q['answer']
                    st.markdown(f"Q{i+1}: {q['question']}")
                    for idx, opt in enumerate(['A', 'B', 'C', 'D']):
                        option_text = f"{opt}) {q['options'][idx]}"
                        if user_answers[i] == opt:
                            if is_correct:
                                st.markdown(f"<span style='color: green; font-weight: bold'>Your answer: {option_text} ✔</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span style='color: red; font-weight: bold'>Your answer: {option_text} ❌</span>", unsafe_allow_html=True)
                        elif not is_correct and q['answer'] == opt:
                            st.markdown(f"<span style='color: green; font-weight: bold'>Correct answer: {option_text} ✔</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"{option_text}")
                    st.markdown('---')
                    if is_correct:
                        score += 1
                st.success(f"Your score: {score}/{len(questions)}")

if __name__ == "__main__":
    main()
