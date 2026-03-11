import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# LLM Settings
GROQ_MODEL = "llama-3.3-70b-versatile"
OLLAMA_MODEL = "llama3.2"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# Embedding Settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Qdrant Settings
QDRANT_PATH = "./qdrant_local_db"
QDRANT_COLLECTION = "rag_docs"

# Pinecone Settings
PINECONE_INDEX  = "rag-chatbot"
PINECONE_CLOUD  = "aws"
PINECONE_REGION = "us-east-1"

# Chunking Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval Settings
TOP_K = 5

# Tesseract path (Windows)
TESSERACT_PATH = r"C:/Program Files/Tesseract-OCR/"

# Set True during development to always start fresh
RESET_ON_START = False