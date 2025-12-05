# constants.py
import os

# Chroma persistent directories or path used when creating PersistentClient
CHROMA_KB_DIR = os.getenv("CHROMA_KB_DIR", "./chromadb_longhorn_kb")
CHROMA_SB_DIR = os.getenv("CHROMA_SB_DIR", "./chromadb_sb_logs")

# Chroma collection names
COLLECTION_KB = "longhorn_kb"
COLLECTION_SB = "sb_logs"

# Embedding model wrapper, used HuggingFaceEmbedding
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-base-en")

# Generative model, used Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
GENERATIVE_MODEL = "llama3.1"

# Retrieval settings
TOP_K_KB = int(os.getenv("TOP_K_KB", 4))
TOP_K_SB = int(os.getenv("TOP_K_SB", 6))
