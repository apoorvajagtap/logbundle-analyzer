# runtime/query_engine.py
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List, Dict
import constants

# init embedding model (same as ingestion)
embed_model = HuggingFaceEmbedding(model_name=constants.EMBED_MODEL_NAME)

# persistent clients
_client_kb = chromadb.PersistentClient(constants.CHROMA_KB_DIR)
_client_sb = chromadb.PersistentClient(constants.CHROMA_SB_DIR)

def _get_collection(client: chromadb.PersistentClient, name: str):
    try:
        return client.get_collection(name)
    except Exception:
        return None

def embed_text(text: str) -> List[float]:
    emb = embed_model.get_text_embedding(text)
    if hasattr(emb, "tolist"):
        emb = emb.tolist()
    return emb

def retrieve_kb(query: str, top_k: int = None) -> List[Dict]:
    """Return list of dicts: {document, metadata, distance}"""
    top_k = top_k or constants.TOP_K_KB
    kb_col = _get_collection(_client_kb, constants.COLLECTION_KB)
    if not kb_col:
        return []
    raw_kb = kb_col.query(query_texts=[query], n_results=top_k)
    if not raw_kb or not raw_kb.get("documents"):
        return []
    docs = []
    docs_list = raw_kb["documents"][0]
    dist_list = raw_kb.get("distances", [[]])[0]
    for i in range(len(docs_list)):
        docs.append({
            "document": docs_list[i],
            "distance": dist_list[i] if i < len(dist_list) else None,
        })
    return docs

def retrieve_sb_logs(query: str, top_k: int = None) -> List[Dict]:
    top_k = top_k or constants.TOP_K_SB
    sb_col = _get_collection(_client_sb, constants.COLLECTION_SB)
    if not sb_col:
        return []
    raw_sb = sb_col.query(query_texts=[query], n_results=top_k)
    if not raw_sb or not raw_sb.get("documents"):
        return []

    docs = []
    docs_list = raw_sb["documents"][0]
    dist_list = raw_sb.get("distances", [[]])[0]

    for i in range(len(docs_list)):
        docs.append({
            "document": docs_list[i],
            "distance": dist_list[i] if i < len(dist_list) else None,
        })

    return docs

def normalize_chroma_results(raw):
    """
    Convert Chroma query output into the structure expected by llm_reasoner.
    """
    hits = []

    docs = raw.get("documents", [])
    metas = raw.get("metadatas", [])

    for doc, meta in zip(docs, metas):
        hits.append({
            "document": doc,
            "metadata": meta,
        })

    return hits
