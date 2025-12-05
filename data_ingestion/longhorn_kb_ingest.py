from llama_index.readers.remote_depth import RemoteDepthReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import Document, MetadataMode
import chromadb

import constants

PERSISTENT_DIR = "./db/chromadb_longhorn_kb"

def load_kb_links(url):
    """Fetch Longhorn KB pages recursively using RemoteDepthReader."""
    print("[+] Loading KB pages using LlamaIndex...")
    reader = RemoteDepthReader(
        depth=2,
        domain_lock=True,
    )
    documents = reader.load_data(url)
    return documents

def chunk_content(content):
    """Convert raw documents into smaller chunks (nodes) for embedding."""
    print("[+] Chunking the loaded documents...")
    docs = [
        Document(text=d.text)
        for d in content
    ]
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(docs)
    return nodes

def embed_nodes(model, nodes):
    """Generate vector embeddings for each chunk/node using a HF embedding model."""
    print("[+] Embedding the nodes...")
    texts = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes]
    # Get embeddings as a list of lists
    embeddings = []
    for text in texts:
        emb = model.get_text_embedding(text)
        # Convert numpy array to list if needed
        if hasattr(emb, 'tolist'):
            emb = emb.tolist()
        embeddings.append(emb)
    return embeddings

client = chromadb.PersistentClient(PERSISTENT_DIR)

def store_in_db(nodes, embeddings):
    """Store node embeddings and metadata into a persistent ChromaDB collection."""
    print("[+] Storing vectors in ChromaDB...")
    try:
        existing = client.get_collection(name=constants.COLLECTION_KB)
        client.delete_collection(existing.name)
        print("[+] Deleted existing collection...")
    except:
        pass

    collection = client.get_or_create_collection(
        name=constants.COLLECTION_KB,
    )
    collection.add(
        ids=[f"node-{i}" for i in range(len(nodes))],
        embeddings=embeddings,
        # metadatas=[node.metadata for node in nodes],
        documents=[node.get_content() for node in nodes],
    )

# def query_db(model, data):
#     """Query ChromaDB using HF embedding of the user prompt and print top matches."""
#     # client = chromadb.PersistentClient(path="./kb_vector_db")
#     print("[+] Querying ChromadDB...")
#     collection = client.get_collection("longhorn_kb")
#
#     query_text = data[0] if isinstance(data, list) else data
#     query_embedding = model.get_text_embedding(query_text)
#
#     if hasattr(query_embedding, 'tolist'):
#         query_embedding = query_embedding.tolist()
#
#     results = collection.query(
#         # query_texts=data,
#         query_embeddings=[query_embedding],
#         n_results=5
#     )
#     print(f"\nQuery: {data}\n")
#     for i, doc in enumerate(results["documents"][0]):
#         metadata = results["metadatas"][0][i]
#         distance = results["distances"][0][i]
#         print(f"--- Result {i+1} ---")
#         print(f"Similarity score: {distance:.4f}")
#         print(f"Title: {metadata.get('title', 'N/A')}")
#         print(f"URL: {metadata.get('source', 'N/A')}")
#         print(f"Content snippet: {doc[:300]}...")
#         print()

def main():
    documents = load_kb_links("https://longhorn.io/kb/")
    nodes = chunk_content(documents)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en")
    # embeddings = embed_nodes(embed_model, nodes)
    store_in_db(nodes, embed_nodes(embed_model, nodes))
    # query_db(embed_model, ["backup target error recurring failure"])

if __name__ == "__main__":
    main()
