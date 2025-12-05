from llama_index.core import SimpleDirectoryReader
# from llama_index.readers.remote_depth import RemoteDepthReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import Document, MetadataMode
import chromadb
import os
import patoolib

import constants

SUPPORTBUNDLE_DIR = "./bundles/"
SUPPORTBUNDLE_PATH =  "20251124.zip"
CHROMA_SB_DIR = "./db/chroma_sb"
PERSISTENT_DIR = "./db/chromadb_sb"

def extract_bundle(path):
    print(f"[+] Extracting support bundle: {path}")
    os.makedirs(SUPPORTBUNDLE_DIR, exist_ok=True)

    patoolib.extract_archive(path, outdir=SUPPORTBUNDLE_DIR)
    for root, _, files in os.walk(SUPPORTBUNDLE_DIR):
        if "__MACOSX" in root:
            continue
        for file in files:
            if file.startswith("."):
                continue

            full_path = root + os.sep + file
            if patoolib.is_archive(full_path):
                patoolib.extract_archive(full_path, outdir=path.split(".zip", 1)[0])
    return SUPPORTBUNDLE_DIR

# def collect_text_files(path):
#     print("[+] Searching for log/YAML files...")
#     extensions = [".log", ".txt", ".yaml", ".yml", ".json"]
#
#     files = []
#     for root, _, filenames in os.walk(path):
#         for name in filenames:
#             if "__MACOSX" in root:
#                 continue
#             # Skip hidden files
#             if name.startswith("."):
#                 continue
#             if name.endswith(extensions):
#                 full_path = os.path.join(root, name)
#                 files.append(full_path)
#
#     print(f"[+] Found {len(files)} relevant files.")
#     return files

def load_documents():
    extract_bundle(os.path.join(SUPPORTBUNDLE_DIR, SUPPORTBUNDLE_PATH))
    print("[+] Loading Documents using LlamaIndex...")
    reader = SimpleDirectoryReader(
        input_dir=SUPPORTBUNDLE_DIR,
        required_exts=[".log", ".txt", ".yaml", ".yml", ".json"],
        recursive=True,
    )
    documents = reader.load_data()
    print(f"[+] Loaded {len(documents)} documents")
    return documents


def chunk_content(content):
    """Convert raw documents into smaller chunks (nodes) for embedding."""
    print("[+] Chunking the loaded documents...")
    print(f"[+] Total documents to process: {len(content)}")

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    all_nodes = []
    batch_size = 50

    for i in range(0, len(content), batch_size):
        batch = content[i:i+batch_size]
        batch = content[i:i+batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(content) + batch_size - 1)//batch_size
        print(f"[+] Processing batch {batch_num}/{total_batches}")

        # Process each document individually in this batch
        for doc_idx, d in enumerate(batch):
            global_idx = i + doc_idx

            if not d.text or len(d.text.strip()) < 10:
                continue

            # Log progress for each document
            text_len = len(d.text)
            if text_len > 500_000:
                print(f"    [!] Truncating from {text_len} to 500k chars")
                truncated_text = d.text[:500_000]
            else:
                truncated_text = d.text

            try:
                doc_obj = Document(text=truncated_text, metadata=d.metadata)
                nodes = splitter.get_nodes_from_documents([doc_obj])
                all_nodes.extend(nodes)
                print(f"    [+] Created {len(nodes)} chunks")
            except Exception as e:
                print(f"    [!] Error processing doc {global_idx}: {e}")
                continue

            print(f"[+] Batch {batch_num} complete (total: {len(all_nodes)} chunks)")

    print(f"[+] Created {len(all_nodes)} total chunks")
    return all_nodes

def embed_nodes(model, nodes):
    """Generate vector embeddings for each chunk/node using a HF embedding model."""
    print("[+] Embedding the nodes...")
    print(f"[+] Total nodes to embed: {len(nodes)}")

    texts = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes]

    # Batch embedding is MUCH faster
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"  [+] Embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({i}/{len(texts)})")

        try:
            # Batch embedding
            batch_embeddings = model.get_text_embedding_batch(batch_texts)

            # Convert to list
            for emb in batch_embeddings:
                if hasattr(emb, 'tolist'):
                    all_embeddings.append(emb.tolist())
                else:
                    all_embeddings.append(list(emb))

        except Exception as e:
            print(f"  [!] Error in batch {i//batch_size + 1}: {e}")
            # Fall back to individual embedding for this batch
            for text in batch_texts:
                try:
                    emb = model.get_text_embedding(text)
                    if hasattr(emb, 'tolist'):
                        all_embeddings.append(emb.tolist())
                    else:
                        all_embeddings.append(list(emb))
                except:
                    all_embeddings.append([0.0] * 384)

    print(f"[+] Successfully embedded {len(all_embeddings)} nodes")
    return all_embeddings

client = chromadb.PersistentClient(PERSISTENT_DIR)

def store_in_db(nodes, embeddings):
    """Store node embeddings and metadata into a persistent ChromaDB collection."""
    print("[+] Storing vectors in ChromaDB...")
    try:
        existing = client.get_collection(name=constants.COLLECTION_SB)
        client.delete_collection(existing.name)
        print("[+] Deleted existing collection...")
    except:
        pass

    collection = client.get_or_create_collection(
        name=constants.COLLECTION_SB,
    )

    # ChromaDB has a max batch size limit - split into chunks
    batch_size = 5000
    total_nodes = len(nodes)

    for i in range(0, total_nodes, batch_size):
        end_idx = min(i + batch_size, total_nodes)
        batch_nodes = nodes[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]

        # Extract metadata from nodes
        batch_metadatas = []
        for node in batch_nodes:
            metadata = {
                "source": node.metadata.get("source", "N/A"),
                "title": node.metadata.get("title", "N/A"),
                # Add any other metadata fields you need
            }
            batch_metadatas.append(metadata)

        print(f"[+] Adding batch {i//batch_size + 1}: items {i} to {end_idx-1}")

        collection.add(
            ids=[f"node-{j}" for j in range(i, end_idx)],
            embeddings=batch_embeddings,
            documents=[node.get_content() for node in batch_nodes],
            metadatas=batch_metadatas,  # ADD THIS
        )

    print(f"[+] Successfully stored {total_nodes} vectors in ChromaDB")


# def query_db(model, data):
#     """Query ChromaDB using HF embedding of the user prompt and print top matches."""
#     print("[+] Querying ChromaDB...")
#
#     try:
#         collection = client.get_collection("supportbundle")
#     except Exception as e:
#         print(f"[!] Error: Collection 'supportbundle' not found: {e}")
#         return None
#
#     # Handle both string and list input
#     query_text = data[0] if isinstance(data, list) else data
#
#     # Get embedding
#     query_embedding = model.get_text_embedding(query_text)
#     if hasattr(query_embedding, 'tolist'):
#         query_embedding = query_embedding.tolist()
#
#     # Query the collection
#     try:
#         results = collection.query(
#             query_embeddings=[query_embedding],
#             n_results=5
#         )
#     except Exception as e:
#         print(f"[!] Query failed: {e}")
#         return None
#
#     # Check if we got results
#     if not results["documents"][0]:
#         print("No results found.")
#         return results
#
#     # Display results
#     print(f"\nQuery: {query_text}\n")
#     print("="*80)
#
#     for i, doc in enumerate(results["documents"][0]):
#         metadata = results["metadatas"][0][i] if results["metadatas"] else {}
#         distance = results["distances"][0][i]
#
#         # Convert distance to similarity score (0-1, higher = more similar)
#         # For L2 distance, you can use: similarity = 1 / (1 + distance)
#         similarity = 1 / (1 + distance)
#
#         print(f"\n--- Result {i+1} ---")
#         print(f"Distance: {distance:.4f} | Similarity: {similarity:.4f}")
#         print(f"Title: {metadata.get('title', 'N/A')}")
#         print(f"Source: {metadata.get('source', 'N/A')}")
#         print(f"Content snippet:\n{doc[:300]}...")
#         print("-"*80)
#
#     return results

# def query_db(model, data):
#     """Query ChromaDB using HF embedding of the user prompt and print top matches."""
#     # client = chromadb.PersistentClient(path="./kb_vector_db")
#     print("[+] Querying ChromadDB...")
#     collection = client.get_collection("supportbundle")
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
    documents = load_documents()
    nodes = chunk_content(documents)
    print(len(nodes))
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en")
    # # embeddings = embed_nodes(embed_model, nodes)
    store_in_db(nodes, embed_nodes(embed_model, nodes))
    # query_db(embed_model, ["backup target error recurring failure"])

if __name__ == "__main__":
    main()
