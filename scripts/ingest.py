import csv
import sys
from pathlib import Path

import chromadb
import ollama

DATA_PATH = Path(__file__).parent.parent / "data" / "parts_catalog.csv"
CHROMA_PATH = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "parts_catalog"
EMBED_MODEL = "nomic-embed-text"


def embed(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def part_to_text(row: dict) -> str:
    return f"{row['description']} | {row['category']} | SKU: {row['sku']}"


def ingest():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found. Generate the catalog first.")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Cleared existing '{COLLECTION_NAME}' collection.")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Ingesting {len(rows)} parts into ChromaDB...")

    ids, embeddings, metadatas, documents = [], [], [], []

    for i, row in enumerate(rows):
        text = part_to_text(row)
        embedding = embed(text)

        ids.append(row["sku"])
        embeddings.append(embedding)
        metadatas.append({
            "sku": row["sku"],
            "description": row["description"],
            "category": row["category"],
            "stock_qty": int(row["stock_qty"]),
            "unit_price": float(row["unit_price"]),
            "supplier": row["supplier"],
            "supplier_lead_days": int(row["supplier_lead_days"]),
            "reorder_threshold": int(row["reorder_threshold"]),
        })
        documents.append(text)

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} embedded...")

    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    print(f"\nDone. {len(rows)} parts ingested into '{COLLECTION_NAME}'.")


def verify(query: str = "3/8 inch hydraulic fitting"):
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    embedding = embed(query)
    results = collection.query(query_embeddings=[embedding], n_results=3, include=["metadatas", "distances"])

    print(f"\nRetrieval test — query: \"{query}\"")
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        print(f"  [{dist:.3f}] {meta['sku']} — {meta['description']}")


if __name__ == "__main__":
    ingest()
    verify()
