from pathlib import Path

import chromadb
import ollama

CHROMA_PATH = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "parts_catalog"
EMBED_MODEL = "nomic-embed-text"


def _embed(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def _part_to_text(part: dict) -> str:
    return f"{part['description']} | {part['category']} | SKU: {part['sku']}"


class PartsMemory:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self._collection = self._client.get_collection(COLLECTION_NAME)

    def search(self, query: str, n: int = 3) -> list[dict]:
        embedding = _embed(query)
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["metadatas", "distances"],
        )
        return results["metadatas"][0]

    def get_by_sku(self, sku: str) -> dict | None:
        results = self._collection.get(ids=[sku], include=["metadatas"])
        if results["metadatas"]:
            return results["metadatas"][0]
        return None

    def add_part(self, part: dict):
        if self.get_by_sku(part["sku"]) is not None:
            raise ValueError(f"SKU {part['sku']} already exists — use update_part to modify it")
        text = _part_to_text(part)
        self._collection.add(
            ids=[part["sku"]],
            embeddings=[_embed(text)],
            metadatas=[part],
            documents=[text],
        )

    def update_part(self, sku: str, fields: dict):
        existing = self.get_by_sku(sku)
        if existing is None:
            raise KeyError(f"SKU {sku} not found")
        for field, value in fields.items():
            if field not in existing:
                raise KeyError(f"Unknown field '{field}'")
            expected_type = type(existing[field])
            if not isinstance(value, expected_type):
                try:
                    fields = {**fields, field: expected_type(value)}
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Field '{field}' expects {expected_type.__name__}, got '{value}'"
                    )
        existing.update(fields)
        text = _part_to_text(existing)
        self._collection.update(
            ids=[sku],
            embeddings=[_embed(text)],
            metadatas=[existing],
            documents=[text],
        )

    def remove_part(self, sku: str):
        if self.get_by_sku(sku) is None:
            raise KeyError(f"SKU {sku} not found")
        self._collection.delete(ids=[sku])

    def list_by_category(self, category: str) -> list[dict]:
        results = self._collection.get(
            where={"category": {"$eq": category}},
            include=["metadatas"],
        )
        return results["metadatas"]
