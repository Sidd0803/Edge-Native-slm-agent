import sys
from pathlib import Path

# Ensure the project root is on sys.path so agent.* imports resolve.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import chromadb
from unittest.mock import patch

from agent.memory import PartsMemory, COLLECTION_NAME, _part_to_text

# nomic-embed-text produces 768-dimensional vectors.
MOCK_EMBEDDING = [0.1] * 768

SAMPLE_PARTS = [
    {
        "sku": "TEST-0001-A",
        "description": "Hydraulic fitting 3/8 inch NPT male",
        "category": "hydraulic fittings",
        "stock_qty": 50,
        "unit_price": 12.50,
        "supplier": "Parker Hannifin",
        "supplier_lead_days": 5,
        "reorder_threshold": 10,
    },
    {
        "sku": "TEST-0002-B",
        "description": "Solenoid valve industrial grade",
        "category": "pneumatic valves",
        "stock_qty": 0,
        "unit_price": 149.99,
        "supplier": "SMC",
        "supplier_lead_days": 7,
        "reorder_threshold": 5,
    },
    {
        "sku": "TEST-0003-C",
        "description": "O-Ring kit standard size",
        "category": "seals and gaskets",
        "stock_qty": 120,
        "unit_price": 8.75,
        "supplier": "Eaton",
        "supplier_lead_days": 3,
        "reorder_threshold": 20,
    },
]


@pytest.fixture
def mock_embed():
    """Patch agent.memory._embed so tests never call Ollama."""
    with patch("agent.memory._embed", return_value=MOCK_EMBEDDING):
        yield


@pytest.fixture
def temp_memory(tmp_path, mock_embed):
    """
    A PartsMemory instance backed by a throwaway ChromaDB with SAMPLE_PARTS
    loaded. mock_embed must be a dependency so _embed is patched for the
    entire test, not just during fixture setup.
    """
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection(COLLECTION_NAME)

    for part in SAMPLE_PARTS:
        collection.add(
            ids=[part["sku"]],
            embeddings=[MOCK_EMBEDDING],
            metadatas=[dict(part)],
            documents=[_part_to_text(part)],
        )

    mem = object.__new__(PartsMemory)
    mem._client = client
    mem._collection = collection
    return mem
