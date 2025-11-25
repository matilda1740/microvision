import json
import sys
from pathlib import Path

import numpy as np

from src.encode_chroma.chroma import ingest_to_chroma_atomic


class FakeCollection:
    def __init__(self, name):
        self.name = name

    def add(self, ids, metadatas, documents, embeddings):
        # simple no-op
        self._count = len(ids)


class FakeClient:
    def __init__(self, settings):
        self._settings = settings
        self._collections = {}

    def list_collections(self):
        return [FakeCollection(n) for n in self._collections.keys()]

    def create_collection(self, name):
        col = FakeCollection(name)
        self._collections[name] = col
        return col

    def get_or_create_collection(self, name):
        return self.create_collection(name)

    def delete_collection(self, name):
        if name in self._collections:
            del self._collections[name]

    def persist(self):
        return True

    def get_collection(self, name):
        return self._collections.get(name)


class FakeSettings:
    def __init__(self, persist_directory):
        self.persist_directory = persist_directory


def test_ingest_atomic(tmp_path, monkeypatch):
    # inject fake chromadb module
    fake_chromadb = type(sys)("chromadb")
    fake_config = type(sys)("chromadb.config")

    def fake_client_factory(settings):
        return FakeClient(settings)

    fake_chromadb.Client = fake_client_factory
    fake_config.Settings = FakeSettings

    sys.modules["chromadb"] = fake_chromadb
    sys.modules["chromadb.config"] = fake_config

    collection_dir = tmp_path / "chroma_test"
    ids = ["0", "1", "2"]
    metadatas = [{"id": i} for i in ids]
    documents = ["a", "b", "c"]
    embeddings = np.zeros((3, 5))

    client_final, collection_final = ingest_to_chroma_atomic(collection_dir, ids, metadatas, documents, embeddings)

    # ensure final collection dir exists or tmp was used
    assert collection_dir.exists() or str(collection_dir).endswith("_tmp")
    # manifest should exist inside the final dir
    manifest_path = collection_dir / "manifest.json"
    if not manifest_path.exists():
        # fallback: tmp moved into place; check for manifest in any child
        matches = list(collection_dir.rglob("manifest.json"))
        assert matches, "manifest.json not found"
        manifest_path = matches[0]

    data = json.loads(manifest_path.read_text())
    assert data.get("count") == len(ids)
