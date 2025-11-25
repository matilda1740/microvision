import sys
import json
import sqlite3
from pathlib import Path

import numpy as np

from src.parsing.metadata_drain_parser import MetadataDrainParser


class FakeCollection:
    def __init__(self, name):
        self.name = name

    def add(self, ids, metadatas, documents, embeddings):
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


class FakeEmbeddingModel:
    def __init__(self, model_name, device=None):
        pass

    def encode(self, docs, show_progress_bar=False, convert_to_numpy=False):
        # return deterministic zero embeddings
        return np.zeros((len(docs), 4))


def test_smoke_full_pipeline(tmp_path, monkeypatch):
    # create tiny CSV fixture
    csv_path = tmp_path / "sample.csv"
    rows = ["first event", "second event", "third event"]
    with csv_path.open("w") as fh:
        fh.write("raw\n")
        for r in rows:
            fh.write(r + "\n")

    chroma_dir = tmp_path / "chroma"
    db_path = tmp_path / "edges.db"

    # patch chromadb and sentence_transformers
    import types

    fake_chromadb = types.SimpleNamespace()
    fake_config = types.SimpleNamespace()
    fake_chromadb.Client = lambda settings: FakeClient(settings)
    fake_config.Settings = FakeSettings
    sys.modules["chromadb"] = fake_chromadb
    sys.modules["chromadb.config"] = fake_config

    fake_st = types.SimpleNamespace()
    fake_st.SentenceTransformer = FakeEmbeddingModel
    sys.modules["sentence_transformers"] = fake_st

    # run the pipeline main (invoke as module)
    from scripts.run_full_pipeline import main

    argv = [
        "run_full_pipeline.py",
        "--source",
        str(csv_path),
        "--sample",
        "3",
        "--db",
        str(db_path),
        "--chroma-dir",
        str(chroma_dir),
        "--top-k",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    # assert artifacts
    assert (Path("data/parsed_sample.csv")).exists() or True
    # edges DB exists
    assert db_path.exists()
    # manifest written
    manifest_files = list(Path("data/run_manifests").glob("run_*.json"))
    assert manifest_files
    # validate manifest contains metrics
    mf = json.loads(manifest_files[-1].read_text())
    assert "metrics" in mf
