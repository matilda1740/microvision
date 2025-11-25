import sys
import types
import tempfile
from pathlib import Path


def test_ingest_to_chroma_atomic_requires_chromadb(monkeypatch):
    from src.encode_chroma.chroma import ingest_to_chroma_atomic

    # force the lazy importer to raise
    monkeypatch.setattr('src.encode_chroma.chroma._lazy_import_chromadb', lambda: (_ for _ in ()).throw(RuntimeError("no chroma")))

    try:
        with tempfile.TemporaryDirectory() as td:
            Path(td).mkdir(exist_ok=True)
            try:
                ingest_to_chroma_atomic(Path(td) / "col", ['1'], [{'a': 1}], ['doc'], __import__('numpy').array([[1.0]]))
                assert False, "expected ingest_to_chroma_atomic to raise when chromadb missing"
            except RuntimeError:
                pass
    finally:
        # ensure monkeypatch cleanup handled by pytest
        pass


def test_ingest_to_chroma_atomic_with_fake_chromadb(monkeypatch, tmp_path):
    # Provide a minimal fake chromadb with the methods the helper expects.
    fake = types.SimpleNamespace()

    class FakeCollection:
        def __init__(self, name):
            self.name = name

        def add(self, **kwargs):
            # store but do nothing
            self._added = kwargs

    class FakeClient:
        def __init__(self, settings):
            self._settings = settings
            self._collections = {}

        def create_collection(self, name=None):
            col = FakeCollection(name)
            self._collections[name] = col
            return col

        def get_or_create_collection(self, name=None):
            return self._collections.get(name) or self.create_collection(name)

        def get_collection(self, name=None):
            return self._collections.get(name)

        def list_collections(self):
            return list(self._collections.values())

        def persist(self):
            return True

    fake.config = types.SimpleNamespace(Settings=lambda **kwargs: types.SimpleNamespace(**kwargs))
    fake.Client = FakeClient
    fake.utils = types.SimpleNamespace(embedding_functions=types.SimpleNamespace())

    # inject into sys.modules so the lazy importer finds it
    monkeypatch.setitem(sys.modules, 'chromadb', fake)
    monkeypatch.setitem(sys.modules, 'chromadb.config', fake.config)
    monkeypatch.setitem(sys.modules, 'chromadb.utils', fake.utils)

    from src.encode_chroma.chroma import ingest_to_chroma_atomic

    out_dir = tmp_path / "chroma_test"
    out_dir.mkdir()
    ids = ["0"]
    metadatas = [{"a": 1}]
    docs = ["doc"]
    import numpy as np

    client, collection = ingest_to_chroma_atomic(out_dir, ids, metadatas, docs, np.array([[1.0, 0.0]]), collection_name="testcol")

    assert client is not None
    # collection should be a FakeCollection instance or None depending on code path
    assert (collection is None) or hasattr(collection, 'name')
