import os
import time
import multiprocessing
from pathlib import Path

import pytest

from tests.utils.import_helpers import import_src_submodule

# Import the chroma module directly from src/ using the helper
chroma = import_src_submodule("encode_chroma.chroma")
import numpy as np


def _hold_lock(lock_path: str, hold_seconds: float) -> None:
    """Helper run in a separate process that locks the given path for a while."""
    import portalocker

    # Ensure parent dir exists
    p = Path(lock_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    try:
        portalocker.lock(fh, portalocker.LOCK_EX)
        time.sleep(hold_seconds)
    finally:
        try:
            portalocker.unlock(fh)
        except Exception:
            pass
        fh.close()


@pytest.mark.unit
def test_chroma_lock_strict_mode_fails_when_locked(tmp_path, monkeypatch):
    # Skip the test if portalocker is not available in the test environment
    pytest.importorskip("portalocker")

    chroma_dir = tmp_path / "chroma_collection"
    chroma_dir_parent = chroma_dir.parent
    lock_path = chroma_dir_parent / f".{chroma_dir.name}.lock"

    # Start a process that acquires the lock and holds it for 2 seconds
    p = multiprocessing.Process(target=_hold_lock, args=(str(lock_path), 2.0))
    p.start()
    # Give the process a moment to acquire the lock
    time.sleep(0.2)

    # Force a short timeout and strict mode
    monkeypatch.setenv("CHROMA_LOCK_TIMEOUT", "0.2")
    monkeypatch.setenv("CHROMA_LOCK_STRICT", "1")

    # Provide a minimal fake chromadb implementation so the function can proceed
    class FakeConfig:
        class Settings:
            def __init__(self, *args, **kwargs):
                pass

    class FakeCollection:
        def add(self, *args, **kwargs):
            return None

        def query(self, *args, **kwargs):
            return None

    class FakeClient:
        def __init__(self, settings):
            pass

        def get_or_create_collection(self, name):
            return FakeCollection()

        def create_collection(self, name):
            return FakeCollection()

        def persist(self):
            return None

    fake_chromadb = type("FakeChroma", (), {"config": FakeConfig, "Client": FakeClient})

    monkeypatch.setattr(chroma, "_lazy_import_chromadb", lambda: (fake_chromadb, None))

    # Call ingest and expect a RuntimeError due to strict lock
    with pytest.raises(RuntimeError):
        chroma.ingest_to_chroma_atomic(
            chroma_dir=chroma_dir,
            ids=["a"],
            metadatas=[{"k": "v"}],
            documents=["doc"],
            embeddings=np.array([[0.1, 0.2]]),
            collection_name="test",
            manifest={"m": 1},
        )

    p.join(timeout=5)
    if p.is_alive():
        p.terminate()


@pytest.mark.unit
def test_chroma_lock_non_strict_mode_proceeds_when_locked(tmp_path, monkeypatch):
    pytest.importorskip("portalocker")

    chroma_dir = tmp_path / "chroma_collection"
    chroma_dir_parent = chroma_dir.parent
    lock_path = chroma_dir_parent / f".{chroma_dir.name}.lock"

    # Start a process that acquires the lock and holds it for 1 second
    p = multiprocessing.Process(target=_hold_lock, args=(str(lock_path), 1.0))
    p.start()
    time.sleep(0.1)

    # Short timeout but non-strict: should proceed without raising
    monkeypatch.setenv("CHROMA_LOCK_TIMEOUT", "0.2")
    monkeypatch.setenv("CHROMA_LOCK_STRICT", "0")

    class FakeConfig:
        class Settings:
            def __init__(self, *args, **kwargs):
                pass

    class FakeCollection:
        def add(self, *args, **kwargs):
            return None

        def query(self, *args, **kwargs):
            return None

    class FakeClient:
        def __init__(self, settings):
            pass

        def get_or_create_collection(self, name):
            return FakeCollection()

        def create_collection(self, name):
            return FakeCollection()

        def persist(self):
            return None

    fake_chromadb = type("FakeChroma", (), {"config": FakeConfig, "Client": FakeClient})
    monkeypatch.setattr(chroma, "_lazy_import_chromadb", lambda: (fake_chromadb, None))

    client, collection = chroma.ingest_to_chroma_atomic(
        chroma_dir=chroma_dir,
        ids=["a"],
        metadatas=[{"k": "v"}],
        documents=["doc"],
        embeddings=np.array([[0.1, 0.2]]),
        collection_name="test",
        manifest={"m": 1},
    )

    # Should return a client object (our fake) and a collection (or None if query missing)
    assert client is not None

    p.join(timeout=5)
    if p.is_alive():
        p.terminate()
