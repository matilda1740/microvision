"""Helpers around Chromadb local collections and atomic ingestion.

This module performs lazy imports of chromadb to avoid requiring it for tests
that don't touch the vector DB. Ingest is done by writing to a temporary
directory and moving into place to avoid partial states.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import logging

# Cache chromadb clients per persist directory so interactive runs reuse the
# same client object and avoid chromadb errors about multiple instances with
# differing settings in the same process.
# Key: str(persist_directory) -> chromadb.Client
_CHROMA_CLIENT_CACHE = {}


def _lazy_import_chromadb():
    try:
        import chromadb
        try:
            from chromadb.utils import embedding_functions
        except Exception:
            embedding_functions = None

        return chromadb, embedding_functions
    except Exception as e:
        raise RuntimeError("chromadb is required for Chroma operations") from e


def load_chroma_collection(chroma_dir: Path, collection_name: str):
    chromadb, _ = _lazy_import_chromadb()
    # Be permissive about the Settings constructor signature (real chromadb
    # and test fakes may differ).
    try:
        config_mod = getattr(chromadb, "config")
    except Exception:
        import importlib

        config_mod = importlib.import_module("chromadb.config")

    try:
        settings_obj = config_mod.Settings(persist_directory=str(chroma_dir))
    except TypeError:
        try:
            settings_obj = config_mod.Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(chroma_dir))
        except TypeError:
            settings_obj = config_mod.Settings(str(chroma_dir))

    client = chromadb.Client(settings_obj)
    try:
        col = client.get_collection(name=collection_name)
    except Exception:
        col = None
    return client, col


def ingest_to_chroma_atomic(
    chroma_dir: Path,
    ids: Iterable[str],
    metadatas: Iterable[dict],
    documents: Iterable[str],
    embeddings: np.ndarray,
    collection_name: str = "smoke_collection",
    manifest: Optional[dict] = None,
    overwrite: bool = True,
) -> Tuple[object, object]:
    """Atomically write a chroma collection by creating a temp dir and moving it.

    Args:
        chroma_dir: final persist directory
        ids, metadatas, documents, embeddings: data to add
        collection_name: collection name in Chroma
        manifest: optional dict to write into the collection directory as manifest.json
        overwrite: if True, remove any existing collection dir before move

    Returns (client, collection) connected to the final persisted directory.
    """
    chromadb, embedding_functions = _lazy_import_chromadb()

    chroma_dir = Path(chroma_dir)
    chroma_dir_parent = chroma_dir.parent
    chroma_dir_parent.mkdir(parents=True, exist_ok=True)

    final_client = None
    final_collection = None

    # Lock configuration via environment variables
    import os, time

    try:
        LOCK_TIMEOUT = float(os.getenv("CHROMA_LOCK_TIMEOUT", "5"))
    except Exception:
        LOCK_TIMEOUT = 5.0
    LOCK_STRICT = str(os.getenv("CHROMA_LOCK_STRICT", "0")).lower() in ("1", "true", "yes")

    # Attempt to import portalocker; if missing, behavior depends on LOCK_STRICT
    portalocker = None
    try:
        import portalocker as _portalocker

        portalocker = _portalocker
    except Exception:
        if LOCK_STRICT:
            raise

    # If portalocker available, try to acquire an exclusive lock with retries
    lock_fh = None
    lock_acquired = False
    lock_path = chroma_dir_parent / f".{chroma_dir.name}.lock"
    if portalocker is not None:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_fh = open(lock_path, "w")
        start = time.time()
        wait = 0.05
        while True:
            try:
                portalocker.lock(lock_fh, portalocker.LOCK_EX | portalocker.LOCK_NB)
                lock_acquired = True
                break
            except Exception:
                if time.time() - start > LOCK_TIMEOUT:
                    if LOCK_STRICT:
                        raise RuntimeError(f"Could not acquire chroma lock within {LOCK_TIMEOUT}s and CHROMA_LOCK_STRICT=1")
                    else:
                        try:
                            import logging

                            logging.getLogger(__name__).warning(
                                "Could not acquire chroma lock within %s seconds; proceeding without lock (CHROMA_LOCK_STRICT=0).",
                                LOCK_TIMEOUT,
                            )
                        except Exception:
                            pass
                        lock_acquired = False
                        break
                time.sleep(wait)
                wait = min(1.0, wait * 2)

    # If a client already exists for the target persist dir, reuse it and add
    # vectors directly to avoid creating a second Chroma instance in-process
    # (which some chromadb versions disallow).
    cached_client = _CHROMA_CLIENT_CACHE.get(str(chroma_dir))
    if cached_client is not None and chroma_dir.exists():
        client = cached_client
        try:
            collection = client.get_or_create_collection(name=collection_name)
        except Exception:
            try:
                collection = client.create_collection(name=collection_name)
            except Exception:
                collection = None

        if collection is not None:
            # Add directly into the existing collection and persist
            try:
                collection.add(ids=list(ids), metadatas=list(metadatas), documents=list(documents), embeddings=np.asarray(embeddings).tolist())
            except Exception:
                # best-effort: try with embeddings as list if asarray fails
                collection.add(ids=list(ids), metadatas=list(metadatas), documents=list(documents), embeddings=list(embeddings))
            try:
                if hasattr(client, "persist"):
                    client.persist()
            except Exception:
                pass

            final_client = client
            final_collection = collection
            # write manifest if requested
            try:
                from src.utils.atomic_write import atomic_write_json

                manifest_to_write = manifest or {"collection_name": collection_name, "count": len(list(ids)), "persist_directory": str(chroma_dir)}
                atomic_write_json(Path(chroma_dir) / "manifest.json", manifest_to_write)
            except Exception:
                pass

            # ensure we still return None for collection if query not available
            if final_collection is not None and not hasattr(final_collection, "query"):
                final_collection = None

            # cache client (already present) and return
            _CHROMA_CLIENT_CACHE[str(chroma_dir)] = final_client
            try:
                logging.getLogger(__name__).info(
                    "Reusing cached chromadb client for %s", str(chroma_dir)
                )
            except Exception:
                pass
            # release advisory lock if we acquired one
            if lock_fh is not None:
                try:
                    if lock_acquired and portalocker is not None:
                        try:
                            portalocker.unlock(lock_fh)
                        except Exception:
                            pass
                    lock_fh.close()
                except Exception:
                    pass

            return final_client, final_collection

    # Create temporary dir, build a client pointing at it, add vectors, persist and move
    with tempfile.TemporaryDirectory(dir=str(chroma_dir_parent)) as tmpdir:
        tmpdir_path = Path(tmpdir) / chroma_dir.name
        tmpdir_path.mkdir(parents=True, exist_ok=True)
        # create client pointing at tmpdir and persist
        # Some test fakes expose chromadb.config as a separate module entry in
        # sys.modules rather than as an attribute on the chromadb module. Be
        # robust and import the config module if needed.
        try:
            config_mod = getattr(chromadb, "config")
        except Exception:
            import importlib

            config_mod = importlib.import_module("chromadb.config")

        try:
            settings_obj = config_mod.Settings(persist_directory=str(tmpdir_path))
        except TypeError:
            try:
                settings_obj = config_mod.Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(tmpdir_path))
            except TypeError:
                settings_obj = config_mod.Settings(str(tmpdir_path))

        client = chromadb.Client(settings_obj)
        # create or reset collection
        try:
            collection = client.get_or_create_collection(name=collection_name)
        except Exception:
            collection = client.create_collection(name=collection_name)

        # chroma expects list-likes for ids/metadatas/documents and embeddings as list of lists
        # Ingest in batches to avoid hitting Chroma's max batch size limit (approx 5k-40k depending on version)
        # and to keep memory usage reasonable.
        BATCH_SIZE = 2000
        ids_list = list(ids)
        metas_list = list(metadatas)
        docs_list = list(documents)
        embs_list = embeddings.tolist()
        
        total = len(ids_list)
        for i in range(0, total, BATCH_SIZE):
            end = min(i + BATCH_SIZE, total)
            collection.add(
                ids=ids_list[i:end],
                metadatas=metas_list[i:end],
                documents=docs_list[i:end],
                embeddings=embs_list[i:end]
            )

        # Persist if the client exposes a persist method; test fakes or
        # different chromadb versions may not implement it the same way.
        try:
            if hasattr(client, "persist"):
                client.persist()
        except Exception:
            # Non-fatal; collection data may already be persisted depending on
            # the chromadb implementation. Continue regardless.
            pass

        # write manifest (provide a minimal default so callers and tests can
        # rely on a manifest being present). Use atomic write to avoid
        # partially-written manifest files.
        try:
            from src.utils.atomic_write import atomic_write_json

            manifest_to_write = manifest or {"collection_name": collection_name, "count": len(list(ids)), "persist_directory": str(tmpdir_path)}
            atomic_write_json(tmpdir_path / "manifest.json", manifest_to_write)
        except Exception:
            pass

        # Move tmpdir atomically into place by deleting old dir then moving
        if chroma_dir.exists() and overwrite:
            shutil.rmtree(chroma_dir)
        shutil.move(str(tmpdir_path), str(chroma_dir))

        # Reuse the client we created that points at the (now moved) data
        # directory. Cache the client for future reuse in this process to
        # avoid creating multiple chromadb instances with conflicting
        # settings.
        final_client = client
        final_collection = collection
        try:
            _CHROMA_CLIENT_CACHE[str(chroma_dir)] = final_client
        except Exception:
            pass
        try:
            logging.getLogger(__name__).info("Created and cached chromadb client for %s", str(chroma_dir))
        except Exception:
            pass
        # If the collection doesn't implement the newer `query` API (some
        # test fakes provide a minimal collection that only supports add),
        # return None for the collection so callers will fall back to the
        # in-memory embeddings path. This keeps behavior consistent across
        # different chromadb implementations and fakes.
        if final_collection is not None and not hasattr(final_collection, "query"):
            final_collection = None

    # release advisory lock if we acquired one
    if lock_fh is not None:
        try:
            if lock_acquired and portalocker is not None:
                try:
                    portalocker.unlock(lock_fh)
                except Exception:
                    pass
            lock_fh.close()
        except Exception:
            pass

    return final_client, final_collection
