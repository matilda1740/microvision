import json
import tarfile
import subprocess
import sys
from pathlib import Path


def write_manifest(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh)


def touch(path: Path, content: bytes = b"ok"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_archive_run_and_preserve_log(tmp_path: Path):
    # prepare dirs
    manifests_dir = tmp_path / "manifests"
    archives_dir = tmp_path / "archives"
    artifacts_dir = tmp_path / "artifacts"

    # create sample artifacts: a .db and a .log
    edges_db = artifacts_dir / "edges_test.db"
    log_file = artifacts_dir / "run_full.log"
    touch(edges_db, b"dbdata")
    touch(log_file, b"logdata")

    # create manifest referencing these
    manifest = manifests_dir / "run_test.json"
    write_manifest(manifest, {"run_id": "test", "edges_db": str(edges_db), "source": str(log_file)})

    # run cleanup (archive + delete); default preserve pattern includes '*.log'
    cmd = [sys.executable, "scripts/cleanup_runs.py", "--manifests-dir", str(manifests_dir), "--archives-dir", str(archives_dir), "--older-than-days", "0", "--archive", "--delete"]
    subprocess.run(cmd, check=True)

    # assert archive created
    archives = list(archives_dir.glob("*.tar.gz"))
    assert len(archives) == 1
    arch = archives[0]
    with tarfile.open(arch, "r:gz") as t:
        members = {m.name for m in t.getmembers()}
    # archive should contain the manifest and the artifacts (by name)
    assert "run_test.json" in members
    assert "edges_test.db" in members
    assert "run_full.log" in members

    # manifest should be removed
    assert not manifest.exists()
    # edges_db should be removed (deleted), but log file preserved per default preserve pattern
    assert not edges_db.exists()
    assert log_file.exists()


def test_prune_tmp_dirs(tmp_path: Path):
    manifests_dir = tmp_path / "manifests"
    archives_dir = tmp_path / "archives"
    artifacts_dir = tmp_path / "artifacts"

    tmp_dir = artifacts_dir / "session_tmp"
    tmp_dir_tmp = artifacts_dir / "session_tmp_tmp"  # endswith _tmp
    tmp_dir_tmp.mkdir(parents=True)
    touch(tmp_dir_tmp / "tempfile", b"x")

    manifest = manifests_dir / "run_tmp.json"
    write_manifest(manifest, {"run_id": "tmp", "persist_dir": str(tmp_dir_tmp)})

    # run cleanup with prune-tmp
    cmd = [sys.executable, "scripts/cleanup_runs.py", "--manifests-dir", str(manifests_dir), "--archives-dir", str(archives_dir), "--older-than-days", "0", "--archive", "--delete", "--prune-tmp"]
    subprocess.run(cmd, check=True)

    # tmp dir should be pruned
    assert not tmp_dir_tmp.exists()
