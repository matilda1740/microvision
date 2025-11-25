"""Utility to archive or delete old run artifacts (manifests, DBs, chroma dirs).

Usage examples:
    python scripts/cleanup_runs.py --older-than-days 30 --archive --dry-run
    python scripts/cleanup_runs.py --keep-last 5 --delete

The tool looks at `data/run_manifests/*.json` and uses fields in the
manifest to locate related artifacts (edges DB, chroma dir). Archiving
packages manifest + discovered artifacts into a tar.gz archive under
`data/archives/`.
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List


def _list_manifests(manifest_dir: Path) -> List[Path]:
    if not manifest_dir.exists():
        return []
    return sorted(manifest_dir.glob("*.json"))


def _manifest_finished_at(manifest_path: Path) -> datetime:
    try:
        j = json.load(open(manifest_path))
        t = j.get("finished_at") or j.get("started_at")
        if t is None:
            # fallback to file mtime
            return datetime.fromtimestamp(manifest_path.stat().st_mtime, tz=timezone.utc)
        # expect ISO-8601 with Z
        return datetime.fromisoformat(t.replace("Z", "+00:00"))
    except Exception:
        return datetime.fromtimestamp(manifest_path.stat().st_mtime, tz=timezone.utc)


def archive_run(manifest_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path) as fh:
        manifest = json.load(fh)

    run_id = manifest.get("run_id") or manifest_path.stem
    archive_name = out_dir / f"run_{run_id}.tar.gz"

    # collect candidate artifact paths mentioned in the manifest
    candidates = []
    for key in ("edges_db", "db", "chroma_dir", "persist_dir", "source", "merged_path"):
        val = manifest.get(key)
        if not val:
            continue
        p = Path(val)
        if p.exists():
            candidates.append(p)

    # always include the manifest itself
    candidates.append(manifest_path)

    # create tar.gz containing those files/dirs
    with tarfile.open(archive_name, "w:gz") as tar:
        for p in candidates:
            try:
                tar.add(str(p), arcname=str(p.name))
            except Exception:
                # ignore missing / unreadable
                continue

    return archive_name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifests-dir", default="data/run_manifests")
    p.add_argument("--archives-dir", default="data/archives")
    p.add_argument("--older-than-days", type=int, default=None)
    p.add_argument("--keep-last", type=int, default=None)
    p.add_argument("--archive", action="store_true", help="Archive selected runs")
    p.add_argument("--delete", action="store_true", help="Delete selected run artifacts after archiving (implies --archive)")
    p.add_argument("--prune-tmp", action="store_true", help="Prune incomplete tmp directories (e.g., chroma _tmp dirs)")
    p.add_argument("--delete-archives", action="store_true", help="Delete archive files older than the cutoff")
    p.add_argument("--exclude-source-substr", action="append", default=[], help="Exclude manifests whose 'source' contains this substring (repeatable)")
    p.add_argument("--preserve-pattern", action="append", default=["*.log"], help="Glob pattern(s) for artifacts to preserve (repeatable); defaults to '*.log'")
    p.add_argument("--dry-run", action="store_true", help="Show actions without performing them")
    args = p.parse_args()

    manifests_dir = Path(args.manifests_dir)
    archives_dir = Path(args.archives_dir)

    manifests = _list_manifests(manifests_dir)
    if not manifests:
        print("No run manifests found in", manifests_dir)
        return 0

    # determine cutoff based on older-than-days
    now = datetime.now(timezone.utc)
    cutoff = None
    if args.older_than_days is not None:
        cutoff = now - timedelta(days=int(args.older_than_days))

    # option: keep-last N (preserve the most recent N manifests)
    keep_last = int(args.keep_last) if args.keep_last is not None else None

    # build list of candidates to act upon
    manifest_dates = [(p, _manifest_finished_at(p)) for p in manifests]
    # sort by finished_at ascending (oldest first)
    manifest_dates.sort(key=lambda x: x[1])

    to_act: List[Path] = []
    for idx, (p, dt) in enumerate(manifest_dates):
        # skip if in keep-last window
        if keep_last is not None:
            if len(manifest_dates) - idx <= keep_last:
                continue
        if cutoff is not None and dt > cutoff:
            continue
        to_act.append(p)

    if not to_act:
        print("No runs matched selection criteria")
        return 0

    for m in to_act:
        print("Selected:", m)

    if args.dry_run:
        print("Dry-run; no changes made")
        return 0

    if args.archive or args.delete or args.prune_tmp or args.delete_archives:
        archives_dir = Path(args.archives_dir)
        archives_dir.mkdir(parents=True, exist_ok=True)

        # helper: determine whether to skip a manifest based on source substrings
        def _should_exclude(manifest_path: Path) -> bool:
            if not args.exclude_source_substr:
                return False
            try:
                j = json.load(open(manifest_path))
                src = j.get("source", "") or ""
                for sub in args.exclude_source_substr:
                    if sub and sub in src:
                        return True
            except Exception:
                # if manifest unreadable, don't exclude
                return False
            return False

        for m in to_act:
            if _should_exclude(m):
                print("Skipping excluded manifest:", m)
                continue

            try:
                arch = archive_run(m, archives_dir)
                print("Archived:", m, "->", arch)

                # optionally prune tmp dirs related to this run (best-effort)
                if args.prune_tmp:
                    try:
                        with open(m) as fh:
                            j = json.load(fh)
                        # common tmp/ephemeral locations to consider
                        candidates = []
                        # chroma tmp dir (persist_directory/tmp name)
                        for key in ("persist_directory", "persist_dir", "chroma_dir", "tmp_dir"):
                            val = j.get(key)
                            if val:
                                p = Path(val)
                                # if path endswith _tmp or contains _tmp, remove it
                                if p.exists() and (p.name.endswith("_tmp") or "_tmp" in p.name):
                                    candidates.append(p)
                        # generic search under data for dirs ending with _tmp
                        datadir = Path("data")
                        if datadir.exists():
                            for p in datadir.rglob("*_tmp"):
                                if p.is_dir():
                                    candidates.append(p)

                        for p in candidates:
                            try:
                                import shutil

                                if p.exists():
                                    shutil.rmtree(p)
                                    print("Pruned tmp dir:", p)
                            except Exception as e:
                                print("Could not prune tmp dir:", p, e)
                    except Exception:
                        pass

                if args.delete:
                    # best-effort deletion of manifest and referenced artifacts
                    try:
                        with open(m) as fh:
                            j = json.load(fh)
                        for key in ("edges_db", "db", "chroma_dir", "persist_dir", "source", "merged_path"):
                            val = j.get(key)
                            if not val:
                                continue
                            p = Path(val)
                            # skip deletion for preserved patterns (raw logs, etc.)
                            skip = False
                            try:
                                for pat in args.preserve_pattern:
                                    if p.match(pat) or p.name == pat or p.name.endswith(pat.lstrip('*')):
                                        skip = True
                                        break
                            except Exception:
                                pass
                            if skip:
                                print("Preserved (per pattern):", p)
                                continue
                            if p.exists():
                                if p.is_dir():
                                    import shutil

                                    shutil.rmtree(p)
                                    print("Removed dir:", p)
                                else:
                                    p.unlink()
                                    print("Removed file:", p)
                    except Exception as e:
                        print("Could not delete artifacts for:", m, "error:", e)
                    # finally remove manifest
                    try:
                        m.unlink()
                        print("Removed manifest:", m)
                    except Exception as e:
                        print("Could not remove manifest:", m, "error:", e)
            except Exception as e:
                print("Failed to archive ", m, e)

        # delete archive files older than cutoff when requested
        if args.delete_archives and cutoff is not None:
            for arch in Path(args.archives_dir).glob("*.tar.gz"):
                try:
                    st = datetime.fromtimestamp(arch.stat().st_mtime, tz=timezone.utc)
                    if st <= cutoff:
                        try:
                            arch.unlink()
                            print("Removed old archive:", arch)
                        except Exception as e:
                            print("Could not remove archive:", arch, e)
                except Exception:
                    continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
