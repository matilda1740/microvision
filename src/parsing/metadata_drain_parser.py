"""Portable MetadataDrainParser extracted from the notebook.

This module provides the MetadataDrainParser class as a reusable component
that wraps Drain3's TemplateMiner for template mining and extracts
structured metadata. Importing this module will not fail if drain3 is
missing; the actual drain3 imports are delayed until the miner is
initialized, allowing the rest of the codebase to import this file safely.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import pandas as pd

from typing import List

# bring extractor and defaults from a small helper module so this file stays focused
from src.parsing.metadata_utils import FIELD_CONFIG, extract_and_build_metadata, is_numeric_like
from src.parsing.regex_utils import normalize_service_from_component




class MetadataDrainParser:
    """Custom parser that uses Drain3 TemplateMiner for template extraction.

    Parameters:
        log_format: optional LogPai-style format string (e.g. '<Date> <Time> <Component> <Content>').
        structured_csv: path to append structured parsed rows.
        templates_csv: path to write template catalogue.
        save_every: flush buffer size.
        field_config: mapping for which fields to extract (optional).
        mode: 'fresh' or 'incremental'. For 'fresh' the Drain3 persistence is not loaded.
        persistence_path: optional path for Drain3 persistence when using incremental mode.
    """

    def __init__(
        self,
        log_format: Optional[str],
        structured_csv: str,
        templates_csv: str,
        save_every: int = 1000,
        field_config: Optional[dict] = None,
        mode: str = "fresh",
        persistence_path: Optional[str] = None,
        mapping_key: Optional[str] = None,
    ) -> None:
        """
        log_format may be:
          - a Drain-style format string (contains '<'), e.g. '<Date> <Time> <Pid> <Level> <Component> <Content>'
          - a mapping key like 'OpenStack' which will be resolved via config.settings.LOG_FORMAT_MAPPINGS
          - a mapping dict {name: format_string} passed directly. If a dict is passed:
             * if mapping_key is provided and exists in the dict, that value is used
             * elif the dict has a 'default' key, that value is used
             * elif the dict has exactly one entry, that single value is used
             * otherwise the mapping is stored on the instance for possible later use and no format is selected (auto-detect)
        """
        resolved_format = log_format
        self._log_format_mappings = None

        # If caller passed a mapping dict directly
        if isinstance(log_format, dict):
            mappings = log_format
            self._log_format_mappings = mappings
            if mapping_key and mapping_key in mappings:
                resolved_format = mappings.get(mapping_key)
            elif "default" in mappings:
                resolved_format = mappings.get("default")
            elif len(mappings) == 1:
                resolved_format = next(iter(mappings.values()))
            else:
                # ambiguous mapping dict: store it and defer selection to auto-detect
                resolved_format = None

        # If caller provided a string key, try resolving via config settings
        elif isinstance(log_format, str) and "<" not in log_format:
            try:
                from config import settings as _settings

                mapping = getattr(_settings, "LOG_FORMAT_MAPPINGS", {}) or {}
                if log_format in mapping:
                    resolved_format = mapping.get(log_format)
            except Exception:
                # if settings can't be imported or mapping missing, keep the original
                resolved_format = log_format

        self.log_format = resolved_format
        self.structured_csv = structured_csv
        self.templates_csv = templates_csv
        self.save_every = int(save_every) if save_every else 1000
        # default field configuration if none provided
        # field_config can be injected from config.settings to allow runtime tuning.
        if field_config is not None:
            self.field_config = field_config
        else:
            try:
                from config import settings as _settings

                cfg = getattr(_settings, "FIELD_CONFIG", None)
                self.field_config = cfg if cfg is not None else FIELD_CONFIG
            except Exception:
                self.field_config = FIELD_CONFIG
        self.mode = mode.lower()
        self.persistence_path = persistence_path

        # internal state
        self.buffer = []
        self.total = 0
        self.unique_templates = set()
        # CSV column ordering persisted across flushes to avoid misalignment
        self._csv_columns = None

        # Drain3 attributes (initialized lazily)
        self.template_miner = None
        self._drain_config = None

        # Remove existing output files for a fresh run where appropriate
        try:
            if self.mode == "fresh":
                for path in (self.structured_csv, self.templates_csv):
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception:
                        pass
        except Exception:
            pass

    def _init_template_miner(self):
        """Lazy initialization of Drain3 TemplateMiner.

        This delays imports so modules that don't have drain3 installed can still
        import this file. If drain3 is missing, a RuntimeError is raised when
        the miner is required.
        """
        if self.template_miner is not None:
            return

        try:
            from drain3.template_miner_config import TemplateMinerConfig, MaskingInstruction
            from drain3.file_persistence import FilePersistence
            from drain3 import TemplateMiner
        except Exception as e:
            raise RuntimeError("drain3 is required for MetadataDrainParser but not installed") from e

        cfg = TemplateMinerConfig()
        # sensible defaults copied from the notebook
        cfg.profiling_enabled = True
        cfg.drain_sim_th = getattr(cfg, "drain_sim_th", 0.4)
        cfg.drain_depth = getattr(cfg, "drain_depth", 5)
        cfg.mask_prefix = getattr(cfg, "mask_prefix", "<*>")
        cfg.extra_delimiters = getattr(cfg, "extra_delimiters", ["=", ",", " ", ":", "-", '"', "[", "]", "(", ")"])

        # optional masking instructions are not required here; users can set via field_config
        # allow persistence only in incremental mode and when a persistence_path is supplied
        persist = None
        if self.mode == "incremental" and self.persistence_path:
            persist_dir = os.path.dirname(self.persistence_path)
            os.makedirs(persist_dir, exist_ok=True)
            persist = FilePersistence(self.persistence_path)

        self._drain_config = cfg
        self.template_miner = TemplateMiner(persist, cfg)

    def detect_log_format(self, sample_line: str, base_format: Optional[str]) -> Optional[str]:
        if not base_format:
            return base_format
        rotated_pattern = r'^[\w\-.]+\.log(?:\.\d+)?\.\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}'
        if re.match(rotated_pattern, sample_line):
            if not base_format.startswith("<File>"):
                return "<File> " + base_format
        return base_format

    def process_line(self, raw_line: str, line_no: int) -> None:
        """Parse one log line, enrich and buffer for flush."""
        raw = raw_line.rstrip("\n")

        if line_no == 1 and self.log_format:
            self.log_format = self.detect_log_format(raw, self.log_format)

        # lightweight parsing: if no log_format provided fall back to raw as content
        parsed_meta = {"Content": raw}
        if self.log_format:
            # use the same helper logic as in the notebook for regex parsing
            from src.parsing.log_parser import parse_line_with_format

            parsed_meta = parse_line_with_format(raw, self.log_format)

        content = parsed_meta.get("Content") or raw

        # normalize parsed_meta: remove diagnostic keys and canonicalize keys to lowercase
        for _k in ("Content", "ParseError", "Raw"):
            parsed_meta.pop(_k, None)

        # canonicalize keys to lowercase so we don't produce duplicated
        # capitalized vs lowercase columns later in the CSV
        parsed_meta = { (k.lower() if isinstance(k, str) else k): v for k, v in parsed_meta.items() }

        # Build/augment metadata from content and parsed_meta using field_config
        try:
            base_series = pd.Series(parsed_meta)
        except Exception:
            base_series = None
        try:
            meta_fields = list(self.field_config.get("metadata_fields", [])) if self.field_config else []
        except Exception:
            meta_fields = []

        try:
            built_meta = extract_and_build_metadata(
                content,
                component=parsed_meta.get("component"),
                base_row=base_series,
                metadata_fields=meta_fields,
            )
            # Only inject lowercase keys returned by the extractor. Downstream
            # components should read lowercase metadata fields (e.g. 'reqid', 'ip').
            for k, v in built_meta.items():
                if v is not None:
                    parsed_meta[k] = v
        except Exception:
            # non-fatal metadata build failure; proceed without augmentation
            pass

        # Defensive sanitization: avoid numeric tokens leaking into textual
        # metadata fields (observed: responsetime values sometimes landed in
        # the `service` field due to upstream parsing misalignment). We
        # prioritize extractor-provided values and fall back to deriving from
        # component. Also coerce common numeric fields to proper numeric types.

        # If `service` looks numeric, move it into `responsetime` (or
        # `responselength`) if those fields are missing or non-numeric.
        svc = parsed_meta.get("service")
        if svc is not None and is_numeric_like(svc):
            # prefer responsetime, then responselength
            if not parsed_meta.get("responsetime"):
                parsed_meta["responsetime"] = svc
            elif not parsed_meta.get("responselength"):
                parsed_meta["responselength"] = svc
            # remove the contaminated service value so we can recompute it
            parsed_meta.pop("service", None)

        # Ensure service exists and is a textual short-name derived from component
        if not parsed_meta.get("service"):
            comp = parsed_meta.get("component")
            service_name = normalize_service_from_component(comp)
            if service_name:
                parsed_meta["service"] = service_name

        # Coerce response-time/length to numeric types when possible
        if "responsetime" in parsed_meta and is_numeric_like(parsed_meta["responsetime"]):
            try:
                parsed_meta["responsetime"] = float(parsed_meta["responsetime"])
            except Exception:
                pass
        if "responselength" in parsed_meta and is_numeric_like(parsed_meta["responselength"]):
            try:
                parsed_meta["responselength"] = int(float(parsed_meta["responselength"]))
            except Exception:
                pass

        # initialize drain3 miner lazily (only when we need template mining)
        try:
            self._init_template_miner()
        except RuntimeError:
            # drain3 not available; produce minimal row without template mining
            row = {
                "line_no": line_no,
                "raw": raw,
                "content": content,
                "template_id": None,
                "template": None,
            }
            # normalize: prefer lowercase 'content' and avoid duplicate 'Content' column
            # also remove diagnostic keys returned by the low-level parser so
            # they don't become extra CSV columns (e.g. ParseError, Raw)
            for _k in ("Content", "ParseError", "Raw"):
                parsed_meta.pop(_k, None)
            row.update(parsed_meta)
            self.buffer.append(row)
            self.total += 1
            if len(self.buffer) >= self.save_every:
                self.flush_to_csv()
            return

        # perform template mining
        try:
            res = self.template_miner.add_log_message(content)
        except Exception:
            res = {"cluster_id": None, "template_mined": None}

        template = res.get("template_mined")
        template_id = res.get("cluster_id")

        self.unique_templates.add(template or f"__none_{template_id}")

        row = {
            "line_no": line_no,
            "raw": raw,
            "content": content,
            "template_id": template_id,
            "template": template,
        }
        # ensure diagnostic keys are not present (safe to pop again)
        for _k in ("Content", "ParseError", "Raw"):
            parsed_meta.pop(_k, None)
        row.update(parsed_meta)

        self.buffer.append(row)
        self.total += 1

        if len(self.buffer) >= self.save_every:
            self.flush_to_csv()

    def flush_to_csv(self) -> None:
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer)
        # Determine whether we need to write a header (first write)
        file_exists = os.path.exists(self.structured_csv)
        header = not file_exists

        # If this is the first write, record the canonical column order.
        if header:
            self._csv_columns = list(df.columns)
            df.to_csv(self.structured_csv, mode="a", index=False, header=True)
        else:
            # Ensure we have the canonical column ordering. If not already set,
            # attempt to read the existing file's header to determine it.
            if self._csv_columns is None and file_exists:
                try:
                    import csv as _csv

                    with open(self.structured_csv, "r", encoding="utf-8", errors="ignore") as fh:
                        reader = _csv.reader(fh)
                        existing_header = next(reader, None)
                        if existing_header:
                            self._csv_columns = existing_header
                except Exception:
                    # fallback: use current df columns
                    self._csv_columns = list(df.columns)

            # Reindex df to match canonical columns to avoid mis-ordered writes
            if self._csv_columns is not None:
                df = df.reindex(columns=self._csv_columns)

            df.to_csv(self.structured_csv, mode="a", index=False, header=False)
        self.buffer = []

    def export_templates(self) -> None:
        if self.template_miner is None:
            return
        clusters = getattr(self.template_miner.drain, "clusters", None)
        if not clusters:
            return
        if isinstance(clusters, dict):
            cluster_list = clusters.values()
        else:
            cluster_list = clusters
        templates = []
        for cluster in cluster_list:
            cid = getattr(cluster, "cluster_id", None)
            templates.append({
                "template_id": cid,
                "template": cluster.get_template(),
                "occurrences": getattr(cluster, "size", None),
            })
        df = pd.DataFrame(templates)
        df.to_csv(self.templates_csv, index=False)

    def finalize(self) -> None:
        if self.buffer:
            self.flush_to_csv()
        # export templates if miner was created
        try:
            self.export_templates()
        except Exception:
            pass
