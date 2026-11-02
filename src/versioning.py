"""Lightweight, DVC-compatible-in-spirit data versioning.

How DVC works and what this module mirrors:
  * The dataset bytes are hashed (MD5) and stored once in a content-addressed
    cache:  data/.versions/<md5[:2]>/<md5[2:]>   (DVC: .dvc/cache/files/md5/..)
  * Git only tracks a tiny metadata pointer per version:
    data/versions/<tag>.dvc  (YAML with md5, size, rows, schema, parent)
  * "Checking out" a version = resolving its md5 and reading from the cache.
Because the hash is of the content, identical data -> identical id, and any
single changed cell -> a new id. That is what makes experiments reproducible.

The repository also contains a real `dvc.yaml` pipeline (see README) so the
same flow can be run with the actual DVC CLI (`dvc repro`, `dvc push`).
"""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone

import pandas as pd
import yaml

from .config import VERSION_META, VERSION_STORE


def _to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, date_format="%Y-%m-%dT%H:%M:%S")
    return buf.getvalue().encode()


def md5_of(df: pd.DataFrame) -> str:
    return hashlib.md5(_to_bytes(df)).hexdigest()


def _cache_path(md5: str):
    return VERSION_STORE / md5[:2] / md5[2:]


def commit_version(df: pd.DataFrame, tag: str, message: str, parent: str | None = None) -> dict:
    """Store dataset content once (by hash) and write a .dvc-style pointer file."""
    data = _to_bytes(df)
    md5 = hashlib.md5(data).hexdigest()
    path = _cache_path(md5)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    meta = {
        "tag": tag,
        "outs": [{"md5": md5, "size": len(data), "path": "flights.csv", "hash": "md5"}],
        "message": message,
        "parent": parent,
        "rows": int(len(df)),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "date_range": [str(df["scheduled_departure"].min()), str(df["scheduled_departure"].max())],
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    VERSION_META.mkdir(parents=True, exist_ok=True)
    (VERSION_META / f"{tag}.dvc").write_text(yaml.safe_dump(meta, sort_keys=False))
    return meta


def list_versions() -> list[dict]:
    if not VERSION_META.exists():
        return []
    metas = [yaml.safe_load(p.read_text()) for p in VERSION_META.glob("*.dvc")]
    return sorted(metas, key=lambda m: m["tag"])


def get_meta(tag: str) -> dict:
    return yaml.safe_load((VERSION_META / f"{tag}.dvc").read_text())


def checkout(tag: str) -> pd.DataFrame:
    """Load the exact bytes recorded for `tag`, verifying integrity."""
    meta = get_meta(tag)
    md5 = meta["outs"][0]["md5"]
    raw = _cache_path(md5).read_bytes()
    if hashlib.md5(raw).hexdigest() != md5:
        raise IOError(f"Cache corruption detected for {tag}")
    df = pd.read_csv(io.BytesIO(raw), parse_dates=["scheduled_departure"])
    return df


def diff(tag_a: str, tag_b: str) -> dict:
    """Schema + content diff between two versions (like `dvc diff` + data diff)."""
    a, b = checkout(tag_a), checkout(tag_b)
    ma, mb = get_meta(tag_a), get_meta(tag_b)
    ca, cb = set(a.columns), set(b.columns)
    ids_a, ids_b = set(a["flight_id"]), set(b["flight_id"])
    return {
        "md5_a": ma["outs"][0]["md5"], "md5_b": mb["outs"][0]["md5"],
        "rows_a": len(a), "rows_b": len(b),
        "added_columns": sorted(cb - ca), "removed_columns": sorted(ca - cb),
        "type_changes": {c: (ma["columns"][c], mb["columns"][c]) for c in ca & cb
                         if ma["columns"][c] != mb["columns"][c]},
        "rows_added": len(ids_b - ids_a), "rows_removed": len(ids_a - ids_b),
        "dup_a": int(a["flight_id"].duplicated().sum()), "dup_b": int(b["flight_id"].duplicated().sum()),
        "null_a": int(a.isna().sum().sum()), "null_b": int(b.isna().sum().sum()),
        "delay_rate_a": float(a["is_delayed"].mean()), "delay_rate_b": float(b["is_delayed"].mean()),
        "carriers_added": sorted(set(b["carrier"]) - set(a["carrier"])),
    }


def lineage_json() -> str:
    return json.dumps([{k: m[k] for k in ("tag", "parent", "rows")} | {"md5": m["outs"][0]["md5"]}
                       for m in list_versions()], indent=2)
