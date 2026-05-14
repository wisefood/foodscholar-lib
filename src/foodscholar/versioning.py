"""Config hashing and ArtifactMeta helpers.

A config hash is the SHA-256 of the canonical JSON serialization of a
configuration object. Stable across runs as long as the config content is
identical, regardless of dict ordering.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from pydantic import BaseModel

from foodscholar.io.artifacts import ArtifactMeta

SCHEMA_VERSION = "v0.1"


def _canonical(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return _canonical(obj.model_dump(mode="json"))
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def config_hash(config: Any) -> str:
    canonical = _canonical(config)
    encoded = json.dumps(canonical, separators=(",", ":"), sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def new_artifact_id(phase: str) -> str:
    return f"{phase}-{uuid.uuid4().hex[:12]}"


def make_artifact_meta(
    *,
    phase: str,
    config: Any,
    record_count: int,
    upstream_artifact_ids: list[str] | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> ArtifactMeta:
    return ArtifactMeta(
        artifact_id=new_artifact_id(phase),
        phase=phase,
        config_hash=config_hash(config),
        upstream_artifact_ids=upstream_artifact_ids or [],
        record_count=record_count,
        schema_version=schema_version,
    )
