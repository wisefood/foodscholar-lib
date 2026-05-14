from datetime import UTC, datetime

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ArtifactMeta(BaseModel):
    artifact_id: str
    phase: str
    config_hash: str
    upstream_artifact_ids: list[str] = []
    record_count: int
    schema_version: str
    created_at: datetime = Field(default_factory=_utcnow)
