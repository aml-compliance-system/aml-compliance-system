"""Minimal access control and safe path resolution for the investigator API.

The project has no identity provider. This module provides a deliberately small
local-first model:

* If ``FINDETECT_API_TOKEN`` is set, every data endpoint requires
  ``Authorization: Bearer <token>``.
* The investigator identity is the configured ``FINDETECT_INVESTIGATOR_ID`` or a
  validated ``X-Investigator-Id`` header. This is an attribution aid, not
  authenticated multi-user RBAC.
* Pipeline inputs must resolve inside ``FINDETECT_DATA_ROOT``.

Full per-user case permissions require an identity system and are intentionally
out of scope for this release; the limitation is documented in the API and UI.
"""

from __future__ import annotations

import hmac
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import Header, HTTPException

import config

_ACTOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._@-]{0,63}$")


@dataclass(frozen=True)
class Actor:
    id: str
    authenticated: bool


def _validate_actor(raw: str) -> str:
    actor = (raw or "").strip()
    if not actor or not _ACTOR_RE.match(actor):
        raise HTTPException(
            status_code=400,
            detail="Invalid investigator identity. Use 1-64 letters, digits, spaces, . _ @ or -.",
        )
    return actor


def get_actor(
    authorization: Optional[str] = Header(default=None),
    x_investigator_id: Optional[str] = Header(default=None),
) -> Actor:
    """FastAPI dependency enforcing the optional shared-secret gate."""
    authenticated = False
    if config.API_TOKEN:
        expected = f"Bearer {config.API_TOKEN}"
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(
                status_code=401,
                detail="Authentication required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        authenticated = True

    actor = _validate_actor(x_investigator_id or config.INVESTIGATOR_ID)
    return Actor(id=actor, authenticated=authenticated)


def auth_mode() -> str:
    return "token" if config.API_TOKEN else "local"


def resolve_input_path(raw_path: str) -> Path:
    """Resolve a pipeline input path and enforce the configured data root."""
    candidate = Path(str(raw_path)).expanduser()
    if not candidate.is_absolute():
        candidate = config.REPO_ROOT / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Input file not found: {raw_path}")

    root = config.DATA_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise HTTPException(
            status_code=403,
            detail=(
                "Pipeline input must be inside the configured data directory "
                f"({root})."
            ),
        )
    if resolved.suffix.lower() != ".parquet":
        raise HTTPException(status_code=400, detail="Pipeline input must be a .parquet file.")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"Input file not found: {raw_path}")
    return resolved


def safe_filename(value: str, fallback: str = "report") -> str:
    """Return a conservative filename component for Content-Disposition."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "")).strip(".-")
    cleaned = cleaned[:120]
    return cleaned or fallback
