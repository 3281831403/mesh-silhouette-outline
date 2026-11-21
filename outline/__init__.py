"""Cartoon silhouette outline extraction in screen space."""

from .silhouette import (
    OutlineError,
    build_outline_mesh,
    depth_offset,
    expand_edges,
    extract_edges,
    face_normal,
    is_silhouette,
    outline_batch,
    validate_mesh,
)

__all__ = [
    "OutlineError",
    "build_outline_mesh",
    "depth_offset",
    "expand_edges",
    "extract_edges",
    "face_normal",
    "is_silhouette",
    "outline_batch",
    "validate_mesh",
]
