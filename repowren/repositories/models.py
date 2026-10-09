"""Small data objects shared by repository services and persistence."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RepositoryRecord:
    """One local repository registered with RepoWren."""

    id: int
    name: str
    root_path: str
    is_active: bool


@dataclass(frozen=True, slots=True)
class FileMetadata:
    """Searchable metadata for one safe repository file."""

    path: str
    size_bytes: int
    modified_ns: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SearchMatch:
    """One matching source line."""

    path: str
    line: int
    text: str
