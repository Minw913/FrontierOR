"""Bounded, no-follow file operations for untrusted benchmark artifacts."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import tempfile
from pathlib import Path


class SecureFileError(ValueError):
    """Raised when an untrusted path is not a bounded regular file."""


def _open_regular(
    path: os.PathLike[str] | str,
    *,
    max_bytes: int,
    label: str,
    require_single_link: bool = True,
) -> tuple[int, os.stat_result]:
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(os.fspath(path), flags)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise SecureFileError(f"{label} must not be a symlink") from exc
        raise SecureFileError(f"cannot open {label}: {exc.strerror or exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise SecureFileError(f"{label} must be a regular file")
        if require_single_link and info.st_nlink != 1:
            raise SecureFileError(f"{label} must have exactly one hard link")
        if info.st_size > max_bytes:
            raise SecureFileError(
                f"{label} exceeds the {max_bytes}-byte size limit"
            )
        return fd, info
    except Exception:
        os.close(fd)
        raise


def read_regular_file(
    path: os.PathLike[str] | str,
    *,
    max_bytes: int,
    label: str,
    require_single_link: bool = True,
) -> bytes:
    """Read a regular file through an already-validated, no-follow descriptor."""
    fd, _ = _open_regular(
        path,
        max_bytes=max_bytes,
        label=label,
        require_single_link=require_single_link,
    )
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = os.read(fd, min(1024 * 1024, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise SecureFileError(
                    f"{label} exceeds the {max_bytes}-byte size limit"
                )
            chunks.append(chunk)
    finally:
        os.close(fd)
    return b"".join(chunks)


def sha256_regular_file(
    path: os.PathLike[str] | str,
    *,
    max_bytes: int,
    label: str,
    require_single_link: bool = True,
) -> str:
    """Hash a bounded regular file through a no-follow descriptor."""
    fd, _ = _open_regular(
        path,
        max_bytes=max_bytes,
        label=label,
        require_single_link=require_single_link,
    )
    digest = hashlib.sha256()
    total = 0
    try:
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise SecureFileError(
                    f"{label} exceeds the {max_bytes}-byte size limit"
                )
            digest.update(chunk)
    finally:
        os.close(fd)
    return digest.hexdigest()


def copy_regular_file(
    source: os.PathLike[str] | str,
    destination: os.PathLike[str] | str,
    *,
    max_bytes: int,
    label: str,
    require_single_link: bool = True,
    mode: int = 0o600,
) -> int:
    """Copy an untrusted regular file to a trusted path without following links."""
    source_fd, _ = _open_regular(
        source,
        max_bytes=max_bytes,
        label=label,
        require_single_link=require_single_link,
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    total = 0
    try:
        os.fchmod(temp_fd, mode)
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise SecureFileError(
                    f"{label} exceeds the {max_bytes}-byte size limit"
                )
            view = memoryview(chunk)
            while view:
                written = os.write(temp_fd, view)
                view = view[written:]
        os.fsync(temp_fd)
        os.close(temp_fd)
        temp_fd = -1
        os.replace(temp_name, destination)
    finally:
        os.close(source_fd)
        if temp_fd >= 0:
            os.close(temp_fd)
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    return total
