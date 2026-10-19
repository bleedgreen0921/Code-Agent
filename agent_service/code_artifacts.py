"""Managed atomic content storage. Public callers supply IDs, never paths."""

import hashlib
import os
import stat
from pathlib import Path
from uuid import uuid4

from agent_service.execution_errors import ExecutionFailure


class CodeFailure(ExecutionFailure):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


class ArtifactStorage:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, key):
        # Only server generated UUID keys are admitted, including database values.
        parts = key.split("/")
        if (
            len(parts) != 2
            or parts[0] not in ("artifacts", "snapshots", "candidates", "revisions")
            or len(parts[1]) != 32
        ):
            raise CodeFailure("ARTIFACT_INVALID", "Invalid managed storage key")
        try:
            int(parts[1], 16)
        except ValueError as exc:
            raise CodeFailure(
                "ARTIFACT_INVALID", "Invalid managed storage key"
            ) from exc
        path = self.root / key
        if (
            path.is_symlink()
            or path.parent.is_symlink()
            or not path.resolve().is_relative_to(self.root)
        ):
            raise CodeFailure("ARTIFACT_INVALID", "Managed storage identity changed")
        return path

    def write(self, artifact_id, data, maximum):
        if len(data) > maximum:
            raise CodeFailure("ARTIFACT_LIMIT", "Artifact exceeds admission limit")
        key = "artifacts/" + artifact_id.hex
        path = self.path(key)
        path.parent.mkdir(mode=0o700, exist_ok=True)
        checksum = hashlib.sha256(data).hexdigest()
        if path.exists():
            if self.read(key, checksum, len(data)) != data:
                raise CodeFailure("ARTIFACT_INVALID", "Artifact ID collision")
            return key, checksum
        temporary = path.parent / (".pending-" + uuid4().hex)
        try:
            fd = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # No overwrite publication; retries verify the exact previous contents.
            os.link(temporary, path, follow_symlinks=False)
            directory = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
        return key, checksum

    def read(self, key, checksum, size):
        try:
            fd = os.open(self.path(key), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size != size:
                    raise ValueError("size mismatch")
                data = stream.read(size + 1)
            if len(data) != size or hashlib.sha256(data).hexdigest() != checksum:
                raise ValueError("digest mismatch")
            return data
        except (ValueError, OSError) as exc:
            raise CodeFailure(
                "ARTIFACT_INVALID", "Artifact is missing or corrupt"
            ) from exc
