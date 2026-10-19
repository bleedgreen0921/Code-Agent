"""Content-addressed candidate checkpoints. No Git or candidate code runs on the host."""

import difflib
import hashlib
import io
import json
import os
import stat
import zipfile
from pathlib import Path

from agent_service.code_artifacts import CodeFailure
from execution.code_snapshot import digest, put
from execution.code_snapshot import read_regular as snapshot_read_regular


def read_regular(root, name, maximum):
    return snapshot_read_regular(root, name, maximum, preserve_mode=True)


# Frozen server policy, independent of candidate .gitignore/configuration.
EXCLUDED_DIRS = frozenset(
    (
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".cache",
        ".venv",
        "node_modules",
    )
)
EXCLUDED_SUFFIXES = (".pyc", ".pyo")


def source_manifest(manifest):
    return {k: v for k, v in manifest.items() if not k.startswith(".git/")}


def scan_candidate(root, baseline, limits):
    result, total = {}, 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        relative_dir = Path(directory).relative_to(root)
        for name in list(dirs):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValueError("Symlinks cannot be sealed: " + relative)
            if name in EXCLUDED_DIRS and not any(
                p.startswith(relative + "/") for p in baseline
            ):
                dirs.remove(name)
        for name in files:
            relative = (relative_dir / name).as_posix()
            if relative == ".git":
                raise ValueError("Git mount point must remain a directory")
            if not stat.S_ISREG(os.lstat(Path(directory) / name).st_mode):
                raise ValueError("Unsupported candidate file type: " + relative)
            if relative not in baseline and name.endswith(EXCLUDED_SUFFIXES):
                continue
            try:
                data, mode = read_regular(root, relative, limits["file_bytes"])
            except (ValueError, OSError) as exc:
                raise ValueError(
                    "Unsupported or unstable candidate file: " + relative
                ) from exc
            total += len(data)
            if total > limits["snapshot_bytes"] or len(result) >= limits["file_count"]:
                raise CodeFailure(
                    "ARTIFACT_LIMIT", "Candidate version exceeds managed file limits"
                )
            result[relative] = dict(
                sha256=hashlib.sha256(data).hexdigest(), size=len(data), mode=mode
            )
    return result


def copy_manifest(source, destination, manifest, maximum):
    owner = source.stat()
    destination.mkdir(parents=True, exist_ok=False)
    for name, info in manifest.items():
        data, mode = read_regular(source, name, maximum)
        if hashlib.sha256(data).hexdigest() != info["sha256"] or mode != info["mode"]:
            raise ValueError("Candidate changed during content copy: " + name)
        put(destination, name, data, mode)
    if os.getuid() == 0:
        for directory, dirs, files in os.walk(destination):
            os.chown(directory, owner.st_uid, owner.st_gid)
            for name in files:
                os.chown(Path(directory) / name, owner.st_uid, owner.st_gid)


def baseline_archive(root, manifest, maximum):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, info in sorted(manifest.items()):
            data, mode = read_regular(root, name, maximum)
            if (
                hashlib.sha256(data).hexdigest() != info["sha256"]
                or mode != info["mode"]
            ):
                raise CodeFailure("SNAPSHOT_INVALID", "Baseline export content changed")
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(entry, data)
    return output.getvalue()


def freeze(source, destination, baseline_root, baseline, limits):
    before = scan_candidate(source, baseline, limits)
    copy_manifest(source, destination, before, limits["file_bytes"])
    if (
        scan_candidate(source, baseline, limits) != before
        or scan_candidate(destination, baseline, limits) != before
    ):
        raise ValueError("Candidate did not remain consistent during freezing")
    # Persist immutable content before its database registration is acknowledged.
    for directory, dirs, files in os.walk(destination, topdown=False):
        for name in files:
            fd = os.open(Path(directory) / name, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        fd = os.open(directory, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    for parent in (destination.parent, destination.parent.parent):
        fd = os.open(parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return build_exports(destination, baseline_root, baseline, before, limits)


def build_exports(destination, baseline_root, baseline, before, limits):
    changes, patches = [], []
    package = io.BytesIO()
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(set(baseline) | set(before)):
            old, new = baseline.get(name), before.get(name)
            if old == new:
                continue
            old_data = (
                read_regular(baseline_root, name, limits["file_bytes"])[0]
                if old
                else b""
            )
            new_data = (
                read_regular(destination, name, limits["file_bytes"])[0] if new else b""
            )
            binary = b"\x00" in old_data or b"\x00" in new_data
            try:
                old_text, new_text = old_data.decode("utf-8"), new_data.decode("utf-8")
            except UnicodeError:
                binary = True
            complete_text = (
                not any(c in name for c in "\t\r\n")
                and not (old_data == new_data == b"" and (old is None or new is None))
                and not binary
                and (not old_data or old_data.endswith(b"\n"))
                and (not new_data or new_data.endswith(b"\n"))
            )
            changes.append(
                dict(
                    path=name,
                    before=old,
                    after=new,
                    text_diff_complete=complete_text
                    and (not old or not new or old["mode"] == new["mode"]),
                )
            )
            if binary:
                patches.append("Binary files differ: " + json.dumps(name) + "\n")
            else:
                patches.extend(
                    difflib.unified_diff(
                        old_text.splitlines(keepends=True),
                        new_text.splitlines(keepends=True),
                        fromfile="a/" + name if old else "/dev/null",
                        tofile="b/" + name if new else "/dev/null",
                    )
                )
            if old and new and old["mode"] != new["mode"]:
                patches.append(
                    f"mode {old['mode']:o} -> {new['mode']:o}: {json.dumps(name)}\n"
                )
            if new:
                entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                entry.external_attr = (stat.S_IFREG | new["mode"]) << 16
                archive.writestr(entry, new_data)
    document = dict(
        policy="repair-candidate-v1",
        content_digest=digest(before),
        manifest=before,
        baseline_digest=digest(baseline),
        changes=changes,
    )
    return document, "".join(patches).encode(), package.getvalue()


def verify_revision(root, document, baseline, limits):
    found = scan_candidate(root, baseline, limits)
    if (
        found != document["manifest"]
        or digest(found) != document["content_digest"]
        or document["baseline_digest"] != digest(baseline)
    ):
        raise CodeFailure(
            "SNAPSHOT_INVALID", "Frozen candidate content identity differs"
        )


def verify_package(baseline_root, baseline, document, package, maximum):
    """Verify reconstruction without extracting paths from an archive."""
    expected = dict(baseline)
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        entries = archive.infolist()
        additions = {c["path"]: c for c in document["changes"] if c["after"]}
        if len(entries) != len(additions) or set(archive.namelist()) != set(additions):
            raise CodeFailure(
                "ARTIFACT_INVALID", "Candidate package inventory mismatch"
            )
        for change in document["changes"]:
            name = change["path"]
            if baseline.get(name) != change["before"]:
                raise CodeFailure(
                    "ARTIFACT_INVALID", "Candidate baseline entry mismatch"
                )
            after = change["after"]
            if after:
                entry = archive.getinfo(name)
                if (
                    entry.file_size > maximum
                    or stat.S_IMODE(entry.external_attr >> 16) != after["mode"]
                ):
                    raise CodeFailure(
                        "ARTIFACT_INVALID", "Candidate package mode or size mismatch"
                    )
                data = archive.read(entry)
                if (
                    len(data) != after["size"]
                    or hashlib.sha256(data).hexdigest() != after["sha256"]
                ):
                    raise CodeFailure(
                        "ARTIFACT_INVALID", "Candidate package content mismatch"
                    )
                expected[name] = after
            else:
                expected.pop(name, None)
    if (
        expected != document["manifest"]
        or digest(expected) != document["content_digest"]
    ):
        raise CodeFailure(
            "ARTIFACT_INVALID", "Reconstructed candidate identity mismatch"
        )
