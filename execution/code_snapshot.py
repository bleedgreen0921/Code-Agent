"""Trusted snapshot helper. Never imports or executes project Python code."""

import hashlib
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

# -I excludes the script directory; this directory contains only image assets.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from git_tools import Git, GitError


class SnapshotError(ValueError):
    def __init__(self, message, code="SNAPSHOT_UNSUPPORTED"):
        self.code = code
        super().__init__(message)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def safe_parts(value):
    parts = value.split("/")
    if (
        not value
        or value.startswith("/")
        or any(p in ("", ".", "..") for p in parts)
        or "\x00" in value
    ):
        raise SnapshotError("Invalid source path")
    return parts


def read_regular(root, name, maximum, *, preserve_mode=False):
    parts = safe_parts(name)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = child
        handle = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
        )
        with os.fdopen(handle, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
                raise SnapshotError(
                    "Special file or size limit exceeded", "SNAPSHOT_LIMIT"
                )
            data = stream.read(maximum + 1)
            if len(data) > maximum:
                raise SnapshotError("File exceeds size limit", "SNAPSHOT_LIMIT")
        return data, stat.S_IMODE(info.st_mode) if preserve_mode else (
            0o755 if info.st_mode & 0o111 else 0o644
        )
    finally:
        os.close(fd)


def put(root, name, data, mode=0o644):
    safe_parts(name)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)


def optional(root, name, maximum):
    try:
        return read_regular(root, name, maximum)[0]
    except FileNotFoundError:
        return None


def tree_records(git, commit, maximum):
    records = git.run(["ls-tree", "-r", "-z", commit], maximum=maximum)
    result = []
    for record in records.split(b"\0"):
        if not record:
            continue
        meta, raw = record.split(b"\t", 1)
        mode, kind, blob = meta.decode("ascii").split()
        path = raw.decode("utf-8")
        safe_parts(path)
        if (
            path.split("/")[0].lower() == ".git"
            or mode not in ("100644", "100755")
            or kind != "blob"
        ):
            raise SnapshotError("Symlinks and submodules are unsupported")
        result.append((path, blob, 0o755 if mode == "100755" else 0o644))
    return result


def metadata(git, limits, workspace=True):
    root = git.root / ".git"
    if workspace and (
        (root / "shallow").exists()
        or (root / "info/sparse-checkout").exists()
        or list(root.glob("sharedindex.*"))
    ):
        raise SnapshotError("Shallow, split and sparse indexes are unsupported")
    # Inspect flags, including sparse-checkout skip-worktree entries.
    if workspace and any(
        line[:2].upper() == b"S "
        for line in git.run(
            ["ls-files", "-v"], maximum=limits["snapshot_bytes"]
        ).splitlines()
    ):
        raise SnapshotError("Sparse index is unsupported")
    entries = (
        git.run(["ls-files", "--stage", "-z"], maximum=limits["snapshot_bytes"])
        if workspace
        else b""
    )
    paths = []
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        meta, raw = entry.split(b"\t", 1)
        mode, _, stage = meta.split()
        if stage != b"0" or mode not in (b"100644", b"100755"):
            raise SnapshotError("Conflicts, symlinks and submodules are unsupported")
        path = raw.decode("utf-8")
        safe_parts(path)
        paths.append(path)
    head, branch = git.head()
    refs = git.run(
        ["for-each-ref", "--format=%(objectname) %(refname)"],
        maximum=limits["snapshot_bytes"],
    )
    if any(b" refs/replace/" in r for r in refs.splitlines()):
        raise SnapshotError("Replacement objects are unsupported")
    return {
        "head": head,
        "branch": branch,
        "refs": refs.decode("utf-8"),
        "index": hashlib.sha256(
            (optional(root, "index", limits["snapshot_bytes"]) or b"")
            if workspace
            else b""
        ).hexdigest(),
        "head_file": (optional(root, "HEAD", 4096) or b"").decode("ascii"),
    }, paths


def workspace_files(git, tracked, limits):
    raw = git.run(
        ["ls-files", "--others", "--exclude-standard", "-z"],
        maximum=limits["snapshot_bytes"],
    )
    names = sorted(set(tracked) | {p.decode("utf-8") for p in raw.split(b"\0") if p})
    if len(names) > limits["file_count"]:
        raise SnapshotError("Too many source files", "SNAPSHOT_LIMIT")
    result = {}
    for name in names:
        if name.split("/")[0].lower() == ".git":
            raise SnapshotError("Invalid source path")
        try:
            data, mode = read_regular(
                git.root,
                name,
                limits["file_bytes"],
                preserve_mode=limits.get("preserve_modes", False),
            )
        except FileNotFoundError:
            if name not in tracked:
                raise SnapshotError(
                    "Source changed during preparation", "SNAPSHOT_UNSTABLE"
                )
            result[name] = {"deleted": True}
            continue
        result[name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "mode": mode,
        }
    return result


def scan(root, limits):
    result = {}
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            path = Path(directory) / name
            if path.is_symlink():
                raise SnapshotError("Symlink in snapshot")
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            data, mode = read_regular(
                root,
                relative,
                limits["snapshot_bytes"],
                preserve_mode=limits.get("preserve_modes", False),
            )
            total += len(data)
            if total > limits["snapshot_bytes"]:
                raise SnapshotError(
                    "Snapshot exceeds total size limit", "SNAPSHOT_LIMIT"
                )
            result[relative] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
                "mode": mode,
            }
    return result


def prepare(source, output, target, limits):
    git = Git(source)
    # Resolve exactly once. Retries never reinterpret a moving user reference.
    fixed = git.resolve(target["ref"]) if target["kind"] == "commit" else None
    index_target = target["kind"] == "index"
    resolved_at = time.time()
    end = time.monotonic() + limits["snapshot_seconds"]
    for attempt in range(3):
        if time.monotonic() >= end:
            break
        try:
            before, tracked = metadata(git, limits, workspace=fixed is None)
            initial = (
                workspace_files(git, tracked, limits)
                if fixed is None and not index_target
                else None
            )
            if output.exists():
                shutil.rmtree(output)
            output.mkdir()
            total_written = 0

            def write(root, name, data, mode=0o644):
                nonlocal total_written
                total_written += len(data)
                if total_written > limits["snapshot_bytes"]:
                    raise SnapshotError(
                        "Snapshot exceeds total size limit", "SNAPSHOT_LIMIT"
                    )
                put(root, name, data, mode)

            if fixed or index_target:
                if fixed:
                    files = tree_records(git, fixed, limits["snapshot_bytes"])
                else:
                    files = []
                    for entry in git.run(
                        ["ls-files", "--stage", "-z"], maximum=limits["snapshot_bytes"]
                    ).split(b"\0"):
                        if entry:
                            meta, path = entry.split(b"\t", 1)
                            mode, blob, stage = meta.decode().split()
                            if mode not in ("100644", "100755") or stage != "0":
                                raise SnapshotError("Unsupported index content")
                            files.append(
                                (
                                    path.decode("utf-8"),
                                    blob,
                                    0o755 if mode == "100755" else 0o644,
                                )
                            )
                if len(files) > limits["file_count"]:
                    raise SnapshotError("Too many files", "SNAPSHOT_LIMIT")
                for path, blob, mode in files:
                    data = git.run(
                        ["cat-file", "blob", blob], maximum=limits["file_bytes"]
                    )
                    write(output, path, data, mode)
            else:
                for path, info in initial.items():
                    if not info.get("deleted"):
                        data, mode = read_regular(
                            source,
                            path,
                            limits["file_bytes"],
                            preserve_mode=limits.get("preserve_modes", False),
                        )
                        if (
                            hashlib.sha256(data).hexdigest() != info["sha256"]
                            or mode != info["mode"]
                        ):
                            raise SnapshotError(
                                "Source changed during copy", "SNAPSHOT_UNSTABLE"
                            )
                        write(output, path, data, mode)
            # Copy immutable objects, never hardlink or use alternates. No config,
            # hooks, logs, credentials, reflogs, or external object dependencies.
            objects = Path(source) / ".git/objects"
            for directory, dirs, files in os.walk(objects, followlinks=False):
                if any((Path(directory) / d).is_symlink() for d in dirs):
                    raise SnapshotError("External object storage is unsupported")
                for name in files:
                    path = (Path(directory) / name).relative_to(objects).as_posix()
                    if path.startswith("info/") or name.endswith(".promisor"):
                        if name.endswith(".promisor"):
                            raise SnapshotError("Partial repository is unsupported")
                        continue
                    data, _ = read_regular(objects, path, limits["snapshot_bytes"])
                    write(output, ".git/objects/" + path, data)
            (output / ".git/objects").mkdir(parents=True, exist_ok=True)
            (output / ".git/refs").mkdir(parents=True, exist_ok=True)
            write(
                output,
                ".git/config",
                b"[core]\nrepositoryformatversion = 0\nbare = false\nfilemode = true\nlogallrefupdates = false\n",
            )
            for line in before["refs"].splitlines():
                oid, ref = line.split(" ", 1)
                safe_parts(ref)
                write(output, ".git/" + ref, (oid + "\n").encode())
            write(
                output,
                ".git/HEAD",
                ((fixed + "\n") if fixed else before["head_file"]).encode(),
            )
            if fixed:
                copied = Git(output)
                copied.run(["read-tree", fixed])  # raw tree, no checkout/filters/hooks
            else:
                index = optional(
                    Path(source) / ".git", "index", limits["snapshot_bytes"]
                )
                if index is not None:
                    write(output, ".git/index", index)
            copied = Git(output)
            # Detect missing/corrupt reachable objects without running project code.
            copied.run(
                ["fsck", "--full", "--no-reflogs", "--no-dangling"],
                maximum=limits["snapshot_bytes"],
            )
            after, paths_after = metadata(git, limits, workspace=fixed is None)
            if (
                before != after
                or tracked != paths_after
                or (
                    initial is not None
                    and initial != workspace_files(git, paths_after, limits)
                )
            ):
                raise SnapshotError(
                    "Source changed during preparation", "SNAPSHOT_UNSTABLE"
                )
            manifest = scan(output, limits)
            if initial is not None:
                for path, info in initial.items():
                    if not info.get("deleted") and manifest.get(path) != info:
                        raise SnapshotError("Source copy mismatch", "SNAPSHOT_UNSTABLE")
            return {
                "target": target,
                "commit_sha": fixed or before["head"],
                "resolved_at": resolved_at,
                "manifest": manifest,
                "content_digest": digest(
                    {k: v for k, v in manifest.items() if not k.startswith(".git/")}
                ),
                "git_digest": digest(
                    {k: v for k, v in manifest.items() if k.startswith(".git/")}
                ),
                "attempts": attempt + 1,
            }
        except SnapshotError as exc:
            if exc.code != "SNAPSHOT_UNSTABLE":
                raise
        except FileNotFoundError:
            pass
    raise SnapshotError(
        "Workspace did not stabilize within preparation budget", "SNAPSHOT_UNSTABLE"
    )


def main():
    try:
        req = json.load(sys.stdin)
        if req.get("stage") == "review":
            from code_review import prepare_review

            result = prepare_review(
                Path("/workspace"),
                Path("/snapshot/source"),
                req["target"],
                req["limits"],
            )
        else:
            result = prepare(
                Path("/workspace"),
                Path("/snapshot/source"),
                req["target"],
                req["limits"],
            )
        # Manifest travels via the managed snapshot directory, never stdout limits.
        Path("/snapshot/manifest.json").write_text(json.dumps(result, sort_keys=True))
        print(
            json.dumps(
                {
                    "ok": True,
                    "content_digest": result["content_digest"],
                    "git_digest": result["git_digest"],
                }
            )
        )
    except (ValueError, OSError, GitError) as exc:
        print(
            json.dumps(
                {
                    "error": str(exc)[:500],
                    "code": getattr(exc, "code", "SNAPSHOT_UNSUPPORTED"),
                }
            )
        )


if __name__ == "__main__":
    main()
