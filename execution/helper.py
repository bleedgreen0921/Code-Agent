"""Trusted image-only helper. Invoked by absolute path with Python -I and JSON stdin."""

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

class VersionConflict(ValueError):
    pass


VERSION = "workspace-v1"
ROOT = Path("/workspace")


def virtual_parts(path):
    if path.startswith("/workspace/"):
        path = path[len("/workspace/") :]
    parts = path.lstrip("/").split("/")
    if any(p in ("..", "") for p in parts) or "\x00" in path:
        raise ValueError("Invalid workspace path")
    return [p for p in parts if p != "."]


def parent(path):
    parts = virtual_parts(path)
    if not parts:
        raise ValueError("A regular file is required")
    fd = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = child
        return fd, parts[-1]
    except BaseException:
        os.close(fd)
        raise


def read_at(fd, name, maximum):
    handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    with os.fdopen(handle, "rb") as file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Only regular files are supported")
        if info.st_size > maximum:
            raise ValueError("File exceeds size limit")
        data = file.read(maximum + 1)
        if len(data) > maximum:
            raise ValueError("File exceeds size limit")
    return data, info


def file_operation(operation, args, maximum):
    fd, name = parent(args["path"])
    temp_name = None
    try:
        if operation == "read_file":
            data, _ = read_at(fd, name, maximum)
            lines = data.decode("utf-8").splitlines()
            start, count = args.get("start_line", 1), args.get("max_lines", 200)
            if not 1 <= start or not 1 <= count <= 200:
                raise ValueError("Invalid line range")
            end = start - 1 + count
            return {
                "content": "\n".join(
                    f"{i}: {line}"
                    for i, line in enumerate(lines[start - 1 : end], start)
                ),
                "sha256": hashlib.sha256(data).hexdigest(),
                "truncated": len(lines) > end,
                "total_lines": len(lines),
                "start_line": start,
            }
        create = args.get("create", False)
        if create:
            if (
                args.get("expected_hash") is not None
                or args.get("old_text") is not None
            ):
                raise ValueError("Create accepts only new_text")
            data, info = None, None
            content = args["new_text"]
        else:
            data, info = read_at(fd, name, maximum)
            if hashlib.sha256(data).hexdigest() != args.get("expected_hash"):
                raise VersionConflict("Content hash conflict; read the file again")
            text = data.decode("utf-8")
            old = args.get("old_text")
            if not old or text.count(old) != 1:
                raise ValueError("Replacement must match exactly once")
            content = text.replace(old, args["new_text"], 1)
        updated = content.encode("utf-8")
        if len(updated) > maximum:
            raise ValueError("File exceeds size limit")
        temp_name = ".agent-edit-" + os.urandom(16).hex()
        out = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=fd,
        )
        with os.fdopen(out, "wb") as file:
            file.write(updated)
            file.flush()
            os.fchmod(
                file.fileno(), stat.S_IMODE(info.st_mode) & 0o777 if info else 0o644
            )
            os.fsync(file.fileno())
        if create:
            # Atomic no-overwrite publication, even when a concurrent creator wins.
            os.link(
                temp_name, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False
            )
        else:
            current, current_info = read_at(fd, name, maximum)
            if current != data or (current_info.st_dev, current_info.st_ino) != (
                info.st_dev,
                info.st_ino,
            ):
                raise VersionConflict("Content hash conflict; read the file again")
            os.replace(temp_name, name, src_dir_fd=fd, dst_dir_fd=fd)
        return {
            "content": "File created" if create else "File updated",
            "sha256": hashlib.sha256(updated).hexdigest(),
        }
    finally:
        if temp_name:
            try:
                os.unlink(temp_name, dir_fd=fd)
            except FileNotFoundError:
                pass
        os.close(fd)


def search(operation, args, maximum):
    from langchain.agents.middleware.file_search import FilesystemFileSearchMiddleware

    class Search(FilesystemFileSearchMiddleware):
        def _python_search(self, *args):
            raise RuntimeError("ripgrep fallback is disabled")

        def _ripgrep_search(self, pattern, base_path, include):
            base = self._validate_and_resolve_path(base_path)
            command = [
                "rg",
                "--json",
                "--max-filesize",
                str(maximum),
                "--max-columns",
                "4096",
            ]
            if include:
                command += ["--glob", include]
            command += ["--", pattern, str(base)]
            results = {}
            retained = 0
            with subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            ) as proc:
                try:
                    while raw := proc.stdout.readline(65536):
                        if not raw.endswith(b"\n"):
                            self.search_truncated = True
                            while raw and not raw.endswith(b"\n"):
                                raw = proc.stdout.readline(65536)
                            continue
                        record = json.loads(raw)
                        if record["type"] != "match":
                            continue
                        item = record["data"]
                        path = Path(item["path"]["text"])
                        if not path.resolve().is_relative_to(ROOT):
                            continue
                        if not path.is_file() or path.stat().st_size > maximum:
                            continue
                        name = "/" + str(path.relative_to(ROOT))
                        line = item["lines"]["text"].rstrip("\n")
                        results.setdefault(name, []).append((item["line_number"], line))
                        retained += len(raw)
                        if retained > 20000 or sum(map(len, results.values())) >= 200:
                            self.search_truncated = True
                            break
                finally:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait()
            if proc.returncode not in (0, 1, -9):
                raise ValueError("ripgrep search failed")
            return results

    engine = Search(root_path=str(ROOT), use_ripgrep=True)
    engine.search_truncated = False
    # Reject traversal and out-of-root links before native search performs its own checks.
    path = args.get("path", "/")
    if path != "/":
        virtual_parts(path)
    base = engine._validate_and_resolve_path(path)
    if not base.resolve().is_relative_to(ROOT):
        raise ValueError("Path outside workspace")
    if base.exists() and not (base.is_dir() or stat.S_ISREG(base.stat().st_mode)):
        raise ValueError("Search accepts only directories and regular files")
    result = getattr(engine, operation).invoke(args)
    return {"content": result, "truncated": engine.search_truncated}


def main():
    request = json.load(sys.stdin)
    operation = request["operation"]
    if operation == "version":
        import langchain

        subprocess.run(["rg", "--version"], stdout=subprocess.DEVNULL, check=True)
        assert langchain.__version__ == "1.4.2"
        print(json.dumps({"version": VERSION}))
        return
    try:
        maximum = request["file_bytes"]
        args = request["args"]
        if operation in ("read_file", "edit_file"):
            result = file_operation(operation, args, maximum)
        elif operation in ("glob_search", "grep_search"):
            result = search(operation, args, maximum)
        else:
            raise ValueError("Unknown operation")
        # Bound the wire response before JSON encoding. Host applies the final limit too.
        raw = result["content"].encode("utf-8")
        if len(raw) > 24000:
            result["content"] = raw[:24000].decode("utf-8", "ignore")
            result["truncated"] = True
        print(json.dumps(result, ensure_ascii=True))
    except (ValueError, OSError, UnicodeError) as exc:
        print(json.dumps({"error": str(exc)[:500], "code": "FILE_VERSION_CONFLICT" if isinstance(exc, VersionConflict) else "FILE_OPERATION_FAILED"}))


if __name__ == "__main__":
    main()
