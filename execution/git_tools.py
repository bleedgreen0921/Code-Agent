"""Read-only Git operations, loaded only by the trusted workspace helper.

No shell, checkout, conversion drivers, network, or repository writes. Streams
are drained with bounded buffers and reaped even when a page stops early.
"""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess


class GitError(ValueError):
    def __init__(self, message, code="GIT_READ_FAILED"):
        super().__init__(message)
        self.code = code


def quoted(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def literal_path(value):
    if not isinstance(value, str) or not value or value.startswith(("/", ":")):
        raise GitError("A literal workspace-relative file path is required", "GIT_ARGUMENTS_INVALID")
    if "\x00" in value or any(p in ("", ".", "..") for p in value.split("/")):
        raise GitError("Invalid workspace-relative path", "GIT_ARGUMENTS_INVALID")
    return value


class Git:
    def __init__(self, root):
        self.root = Path(root)
        metadata = self.root / ".git"
        if not metadata.is_dir() or metadata.is_symlink():
            raise GitError("Workspace root requires an internal .git directory; external worktrees, submodules and bare repositories are unsupported", "GIT_REPOSITORY_UNAVAILABLE")
        for name in ("commondir", "objects/info/alternates", "objects/info/http-alternates"):
            if (metadata / name).exists():
                raise GitError("External Git metadata or object storage is unsupported", "GIT_UNSUPPORTED")
        for name in ("HEAD", "config", "index", "objects", "refs", "packed-refs"):
            p = metadata / name
            if p.is_symlink() or not p.resolve().is_relative_to(metadata):
                raise GitError("External Git metadata is unsupported", "GIT_UNSUPPORTED")
        self.env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8",
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat", "GIT_NO_REPLACE_OBJECTS": "1", "GIT_NO_LAZY_FETCH": "1",
            "GIT_LITERAL_PATHSPECS": "1",
        }
        self.argv = ["git", "--no-pager", "--no-optional-locks", "--git-dir=" + str(metadata), "--work-tree=" + str(self.root)]
        for key, value in {
            "safe.directory": str(self.root), "core.fsmonitor": "false",
            "core.untrackedCache": "false", "core.hooksPath": "/dev/null",
            "core.attributesFile": "/dev/null", "core.quotePath": "true",
            "color.ui": "false", "diff.external": "", "diff.autoRefreshIndex": "false",
            "submodule.recurse": "false", "protocol.allow": "never",
            "log.showSignature": "false", "log.outputEncoding": "UTF-8",
            "core.warnAmbiguousRefs": "true",
        }.items():
            self.argv += ["-c", key + "=" + value]
        # Never invoke lazy fetching / remote helpers on partial repositories.
        if self.run(["config", "--get-regexp", r"^(extensions\.partialclone|remote\..*\.promisor)$"], allowed=(0, 1)):
            raise GitError("Partial repositories and automatic object recovery are unsupported", "GIT_UNSUPPORTED")
        if self.run(["rev-parse", "--is-bare-repository"]).strip() != b"false":
            raise GitError("Bare repositories are unsupported", "GIT_REPOSITORY_UNAVAILABLE")

    @contextmanager
    def stream(self, args, allowed=(0,), reject_stderr=False):
        proc = subprocess.Popen(self.argv + args, cwd=self.root, env=self.env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ, "out")
        selector.register(proc.stderr, selectors.EVENT_READ, "err")
        error = bytearray()

        def chunks():
            while selector.get_map():
                for key, _ in selector.select():
                    chunk = os.read(key.fileobj.fileno(), 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    elif key.data == "out":
                        yield chunk
                    elif len(error) < 512:
                        error.extend(chunk[:512 - len(error)])
            code = proc.wait()
            if code not in allowed or (reject_stderr and error):
                # Do not expose arguments, commit bodies or untrusted stderr.
                raise GitError("Git read failed: repository, reference or object is unavailable/invalid")

        try:
            yield chunks()
        finally:
            selector.close()
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            proc.stdout.close()
            proc.stderr.close()

    def run(self, args, maximum=65536, **kwargs):
        result = bytearray()
        with self.stream(args, **kwargs) as chunks:
            for chunk in chunks:
                if len(result) + len(chunk) > maximum:
                    raise GitError("Git object or metadata exceeds the read limit; not fully read", "GIT_UNSUPPORTED")
                result.extend(chunk)
        return bytes(result)

    @contextmanager
    def records(self, args, separator=b"\x00", **kwargs):
        with self.stream(args, **kwargs) as chunks:
            def items():
                buffer = bytearray()
                for chunk in chunks:
                    buffer.extend(chunk)
                    while (end := buffer.find(separator)) >= 0:
                        if end > 65536:
                            raise GitError("Oversized Git record or single line; not fully read", "GIT_UNSUPPORTED")
                        yield bytes(buffer[:end])
                        del buffer[:end + len(separator)]
                    if len(buffer) > 65536:
                        raise GitError("Oversized Git record or single line; not fully read", "GIT_UNSUPPORTED")
                if buffer:
                    yield bytes(buffer)
            yield items()

    def resolve(self, revision):
        if (not isinstance(revision, str) or len(revision) > 1024
            or not re.fullmatch(r"[^\x00\s:~^{}]+(?:[~^][0-9]*)*", revision)
            or revision.startswith("-") or ".." in revision):
            raise GitError("Expected one commit reference, not a range or option", "GIT_REFERENCE_INVALID")
        try:
            return self.run(["rev-parse", "--verify", "--end-of-options", revision + "^{commit}"], reject_stderr=True).decode("ascii").strip()
        except GitError as exc:
            raise GitError("Commit reference is invalid, ambiguous, not a commit, or missing locally", "GIT_REFERENCE_INVALID") from exc

    def head(self):
        branch = self.run(["symbolic-ref", "--quiet", "HEAD"], allowed=(0, 1)).decode("utf-8").strip()
        if branch:
            # Only a genuinely absent branch is an unborn HEAD. Broken objects fail.
            exists = self.run(["show-ref", "--", branch], allowed=(0, 1))
            if not exists:
                return None, branch
        return self.resolve("HEAD"), branch or None

    def no_filters(self):
        if self.run(["config", "--null", "--get-regexp", r"^filter\..*\.(clean|process)$"], allowed=(0, 1)):
            raise GitError("Current status and unstaged diff with configured external clean/process filters are unsupported; no filter was executed", "GIT_UNSUPPORTED")

    def parents(self, oid):
        # Read only the commit header, never materialize the commit body.
        with self.records(["cat-file", "commit", oid], separator=b"\n") as records:
            parents = []
            for line in records:
                if not line:
                    break
                if line.startswith(b"parent "):
                    parents.append(line[7:].decode("ascii"))
                    if len(parents) > 1000:
                        raise GitError("Oversized commit parent metadata; not fully read", "GIT_UNSUPPORTED")
            return parents

    def entries(self, path, oid=None):
        argv = ["ls-tree", "-z", oid] if oid else ["ls-files", "--stage", "-z"]
        raw = self.run([*argv, "--", path])
        entries = []
        for record in raw.split(b"\x00"):
            if not record:
                continue
            fields, filename = record.split(b"\t", 1)
            if decode(filename) == path:
                entries.append(fields.split())
        return entries

    def check_path(self, path, oid=None, worktree=False):
        parts = path.split("/")
        for n in range(1, len(parts) + 1):
            prefix = "/".join(parts[:n])
            entries = self.entries(prefix, oid)
            if oid is None and any(e[2] != b"0" for e in entries):
                raise GitError("Conflicted index: no unique stage 0 version; patch/context not fully read", "GIT_INDEX_CONFLICT")
            if any(e[0] in (b"120000", b"160000") for e in entries):
                raise GitError("Symlink or submodule content is unsupported; not fully read", "GIT_UNSUPPORTED")
            if worktree and (self.root / prefix).is_symlink():
                raise GitError("Workspace symlink content is unsupported", "GIT_UNSUPPORTED")
        return entries

    def observation(self, path=None):
        names = [self.root / ".git/index", self.root / ".git/HEAD"]
        if path:
            names.append(self.root / path)
        def stamp(p):
            try:
                s = p.lstat()
                return (s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            except FileNotFoundError:
                return None
        return [stamp(p) for p in names], self.head()


class Page:
    def __init__(self, metadata, output_bytes, output_lines, start, next_key):
        self.meta = metadata
        self.header = quoted(metadata)
        self.body = []
        self.items = []
        self.maximum = output_bytes
        self.lines = output_lines
        self.start = start
        self.next_key = next_key
        self.blocked = False
        if len(self.header.encode()) + 170 > output_bytes:
            raise GitError("Source metadata exceeds output budget; no records read", "GIT_UNSUPPORTED")

    def add(self, text, item):
        artifact_bytes = len(json.dumps({**self.meta, "records": [*self.items, item]}, ensure_ascii=False).encode())
        if (len(self.body) >= self.lines - 2 or artifact_bytes + 170 > self.maximum or
            len((self.header + "\n" + "\n".join(self.body) + "\n" + text).encode()) + 170 > self.maximum):
            self.blocked = not self.body
            return False
        self.body.append(text)
        self.items.append(item)
        return True

    def finish(self, more=False):
        next_position = self.start + len(self.items) if more else None
        footer = {"truncated": more, self.next_key: next_position}
        if self.blocked:
            footer["notice"] = "Next record/line exceeds output budget; not read."
        return {**self.meta, **footer, "records": self.items,
                "content": "\n".join([self.header, *self.body, quoted(footer)])}


def decode(value):
    return value.decode("utf-8", "strict")


def paginate(page, records, offset, limit, render):
    for number, record in enumerate(records):
        if number < offset:
            continue
        if len(page.items) >= limit:
            return page.finish(True)
        text, item = render(record, number)
        if not page.add(text, item):
            return page.finish(True)
    return page.finish()


def operation(root, name, args, maximum, output_bytes=16384, output_lines=200):
    if args.get("path") is not None:
        literal_path(args["path"])
    git = Git(root)
    source = args.get("source")
    live = name == "git_status" or source in ("staged", "unstaged", "index")
    before = git.observation(args.get("path")) if live else None
    result = perform(git, name, args, maximum, output_bytes, output_lines)
    if live:
        if before != git.observation(args.get("path")):
            # Retry only observation after detected movement, not a failed command.
            result = perform(git, name, args, maximum, output_bytes, output_lines)
            result["content"] = result["content"].replace('"observation":"live"', '"observation":"live; source changed, reread; no atomic snapshot"', 1)
            # Reserve the same header space for this note in perform().
            result["source_changed"] = True
    return result


def perform(git, name, args, maximum, output_bytes, output_lines):
    source = args.get("source")
    def page(meta, start, key):
        return Page(meta, output_bytes - 80, output_lines, start, key)

    if name == "git_status":
        git.no_filters()
        head, branch = git.head()
        meta = {"source": "status", "HEAD": head, "branch": branch, "observation": "live", "notice": "No atomic snapshot; submodule interiors excluded"}
        offset, limit = args.get("offset", 0), args.get("limit", 50)
        p = page(meta, offset, "next_offset")
        def status(raw, number):
            xy, path = raw[:2].decode("ascii"), decode(raw[3:])
            conflict = xy in ("DD", "AU", "UD", "UA", "DU", "AA", "UU")
            item = {"path": path, "xy": xy, "staged": xy[0] not in " ?", "unstaged": xy[1] not in " ?", "untracked": xy == "??", "conflict": conflict}
            return quoted(item), item
        with git.records(["status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames", "--ignore-submodules=dirty"]) as records:
            return paginate(p, records, offset, limit, status)
    if name == "git_log":
        oid = git.resolve(args.get("revision", "HEAD"))
        offset, limit = args.get("offset", 0), args.get("limit", 10)
        p = page({"source": "log", "revision": oid, "notice": "Use this full revision for subsequent pages"}, offset, "next_offset")
        with git.records(["log", "--no-decorate", "--no-show-signature", "--no-notes", "--encoding=UTF-8", "--format=%H%x00%P%x00%an%x00%aI%x00%s", "-z", "--skip=" + str(offset), "--max-count=" + str(limit + 1), oid, "--"]) as records:
            def commits():
                while True:
                    h = next(records, None)
                    if h is None:
                        break
                    try:
                        fields = [h, *[next(records) for _ in range(4)]]
                    except StopIteration as exc:
                        raise GitError("Malformed commit metadata; not fully read", "GIT_UNSUPPORTED") from exc
                    yield {"commit": decode(fields[0]), "parents": decode(fields[1]).split(), "author": decode(fields[2]), "author_time": decode(fields[3]), "title": decode(fields[4])}
            return paginate(p, commits(), 0, limit, lambda item, n: (quoted(item), item))
    if name == "git_read_file":
        path = literal_path(args["path"])
        if source == "commit":
            oid = git.resolve(args["commit"])
            entries = git.check_path(path, oid)
            meta = {"source": "commit", "commit": oid, "path": path}
        else:
            entries = git.check_path(path)
            meta = {"source": "index", "path": path, "observation": "live"}
            if entries and (len(entries) != 1 or entries[0][2] != b"0"):
                raise GitError("Conflicted index: no unique stage 0 version", "GIT_INDEX_CONFLICT")
        if len(entries) != 1:
            raise GitError("File does not exist in the selected source", "GIT_FILE_NOT_FOUND")
        entry = entries[0]
        if entry[0] not in (b"100644", b"100755") or (source == "commit" and entry[1] != b"blob"):
            raise GitError("Only regular blobs can be read; symlinks, trees and submodule interiors are unsupported", "GIT_UNSUPPORTED")
        blob = decode(entry[2] if source == "commit" else entry[1])
        size = int(git.run(["cat-file", "-s", blob]))
        if size > maximum:
            raise GitError("Blob exceeds file size limit; not read", "GIT_UNSUPPORTED")
        data = git.run(["cat-file", "blob", blob], maximum=maximum)
        if b"\x00" in data:
            raise GitError("Binary blob; not read as text", "GIT_UNSUPPORTED")
        lines = decode(data).splitlines()
        start, count = args.get("start_line", 1), args.get("max_lines", 200)
        p = page({**meta, "blob": blob, "sha256": hashlib.sha256(data).hexdigest(), "total_lines": len(lines)}, start, "next_line")
        return paginate(p, lines, start - 1, count, lambda line, n: (f"{n + 1}: {quoted(line)}", {"line": n + 1, "text": line}))
    if name != "git_diff":
        raise GitError("Unknown Git operation", "GIT_ARGUMENTS_INVALID")
    if source == "unstaged":
        git.no_filters()
    common = ["--no-ext-diff", "--no-textconv", "--no-renames", "--ignore-submodules=dirty", "--submodule=short", "--no-color", "--no-relative", "--src-prefix=a/", "--dst-prefix=b/"]
    meta = {"source": source}
    if source == "commit":
        oid = git.resolve(args["commit"])
        parents = git.parents(oid)
        command = ["diff-tree", "--no-commit-id", "-r", *common]
        command += [parents[0], oid] if parents else ["--root", oid]
        meta.update(commit=oid, base=parents[0] if parents else "root (Git native)", parent_count=len(parents), comparison="first parent")
    else:
        head, _ = git.head()
        command = ["diff", *common] + (["--cached"] if source == "staged" else [])
        if source == "staged" and head:
            command.append(head)
        base = ("HEAD -> index" if head else "unborn HEAD -> index (additions)") if source == "staged" else "index -> worktree"
        meta.update(HEAD=head, base=base, observation="live")
    path = literal_path(args["path"]) if args.get("path") else None
    view = args.get("view", "files")
    meta.update(view=view, path=path, notice="Submodule pointers only; no interiors" if view == "files" else "Output positions are NOT source lines; locate issues using hunk old/new lines; submodule interiors excluded")
    start, count = args.get("start_line", 1), args.get("max_lines", 200)
    p = page(meta, start, "next_line")
    if view == "files":
        with git.records(command + ["--raw", "--abbrev=64", "-z", "--", *([path] if path else [])]) as records:
            def changes():
                while (raw := next(records, None)) is not None:
                    modes = raw.decode("ascii").split()
                    filename = decode(next(records))
                    yield {"path": filename, "status": modes[4], "old_mode": modes[0][1:], "new_mode": modes[1], "old_blob": modes[2], "new_blob": modes[3]}
            return paginate(p, changes(), start - 1, count, lambda item, n: (f"{n + 1}: {quoted(item)}", item))
    if path is None:
        raise GitError("Patch view requires one literal file path; list files first", "GIT_ARGUMENTS_INVALID")
    # Check both sides: deletion must not hide an old symlink/submodule.
    entries = git.check_path(path, meta.get("commit"), worktree=source == "unstaged")
    previous = parents[0] if source == "commit" and parents else head if source == "staged" else None
    if previous:
        entries += git.check_path(path, previous)
    if any(e[0] == b"040000" for e in entries):
        raise GitError("Patch view requires one regular file", "GIT_UNSUPPORTED")
    if not entries:
        raise GitError("No versioned regular file at this path; read untracked files with read_file", "GIT_FILE_NOT_FOUND")
    if source == "unstaged":
        try:
            mode = (git.root / path).lstat().st_mode
        except FileNotFoundError:
            mode = None
        if mode is not None and not stat.S_ISREG(mode):
            raise GitError("Patch view requires one regular file", "GIT_UNSUPPORTED")
    with git.records(command + ["--patch", "--unified=3", "--", path], separator=b"\n") as records:
        def patch(raw, n):
            line = decode(raw)
            if line.startswith(("Binary files ", "GIT binary patch")):
                raise GitError("Binary diff; patch not fully read", "GIT_UNSUPPORTED")
            return f"{n + 1}: {quoted(line)}", {"position": n + 1, "text": line}
        return paginate(p, records, start - 1, count, patch)
