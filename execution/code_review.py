"""Trusted fixed-tree review preparation and blob reads. Never runs source code."""

import hashlib
import json
import re
import subprocess
import time

try:
    from .code_snapshot import (
        Git,
        GitError,
        SnapshotError,
        digest,
        prepare,
        put,
        read_regular,
        scan,
    )
except ImportError:
    from code_snapshot import (
        Git,
        GitError,
        SnapshotError,
        digest,
        prepare,
        put,
        read_regular,
        scan,
    )


def write_git(git, args, data=b""):
    result = subprocess.run(
        git.argv + args,
        cwd=git.root,
        env=git.env,
        input=data,
        capture_output=True,
        timeout=60,
    )
    if result.returncode or len(result.stdout) > 65536:
        raise SnapshotError("Git tree construction failed")
    return result.stdout.decode("ascii").strip()


def make_tree(git, entries):
    # Independent index; all writes go to the managed copy, raw blobs bypass filters.
    index = git.root / ".git/review-index"
    git.env["GIT_INDEX_FILE"] = str(index)
    try:
        write_git(git, ["read-tree", "--empty"])
        data = b"".join(
            f"{mode} {blob}\t".encode() + path.encode() + b"\0"
            for path, blob, mode in sorted(entries)
        )
        write_git(git, ["update-index", "-z", "--index-info"], data)
        return write_git(git, ["write-tree"])
    finally:
        git.env.pop("GIT_INDEX_FILE", None)
        index.unlink(missing_ok=True)


def tree_id(git, commit):
    return git.run(["rev-parse", commit + "^{tree}"]).decode().strip()


def index_entries(git, limits):
    result = []
    for item in git.run(
        ["ls-files", "--stage", "-z"], maximum=limits["snapshot_bytes"]
    ).split(b"\0"):
        if item:
            meta, path = item.split(b"\t", 1)
            mode, blob, stage = meta.decode().split()
            if stage != "0" or mode not in ("100644", "100755"):
                raise SnapshotError("Unsupported index entry")
            result.append((path.decode("utf-8"), blob, mode))
    return result


def source_entries(git, limits):
    entries = []
    for path, info in scan(git.root, limits).items():
        if path.startswith(".git/"):
            continue
        data, _ = read_regular(git.root, path, limits["file_bytes"])
        blob = write_git(git, ["hash-object", "-w", "--stdin"], data)
        entries.append((path, blob, "100755" if info["mode"] & 0o111 else "100644"))
    return entries


def parse_patch(text, change_id):
    records, hunks = [], []
    old = new = 0
    hunk = None
    for position, line in enumerate(text.splitlines(), 1):
        match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if match:
            old, old_count, new, new_count = (
                int(match[1]),
                int(match[2] or 1),
                int(match[3]),
                int(match[4] or 1),
            )
            hunk = {
                "hunk_id": digest([change_id, position, line]),
                "output_start": position,
                "old_start": old,
                "old_count": old_count,
                "new_start": new,
                "new_count": new_count,
                "base_lines": [],
                "head_lines": [],
            }
            hunks.append(hunk)
        record = {
            "output_line": position,
            "text": line,
            "old_line": None,
            "new_line": None,
            "hunk_id": hunk["hunk_id"] if hunk else None,
        }
        if hunk and not match:
            if line.startswith("-"):
                record["old_line"] = old
                hunk["base_lines"].append(old)
                old += 1
            elif line.startswith("+"):
                record["new_line"] = new
                hunk["head_lines"].append(new)
                new += 1
            elif line.startswith(" "):
                record["old_line"], record["new_line"] = old, new
                old += 1
                new += 1
        records.append(record)
        if hunk:
            hunk["output_end"] = position
    return records, hunks


def changes(git, metadata, limits):
    base, head = metadata["base_tree"], metadata["diff_head_tree"]
    raw = git.run(
        [
            "diff",
            "--raw",
            "-z",
            "--no-abbrev",
            "--no-renames",
            "--no-ext-diff",
            "--no-textconv",
            base,
            head,
            "--",
        ],
        maximum=limits["snapshot_bytes"],
    )
    items = raw.split(b"\0")
    result, used = [], 0
    global_limits = []
    encoded_bytes = 0
    for i in range(0, len(items) - 1, 2):
        modes = items[i].decode().split()
        old_mode, new_mode, old_blob, new_blob, status = modes
        path = items[i + 1].decode("utf-8")
        entry = {
            "status": status,
            "base_path": None if status == "A" else path,
            "head_path": None if status == "D" else path,
            "base_blob": None if set(old_blob) == {"0"} else old_blob,
            "head_blob": None if set(new_blob) == {"0"} else new_blob,
            "base_mode": old_mode.lstrip(":"),
            "head_mode": new_mode,
            "kind": "text",
            "limitations": [],
        }
        entry["change_id"] = digest([metadata["range_id"], entry])
        entry["records"], entry["hunks"] = [], []
        if any(
            entry[key] not in ("000000", "100644", "100755")
            for key in ("base_mode", "head_mode")
        ):
            entry["limitations"].append("unsupported_base_type")
        try:
            patch = git.run(
                [
                    "diff",
                    "--no-color",
                    "--no-renames",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--unified=3",
                    base,
                    head,
                    "--",
                    path,
                ],
                maximum=limits["patch_bytes"],
            )
            used += len(patch)
            if used > limits["diff_bytes"]:
                entry["limitations"].append("diff_budget")
            elif any(len(line) > 8192 for line in patch.splitlines()):
                entry["limitations"].append("long_line")
            else:
                entry["records"], entry["hunks"] = parse_patch(
                    patch.decode("utf-8"), entry["change_id"]
                )
            if b"Binary files " in patch or b"GIT binary patch" in patch:
                entry["kind"] = "binary"
                entry["limitations"].append("binary_content")
        except (GitError, UnicodeError):
            entry["limitations"].append("patch_unavailable_or_limit")
        entry["file_location"] = (
            (
                (
                    entry["base_mode"] in ("100644", "100755")
                    and entry["head_mode"] in ("100644", "100755")
                    and entry["base_mode"] != entry["head_mode"]
                )
                or (not entry["hunks"] and status in ("A", "D"))
            )
            and entry["kind"] == "text"
            and not entry["limitations"]
        )
        entry_size = len(json.dumps(entry, ensure_ascii=False).encode())
        if encoded_bytes + entry_size > limits["diff_bytes"]:
            entry["records"], entry["hunks"] = [], []
            entry["limitations"].append("diff_artifact_budget")
            entry["file_location"] = False
        encoded_bytes += len(json.dumps(entry, ensure_ascii=False).encode())
        result.append(entry)
    if encoded_bytes > limits["diff_bytes"]:
        global_limits.append("change_metadata_exceeds_diff_budget")
    return {
        "range": metadata,
        "changes": result,
        "limitations": global_limits,
        "list_complete": True,
        "notice": "Rename detection disabled: renames appear as deletion plus addition.",
    }


def prepare_review(source, output, target, limits):
    original = Git(source)
    resolved_at = time.time()
    kind = target["kind"]
    requested_base = requested_head = base_commit = head_commit = None
    parents = []
    if kind == "commit":
        requested_head = target["ref"]
        head_commit = original.resolve(requested_head)
        parents = original.parents(head_commit)
        base_commit = parents[0] if parents else None
        requested_base = "first_parent"
    elif kind == "range":
        requested_base, requested_head = target["base"], target["head"]
        base_commit, head_commit = (
            original.resolve(requested_base),
            original.resolve(requested_head),
        )
        resolved_base = base_commit
        if target.get("comparison", "direct") == "merge_base":
            candidates = (
                original.run(
                    ["merge-base", "--all", base_commit, head_commit], allowed=(0, 1)
                )
                .decode()
                .splitlines()
            )
            if len(candidates) != 1:
                raise SnapshotError(
                    "merge_base requires exactly one best common ancestor",
                    "REVIEW_RANGE_INVALID",
                )
            base_commit = candidates[0]
    resolved_base = resolved_base if kind == "range" else base_commit
    # prepare resolves only already fixed object IDs; moving refs are never reused.
    snapshot_target = (
        {"kind": "commit", "ref": head_commit}
        if head_commit
        else {"kind": "index"}
        if target.get("mode") == "staged"
        else {"kind": "workspace"}
    )
    snapshot = prepare(source, output, snapshot_target, limits)
    git = Git(output)
    empty = make_tree(git, [])
    metadata = {
        "requested": target,
        "requested_base": requested_base,
        "requested_head": requested_head,
        "resolved_base_commit": resolved_base,
        "head_commit": head_commit or snapshot["commit_sha"],
        "comparison_base_commit": base_commit,
        "parent_count": len(parents),
        "selected_parent": parents[0] if parents else None,
        "resolved_at": resolved_at,
        "execution_context": "head snapshot; unstaged includes nonignored untracked context but excludes it from diff",
    }
    if kind == "workspace":
        mode = target.get("mode", "all")
        entries = index_entries(git, limits)
        index_tree = make_tree(git, entries)
        metadata["comparison_base_commit"] = (
            snapshot["commit_sha"] if mode != "unstaged" else None
        )
        metadata["base_tree"] = (
            index_tree
            if mode == "unstaged"
            else tree_id(git, snapshot["commit_sha"])
            if snapshot["commit_sha"]
            else empty
        )
        if mode == "staged":
            for path in snapshot["manifest"]:
                if not path.startswith(".git/"):
                    (output / path).unlink()
            # Empty directories left by the worktree can conflict with index files.
            for directory in sorted(
                (
                    p
                    for p in output.rglob("*")
                    if p.is_dir() and ".git" not in p.relative_to(output).parts
                ),
                key=lambda p: len(p.parts),
                reverse=True,
            ):
                if not any(directory.iterdir()):
                    directory.rmdir()
            for path, blob, filemode in entries:
                put(
                    output,
                    path,
                    git.run(["cat-file", "blob", blob], maximum=limits["file_bytes"]),
                    0o755 if filemode == "100755" else 0o644,
                )
            head_tree = index_tree
        else:
            actual = source_entries(git, limits)
            head_tree = make_tree(git, actual)
            if mode == "unstaged":
                tracked = {p for p, _, _ in entries}
                metadata["diff_head_tree"] = make_tree(
                    git, [e for e in actual if e[0] in tracked]
                )
        metadata["head_tree"] = head_tree
        metadata.setdefault("diff_head_tree", head_tree)
        metadata["index_tree"] = index_tree
    else:
        metadata["base_tree"] = tree_id(git, base_commit) if base_commit else empty
        metadata["head_tree"] = metadata["diff_head_tree"] = tree_id(git, head_commit)
    # Bind both execution content and comparison identities into the range.
    manifest = scan(output, limits)
    metadata["content_digest"] = digest(
        {k: v for k, v in manifest.items() if not k.startswith(".git/")}
    )
    metadata["range_id"] = digest(metadata)
    document = changes(git, metadata, limits)
    (output.parent / "changes.json").write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True)
    )
    snapshot.update(
        manifest=manifest,
        content_digest=metadata["content_digest"],
        git_digest=digest({k: v for k, v in manifest.items() if k.startswith(".git/")}),
        target=target,
        review=metadata,
    )
    return snapshot


def read_blob(root, metadata, side, path, maximum):
    git = Git(root)
    tree = metadata["base_tree" if side == "base" else "head_tree"]
    from git_tools import literal_path

    literal_path(path)
    entries = git.entries(path, tree)
    if len(entries) != 1 or entries[0][0] not in (b"100644", b"100755"):
        raise SnapshotError("File missing or unsupported on requested fixed side")
    blob = entries[0][2].decode()
    data = git.run(["cat-file", "blob", blob], maximum=maximum)
    text = data.decode("utf-8")
    if "\0" in text:
        raise SnapshotError("Binary source is not readable text")
    return {
        "tree": tree,
        "blob": blob,
        "sha256": hashlib.sha256(data).hexdigest(),
        "lines": text.splitlines(),
    }
