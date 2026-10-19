"""Complete original-workspace observations and deterministic per-path diff3."""

import difflib
import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path

from agent_service.delivery_fs import (
    DeliveryConflict,
    content,
    state_at,
    workspace_identity,
)
from agent_service.repair_versions import read_regular, source_manifest
from execution.code_snapshot import metadata, prepare, workspace_files
from execution.git_tools import Git


def observe(root, destination, limits, expected_identity):
    """Reuse trusted snapshot capture, keeping B and its once-per-Run row intact."""
    if workspace_identity(root) != expected_identity:
        raise DeliveryConflict("Workspace identity changed")
    git = Git(root)
    before, _ = metadata(git, limits)
    document = prepare(root, destination, {"kind": "workspace"}, limits)
    after, tracked = metadata(git, limits)
    if before != after or workspace_identity(root) != expected_identity:
        raise DeliveryConflict("Git state changed during observation")
    manifest = source_manifest(document["manifest"])
    states = {}
    for name, expected in manifest.items():
        from agent_service.delivery_fs import source_parts

        parts = source_parts(name)
        fd = os.open(root, os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts[:-1]:
                child = os.open(part, os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child
            actual = state_at(fd, parts[-1], limits["file_bytes"])
        finally:
            os.close(fd)
        if content(actual) != expected:
            raise DeliveryConflict("Source changed after snapshot")
        states[name] = actual
    current = {
        k: v
        for k, v in workspace_files(git, tracked, limits).items()
        if not v.get("deleted")
    }
    if current != manifest or metadata(git, limits)[0] != after:
        raise DeliveryConflict("Incomplete or unstable observation")
    for directory, _, files in os.walk(destination, topdown=False):
        for name in files:
            fd = os.open(Path(directory) / name, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        from agent_service.delivery_fs import fsync_dir

        fsync_dir(directory)
    document.update(
        identity=expected_identity,
        source_manifest=manifest,
        git_state=after,
        states=states,
        inclusion="tracked worktree bytes plus nonignored untracked; raw modes; no symlinks",
    )
    return document


def current_context(root, limits, expected_identity, private=None, known=None):
    if workspace_identity(root) != expected_identity:
        raise DeliveryConflict("Workspace identity changed")
    if private is not None:
        from agent_service.delivery_fs import validate_private

        validate_private(root, private)
    git = Git(root)
    before, paths = metadata(git, limits)
    files = {
        k: v
        for k, v in workspace_files(git, paths, limits).items()
        if not v.get("deleted")
    }
    # The sealed version's explicit paths remain in scope even when a delivered
    # .gitignore now hides them. Additional nonignored paths are still detected.
    for name in (known or {}).keys() - files.keys():
        try:
            raw, mode = read_regular(root, name, limits["file_bytes"])
        except FileNotFoundError:
            continue
        files[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
            "mode": mode,
        }
    from agent_service.delivery_fs import source_parts

    for name, expected in files.items():
        parts = source_parts(name)
        fd = os.open(root, os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts[:-1]:
                child = os.open(part, os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child
            actual = state_at(fd, parts[-1], limits["file_bytes"])
            if (
                content(actual) != expected
                or actual["device"] != expected_identity["root"]["device"]
            ):
                raise DeliveryConflict("Unsupported or changed final source metadata")
        finally:
            os.close(fd)
    if (
        before != metadata(git, limits)[0]
        or workspace_identity(root) != expected_identity
    ):
        raise DeliveryConflict("Workspace changed during verification")
    return files, before


def git_algorithm():
    executable = Path("/usr/bin/git")
    return dict(
        name="git-merge-file--diff3",
        version=subprocess.check_output(
            [str(executable), "--version"], env={"PATH": "/usr/bin:/bin"}, text=True
        ).strip(),
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
    )


def choose(base, agent, user):
    if agent == user:
        return agent
    if base == agent:
        return user
    if base == user:
        return agent
    raise DeliveryConflict("both_changed")


def merge_versions(
    base_root,
    base,
    agent_root,
    agent,
    user_root,
    user,
    destination,
    maximum,
    algorithm,
    budget=None,
):
    if git_algorithm() != algorithm:
        raise DeliveryConflict("Merge executable identity changed")
    destination.mkdir(parents=True)
    result, conflicts = {}, []
    total = 0
    budget = budget or {
        "snapshot_bytes": max(1, len(set(base) | set(agent) | set(user))) * maximum,
        "file_count": 50000,
        "deadline": float("inf"),
    }
    names = sorted(set(base) | set(agent) | set(user))
    # Detect all file/directory collisions before materializing any version.
    paths = set(agent) | set(user)
    for name in paths:
        if any(
            "/".join(name.split("/")[:i]) in paths
            for i in range(1, len(name.split("/")))
        ):
            # A clean one-sided file -> directory change is also unsupported.
            conflicts.append(dict(path=name, type="file_directory"))
    if conflicts:
        return result, conflicts
    for name in names:
        if time.time() >= budget["deadline"]:
            from agent_service.code_artifacts import CodeFailure

            raise CodeFailure("REPAIR_BUDGET_EXHAUSTED", "Merge deadline expired")
        b, a, u = base.get(name), agent.get(name), user.get(name)
        try:
            if a == u or a == b or u == b:
                selected = choose(b, a, u)
                if selected is None:
                    continue
                source = agent_root if selected == a else user_root
                data, _ = read_regular(source, name, maximum)
                mode = selected["mode"]
            else:
                if b is None:
                    raise DeliveryConflict("add_add")
                if a is None or u is None:
                    raise DeliveryConflict("delete_modify")
                try:
                    mode = choose(b["mode"], a["mode"], u["mode"])
                except DeliveryConflict:
                    raise DeliveryConflict("mode") from None
                blobs = [
                    read_regular(root, name, maximum)[0]
                    for root in (base_root, agent_root, user_root)
                ]
                try:
                    data = choose(*blobs)
                except DeliveryConflict:
                    if any(b"\0" in x for x in blobs):
                        raise DeliveryConflict("binary") from None
                    try:
                        for x in blobs:
                            x.decode("utf-8")
                    except UnicodeError:
                        raise DeliveryConflict("binary") from None
                    with tempfile.TemporaryDirectory() as temp:
                        paths = [Path(temp) / n for n in ("base", "agent", "user")]
                        for path, blob in zip(paths, blobs):
                            path.write_bytes(blob)
                        proc = subprocess.run(
                            [
                                "/usr/bin/git",
                                "merge-file",
                                "--diff3",
                                "-p",
                                "--",
                                str(paths[1]),
                                str(paths[0]),
                                str(paths[2]),
                            ],
                            env={
                                "PATH": "/usr/bin:/bin",
                                "HOME": temp,
                                "GIT_CONFIG_NOSYSTEM": "1",
                                "GIT_CONFIG_GLOBAL": "/dev/null",
                            },
                            cwd=temp,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=min(
                                30, max(0.01, budget["deadline"] - time.time())
                            ),
                        )
                        if proc.returncode != 0:
                            raise DeliveryConflict("text")
                        data = proc.stdout
            if len(data) > maximum:
                raise DeliveryConflict("size")
            total += len(data)
            if total > budget["snapshot_bytes"] or len(result) >= budget["file_count"]:
                from agent_service.code_artifacts import CodeFailure

                raise CodeFailure(
                    "ARTIFACT_LIMIT", "Merged output exceeds pre-admitted limits"
                )
            output = destination / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            output.chmod(mode)
            result[name] = dict(
                sha256=hashlib.sha256(data).hexdigest(), size=len(data), mode=mode
            )
        except DeliveryConflict as exc:
            conflicts.append(dict(path=name, type=str(exc), base=b, agent=a, user=u))
    return result, conflicts


def protection(agent_root, agent, merged_root, merged, maximum):
    """Provenance ranges are in installed M; absent files/modes are protected too."""
    records = {}
    for name in set(agent) | set(merged):
        a, m = agent.get(name), merged.get(name)
        if a == m:
            continue
        entry = dict(installed=m, agent=a)
        if a and m and a["sha256"] != m["sha256"]:
            before, _ = read_regular(agent_root, name, maximum)
            after, _ = read_regular(merged_root, name, maximum)
            try:
                if b"\0" in before + after:
                    raise UnicodeError()
                old, new = (
                    before.decode().splitlines(keepends=True),
                    after.decode().splitlines(keepends=True),
                )
                entry["lines"] = new
                entry["ranges"] = [
                    [j, k]
                    for tag, _, _, j, k in difflib.SequenceMatcher(
                        None, old, new, autojunk=False
                    ).get_opcodes()
                    if tag != "equal"
                ]
            except UnicodeError:
                entry["whole_file"] = True
        records[name] = entry
    return records


def check_protection(
    records, agent_root, agent, user, previous_user, maximum, user_root=None
):
    for name, record in records.items():
        user_changed = user.get(name) != previous_user.get(name)
        if user_changed and (not user.get(name) or not previous_user.get(name)):
            continue
        live_ranges = active_user_ranges(
            record, name, user_root, user, previous_user, maximum
        )
        installed, current = record["installed"], agent.get(name)
        if installed == current:
            continue
        if (
            installed is None
            or current is None
            or (record.get("whole_file") and not user_changed)
            or record["agent"] is None
        ):
            raise DeliveryConflict(
                "Candidate changes integrated user file state: " + name
            )
        if (
            installed["mode"] != current["mode"]
            and (
                record.get("protect_mode", False)
                or installed["mode"] != record["agent"]["mode"]
            )
            and (user.get(name) or {}).get("mode")
            == (previous_user.get(name) or {}).get("mode")
        ):
            raise DeliveryConflict("Candidate changes integrated user mode: " + name)
        if "lines" not in record:
            continue
        raw, _ = read_regular(agent_root, name, maximum)
        try:
            lines = raw.decode().splitlines(keepends=True)
        except UnicodeError:
            raise DeliveryConflict(
                "Candidate changes integrated user text: " + name
            ) from None
        for tag, start, end, _, _ in difflib.SequenceMatcher(
            None, record["lines"], lines, autojunk=False
        ).get_opcodes():
            if tag == "equal":
                continue
            if any((start <= right and end >= left) for left, right in live_ranges):
                raise DeliveryConflict(
                    "Candidate overlaps integrated user text: " + name
                )


def _map_range(before, after, left, right):
    for start, target, length in difflib.SequenceMatcher(
        None, before, after, autojunk=False
    ).get_matching_blocks():
        if start <= left <= right <= start + length:
            return [target + left - start, target + right - start]
    return None


def _overlaps(left, right, start, end):
    # Boundary insertions are conservatively treated as overlapping.
    return start <= right and end >= left


def active_user_ranges(record, name, user_root, user, previous_user, maximum):
    ranges = record.get("ranges", [])
    if user.get(name) == previous_user.get(name):
        return ranges
    if not user.get(name) or not previous_user.get(name):
        return []  # Explicit user file creation/removal supersedes prior state.
    if user_root is None or "user_lines" not in record:
        return []
    try:
        raw, _ = read_regular(user_root, name, maximum)
        lines = raw.decode().splitlines(keepends=True)
    except UnicodeError:
        return ranges  # Cannot localize the user's edit: keep the protection.
    changed = [
        (i, j)
        for tag, i, j, _, _ in difflib.SequenceMatcher(
            None, record["user_lines"], lines, autojunk=False
        ).get_opcodes()
        if tag != "equal"
    ]
    retained = []
    for protected, user_range in zip(ranges, record["user_ranges"]):
        if user_range is None or not any(
            _overlaps(*user_range, i, j) for i, j in changed
        ):
            retained.append(protected)
    return retained


def update_protection(
    previous, fresh, merged_root, merged, user_root, user, previous_user, maximum
):
    """Carry/rebase prior provenance; only actual user overlap supersedes it."""
    result = {}
    for name in set(previous) | set(fresh):
        old, new = previous.get(name), fresh.get(name)
        current = merged.get(name)
        entry = dict(new or old)
        ranges = list(new.get("ranges", [])) if new else []
        if old and bool(user.get(name)) == bool(previous_user.get(name)):
            if current and old["installed"]:
                old_mode = (previous_user.get(name) or {}).get("mode")
                user_mode = (user.get(name) or {}).get("mode")
                if old.get("protect_mode", False) and old_mode == user_mode:
                    entry["protect_mode"] = True
                if old.get("whole_file") and user.get(name) == previous_user.get(name):
                    entry["whole_file"] = True
                if "lines" in old:
                    raw, _ = read_regular(merged_root, name, maximum)
                    try:
                        final_lines = raw.decode().splitlines(keepends=True)
                        for left, right in active_user_ranges(
                            old, name, user_root, user, previous_user, maximum
                        ):
                            translated = _map_range(
                                old["lines"], final_lines, left, right
                            )
                            if translated is None:
                                entry["whole_file"] = True
                            else:
                                ranges.append(translated)
                    except UnicodeError:
                        entry["whole_file"] = True
            elif old["installed"] is None and current is None:
                entry = old
        elif old and new is None:
            continue
        entry["installed"] = current
        if new:
            entry["protect_mode"] = entry.get("protect_mode", False) or bool(
                new["agent"] and current and new["agent"]["mode"] != current["mode"]
            )
        if current:
            try:
                raw, _ = read_regular(merged_root, name, maximum)
                final_lines = raw.decode().splitlines(keepends=True)
                user_raw, _ = read_regular(user_root, name, maximum)
                user_lines = user_raw.decode().splitlines(keepends=True)
                if ranges:
                    entry["lines"] = final_lines
                    entry["ranges"] = [
                        list(r) for r in sorted(set(tuple(r) for r in ranges))
                    ]
                    entry["user_lines"] = user_lines
                    entry["user_ranges"] = [
                        _map_range(final_lines, user_lines, *r) for r in entry["ranges"]
                    ]
                else:
                    for key in ("lines", "ranges", "user_lines", "user_ranges"):
                        entry.pop(key, None)
            except UnicodeError:
                entry["whole_file"] = True
        result[name] = entry
    return result
