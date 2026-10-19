"""Linux host delivery primitives. Never executes project code or writes Git metadata.

Every namespace mutation uses renameat2 and preserves the displaced inode. The
caller journals an intent before calling mutate(), then journals inspect() after
fsync. Filesystem state, rather than a missing DB acknowledgement, decides replay.
"""

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import stat
import time
from contextlib import contextmanager
from pathlib import Path

from execution.code_snapshot import digest, safe_parts


class DeliveryConflict(ValueError):
    pass


def identity(info):
    return dict(device=info.st_dev, inode=info.st_ino)


def directory_identity(path):
    if str(Path(path).resolve(strict=True)) != str(path):
        raise DeliveryConflict("Directory path was replaced or follows a symlink")
    fd = os.open(path, os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        return identity(os.fstat(fd))
    finally:
        os.close(fd)


def workspace_identity(root):
    return dict(root=directory_identity(root), git=directory_identity(root / ".git"))


def validate_private(root, private):
    if os.name != "posix" or not Path("/proc/self/mountinfo").exists():
        raise ValueError("Automatic delivery requires a local Linux filesystem")
    root_id, private_id = directory_identity(root), directory_identity(private)
    info = private.stat()
    if (
        private.is_relative_to(root)
        or root.is_relative_to(private)
        or root_id["device"] != private_id["device"]
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError(
            "Delivery private directory must be outside the workspace, service-owned, mode 0700 and on the same filesystem"
        )

    # Reject network filesystems and nested mounts, including bind mounts whose
    # st_dev happens to match. mountinfo escapes whitespace and backslashes.
    def unescape(value):
        for a, b in [("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")]:
            value = value.replace(a, b)
        return value

    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        mount = Path(unescape(fields[4]))
        filesystem = fields[fields.index("-") + 1]
        mounts.append((mount, filesystem))
        if mount != root and mount.is_relative_to(root):
            raise ValueError("Nested workspace mounts are unsupported")
    for path in (root, private):
        _, filesystem = max(
            ((m, f) for m, f in mounts if path.is_relative_to(m)),
            key=lambda x: len(x[0].parts),
        )
        if filesystem not in (
            "ext2",
            "ext3",
            "ext4",
            "xfs",
            "btrfs",
            "tmpfs",
            "overlay",
        ):
            raise ValueError("Unsupported delivery filesystem: " + filesystem)


def source_parts(path):
    parts = safe_parts(path)
    if any(p.lower() == ".git" for p in parts):
        raise DeliveryConflict("Git metadata is outside delivery authority")
    return parts


@contextmanager
def private_lock(private, root):
    key = hashlib.sha256(str(root).encode()).hexdigest()
    fd = os.open(
        private / ("workspace-" + key + ".lock"),
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def fsync_dir(path):
    fd = os.open(path, os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_json(path, value):
    temporary = path.with_suffix(".pending")
    fd = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(fd, "wb") as out:
        out.write(json.dumps(value, sort_keys=True).encode())
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)
    fsync_dir(path.parent)


def state_at(fd, name, maximum):
    try:
        handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    except FileNotFoundError:
        return None
    with os.fdopen(handle, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum
            or before.st_mode & 0o7000
            or os.listxattr(stream.fileno())
        ):
            raise DeliveryConflict("Unsupported file metadata, hardlink, type or size")
        data = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise DeliveryConflict("File changed while being read")
        if len(data) > maximum:
            raise DeliveryConflict("File size exceeded")
        return dict(
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
            mode=stat.S_IMODE(before.st_mode),
            **identity(before),
            uid=before.st_uid,
            gid=before.st_gid,
        )


def content(state):
    return None if state is None else {k: state[k] for k in ("sha256", "size", "mode")}


def same(actual, expected):
    return actual == expected


_libc = ctypes.CDLL(None, use_errno=True)


def rename(source_fd, source, target_fd, target, flags):
    result = _libc.renameat2(
        ctypes.c_int(source_fd),
        ctypes.c_char_p(os.fsencode(source)),
        ctypes.c_int(target_fd),
        ctypes.c_char_p(os.fsencode(target)),
        ctypes.c_uint(flags),
    )
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


class FileTransaction:
    def __init__(self, root, private, plan, maximum):
        self.root, self.private, self.plan, self.maximum = (
            Path(root),
            Path(private),
            plan,
            maximum,
        )
        self.directory = self.private / plan["id"]
        self.root_fd = os.open(self.root, os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            self.private_fd = os.open(self.directory, os.O_DIRECTORY | os.O_NOFOLLOW)
        except BaseException:
            os.close(self.root_fd)
            raise
        self.deadline_timestamp = None
        try:
            self.assert_root()
        except BaseException:
            self.close()
            raise

    def close(self):
        os.close(self.root_fd)
        os.close(self.private_fd)

    def assert_root(self):
        if (
            identity(os.fstat(self.root_fd)) != self.plan["identity"]["root"]
            or workspace_identity(self.root) != self.plan["identity"]
        ):
            raise DeliveryConflict("Workspace or Git directory identity changed")

    @contextmanager
    def parent(self, path):
        self.assert_root()
        parts = source_parts(path)
        fd = os.dup(self.root_fd)
        try:
            prefix = []
            for part in parts[:-1]:
                prefix.append(part)
                child = os.open(part, os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child
                if identity(os.fstat(fd)) != self.plan["parents"]["/".join(prefix)]:
                    raise DeliveryConflict(
                        "Parent directory identity changed: " + "/".join(prefix)
                    )
            yield fd, parts[-1]
            self.assert_root()
            # Rewalk after mutation. An open directory may have been detached.
            check = os.dup(self.root_fd)
            try:
                for n, part in enumerate(parts[:-1]):
                    child = os.open(part, os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=check)
                    os.close(check)
                    check = child
                    if (
                        identity(os.fstat(check))
                        != self.plan["parents"]["/".join(parts[: n + 1])]
                    ):
                        raise DeliveryConflict("Parent was replaced during operation")
            finally:
                os.close(check)
        finally:
            os.close(fd)

    def inspect(self, operation):
        try:
            with self.parent(operation["path"]) as (parent, name):
                target = state_at(parent, name, self.maximum)
        except FileNotFoundError:
            target = None
        staged = state_at(self.private_fd, operation["slot"], self.maximum)
        undo = state_at(self.private_fd, operation["undo_slot"], self.maximum)
        return dict(target=target, slot=staged, undo=undo)

    @staticmethod
    def classify(operation, actual):
        old, new = operation["before"], operation["staged"]
        target, slot, undo = actual["target"], actual["slot"], actual["undo"]
        if undo is not None:
            if (
                same(target, old)
                and same(undo, new)
                and (slot is None or same(slot, old))
            ):
                return "rolled_back"
            return "unknown"
        if same(target, old) and same(slot, new):
            return "before"
        if same(target, new) and same(slot, old):
            return "after"
        return "unknown"

    def mutate(self, operation, rollback=False):
        actual = self.inspect(operation)
        phase = self.classify(operation, actual)
        if (not rollback and phase == "after") or (
            rollback and phase in ("before", "rolled_back")
        ):
            return actual
        if phase != ("after" if rollback else "before"):
            raise DeliveryConflict(
                "File state is not explained by the journal: " + operation["path"]
            )
        with self.parent(operation["path"]) as (parent, name):
            # Repeat immediately before mutation; rename still preserves any writer
            # that races this check. Unexpected displaced bytes remain in private.
            if not same(state_at(parent, name, self.maximum), actual["target"]):
                raise DeliveryConflict("Concurrent file change: " + operation["path"])
            old, new = operation["before"], operation["staged"]
            if (
                not rollback
                and self.deadline_timestamp is not None
                and time.time() >= self.deadline_timestamp
            ):
                raise DeliveryConflict("Forward write deadline expired")
            if rollback:
                if old is None:
                    rename(parent, name, self.private_fd, operation["undo_slot"], 1)
                elif new is None:
                    rename(self.private_fd, operation["slot"], parent, name, 1)
                else:
                    rename(self.private_fd, operation["slot"], parent, name, 2)
                    # Keep the displaced delivered inode in a separate slot. A
                    # crash between these renames is recognized from swapped slots.
                    rename(
                        self.private_fd,
                        operation["slot"],
                        self.private_fd,
                        operation["undo_slot"],
                        1,
                    )
            elif old is None:
                rename(self.private_fd, operation["slot"], parent, name, 1)
            elif new is None:
                rename(parent, name, self.private_fd, operation["slot"], 1)
            else:
                rename(self.private_fd, operation["slot"], parent, name, 2)
            for slot in (operation["slot"], operation["undo_slot"]):
                try:
                    handle = os.open(
                        slot,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                        dir_fd=self.private_fd,
                    )
                except FileNotFoundError:
                    continue
                try:
                    if not stat.S_ISREG(os.fstat(handle).st_mode):
                        raise DeliveryConflict("Displaced inode is not a regular file")
                    os.fsync(handle)
                finally:
                    os.close(handle)
            os.fsync(parent)
            os.fsync(self.private_fd)
        found = self.inspect(operation)
        if rollback:
            valid = same(found["target"], old) and (
                same(found["undo"], new) if new else found["slot"] is None
            )
        else:
            valid = same(found["target"], new) and same(found["slot"], old)
        if not valid:
            raise DeliveryConflict(
                "Atomic operation retained unexpected concurrent content: "
                + operation["path"]
            )
        return found

    def create_directory(self, operation, rollback=False):
        path = operation["path"]
        with self.parent(path) as (parent, name):
            try:
                actual = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                actual = None
            expected = operation["identity"]
            if actual is not None and identity(actual) != expected:
                raise DeliveryConflict("Created directory was replaced: " + path)
            if rollback:
                if actual is not None:
                    # rmdir refuses nonempty directories; no user files removed.
                    os.rmdir(name, dir_fd=parent)
            elif actual is None:
                rename(self.private_fd, operation["slot"], parent, name, 1)
            os.fsync(parent)
            os.fsync(self.private_fd)


def stage_plan(
    root, private, plan_id, observation, target_root, target_manifest, maximum
):
    """Prepare everything without mutating the original workspace."""
    validate_private(root, private)
    directory = private / plan_id
    directory.mkdir(mode=0o700)
    plan = dict(
        id=plan_id,
        identity=observation["identity"],
        parents={},
        directories=[],
        files=[],
        observation_digest=observation["content_digest"],
        final_digest=digest(target_manifest),
    )
    plan["before_manifest"] = observation["source_manifest"]
    plan["after_manifest"] = target_manifest
    plan["git_state"] = observation["git_state"]
    source_states = observation["states"]
    for name in sorted(set(plan["before_manifest"]) | set(target_manifest)):
        if plan["before_manifest"].get(name) == target_manifest.get(name):
            continue
        parts = source_parts(name)
        # Directory/file transitions are deliberately conflicts, including ignored
        # or empty existing directories not represented in the source inventory.
        for i in range(1, len(parts)):
            prefix = "/".join(parts[:i])
            if prefix in plan["parents"]:
                continue
            path = root / prefix
            if path.exists() or path.is_symlink():
                parent_id = directory_identity(path)
                if parent_id["device"] != plan["identity"]["root"]["device"]:
                    raise DeliveryConflict("Cross-filesystem parent")
            else:
                slot = "directory-" + str(len(plan["directories"]))
                staged = directory / slot
                staged.mkdir(mode=0o755)
                owner = root.stat()
                os.chown(staged, owner.st_uid, owner.st_gid)
                parent_id = identity(staged.stat())
                plan["directories"].append(
                    dict(path=prefix, slot=slot, identity=parent_id)
                )
                fsync_dir(staged)
            plan["parents"][prefix] = parent_id
        old = source_states.get(name)
        new = target_manifest.get(name)
        slot = "file-" + str(len(plan["files"]))
        staged_state = None
        if new:
            from agent_service.repair_versions import read_regular

            data, mode = read_regular(target_root, name, maximum)
            if (
                dict(sha256=hashlib.sha256(data).hexdigest(), size=len(data), mode=mode)
                != new
            ):
                raise DeliveryConflict("Final revision changed during staging")
            fd = os.open(
                directory / slot,
                os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                owner = old or dict(uid=root.stat().st_uid, gid=root.stat().st_gid)
                os.fchown(stream.fileno(), owner["uid"], owner["gid"])
                os.fchmod(stream.fileno(), mode)
                os.fsync(stream.fileno())
            dfd = os.open(directory, os.O_DIRECTORY)
            try:
                staged_state = state_at(dfd, slot, maximum)
            finally:
                os.close(dfd)
        plan["files"].append(
            dict(
                path=name,
                before=old,
                after=new,
                staged=staged_state,
                slot=slot,
                undo_slot=slot + "-undo",
            )
        )
    # Reserve enough space for actual displaced originals. New content has already
    # been fsynced; originals move atomically and are never copied then deleted.
    free = os.statvfs(private)
    if free.f_bavail * free.f_frsize < sum(
        x["before"]["size"] for x in plan["files"] if x["before"]
    ):
        raise DeliveryConflict("Insufficient backup space")
    durable_json(directory / "plan.json", plan)
    fsync_dir(private)
    return plan


def atomic_probe(private):
    """Fail startup before authorization if the filesystem lacks renameat2 flags."""
    import tempfile

    with tempfile.TemporaryDirectory(dir=private) as temporary:
        path = Path(temporary)
        (path / "a").write_bytes(b"a")
        (path / "b").write_bytes(b"b")
        fd = os.open(path, os.O_DIRECTORY)
        try:
            rename(fd, "a", fd, "b", 2)
            try:
                rename(fd, "a", fd, "b", 1)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
            else:
                raise ValueError("Filesystem ignored no-replace")
            if (path / "b").read_bytes() != b"a":
                raise ValueError("Filesystem ignored atomic exchange")
            os.fsync(fd)
        finally:
            os.close(fd)
