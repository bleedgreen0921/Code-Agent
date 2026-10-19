"""Real diff3 and filesystem fault boundaries, without Docker or DB substitutes."""

import hashlib
import os
from uuid import uuid4

import pytest

from agent_service.delivery_fs import (
    DeliveryConflict,
    FileTransaction,
    private_lock,
    stage_plan,
    workspace_identity,
)
from agent_service.delivery_merge import (
    check_protection,
    git_algorithm,
    merge_versions,
    observe,
    protection,
)
from agent_service.repair_registry import CodeRepairSettings
from tests.test_code_analysis import git


def write_version(root, files):
    root.mkdir()
    manifest = {}
    for name, value in files.items():
        data, mode = (value, 0o644) if isinstance(value, bytes) else value
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(mode)
        manifest[name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "mode": mode,
        }
    return manifest


@pytest.mark.parametrize(
    "base,agent,user,expected,conflict",
    [
        ({"f": b"b\n"}, {"f": b"a\n"}, {"f": b"b\n"}, {"f": b"a\n"}, None),
        ({"f": b"b\n"}, {"f": b"b\n"}, {"f": b"u\n"}, {"f": b"u\n"}, None),
        ({"f": b"b\n"}, {"f": b"a\n"}, {"f": b"a\n"}, {"f": b"a\n"}, None),
        ({}, {"f": b"a\n"}, {"f": b"u\n"}, None, "add_add"),
        ({"f": b"b\n"}, {}, {"f": b"u\n"}, None, "delete_modify"),
        ({"f": b"b\n"}, {"f": b"a\n"}, {}, None, "delete_modify"),
        ({"f": b"b\n"}, {"f": b"a\n"}, {"f": b"u\n"}, None, "text"),
        ({"f": b"\0b"}, {"f": b"\0a"}, {"f": b"\0u"}, None, "binary"),
        ({"f": b"\0b"}, {"f": b"\0a"}, {"f": b"\0b"}, {"f": b"\0a"}, None),
        ({"f": b"b\n"}, {"f": (b"b\n", 0o600)}, {"f": (b"b\n", 0o755)}, None, "mode"),
        (
            {"f": b"b\n"},
            {"f": b"a\n"},
            {"f": (b"b\n", 0o755)},
            {"f": (b"a\n", 0o755)},
            None,
        ),
        ({"f": b"b\n"}, {"f": b"b\n"}, {"f/child": b"u\n"}, None, "file_directory"),
        ({"gone": b"old"}, {"gone": b"old"}, {}, {}, None),
        ({}, {}, {"new": b"user"}, {"new": b"user"}, None),
        ({}, {"new": b"agent"}, {}, {"new": b"agent"}, None),
        (
            {"f": b"one\n2\n3\n4\n5\n6\ntail\n"},
            {"f": b"agent\n2\n3\n4\n5\n6\ntail\n"},
            {"f": b"one\n2\n3\n4\n5\n6\nuser\n"},
            {"f": b"agent\n2\n3\n4\n5\n6\nuser\n"},
            None,
        ),
    ],
)
def test_merge_matrix(tmp_path, base, agent, user, expected, conflict):
    b, a, u = [
        write_version(tmp_path / n, f)
        for n, f in zip(("b", "a", "u"), (base, agent, user))
    ]
    manifest, conflicts = merge_versions(
        tmp_path / "b",
        b,
        tmp_path / "a",
        a,
        tmp_path / "u",
        u,
        tmp_path / "m",
        1024,
        git_algorithm(),
    )
    if conflict:
        assert any(c["type"] == conflict for c in conflicts)
    else:
        assert not conflicts
        expected_manifest = write_version(tmp_path / "expected", expected)
        assert manifest == expected_manifest
        for name, value in expected.items():
            assert (tmp_path / "m" / name).read_bytes() == (
                value if isinstance(value, bytes) else value[0]
            )


def transaction_fixture(tmp_path):
    root, private, target = (
        tmp_path / "source",
        tmp_path / "private",
        tmp_path / "target",
    )
    root.mkdir()
    private.mkdir(mode=0o700)
    git(root, "init", "-q")
    (root / "replace").write_bytes(b"old")
    (root / "delete").write_bytes(b"delete old")
    git(root, "add", ".")
    limits = CodeRepairSettings().model_dump()
    observation = observe(
        root, tmp_path / "observation", limits, workspace_identity(root)
    )
    manifest = write_version(
        target, {"replace": (b"new", 0o755), "new": b"added", "d/nested": b"nested"}
    )
    plan = stage_plan(
        root, private, uuid4().hex, observation, target, manifest, limits["file_bytes"]
    )
    tx = FileTransaction(root, private, plan, limits["file_bytes"])
    return root, private, plan, tx


def test_atomic_add_exchange_delete_and_safe_rollback(tmp_path):
    root, private, plan, tx = transaction_fixture(tmp_path)
    try:
        for directory in plan["directories"]:
            tx.create_directory(directory)
        for operation in plan["files"]:
            old = tx.inspect(operation)
            assert tx.classify(operation, old) == "before"
            after = tx.mutate(operation)
            assert tx.classify(operation, after) == "after"
            assert after["slot"] == operation["before"]
            assert after["target"] == operation["staged"]
            # Loss of DB confirmation cannot cause a second exchange.
            assert tx.mutate(operation) == after
        for operation in reversed(plan["files"]):
            tx.mutate(operation, True)
            assert tx.classify(operation, tx.inspect(operation)) in (
                "before",
                "rolled_back",
            )
            tx.mutate(operation, True)
        for directory in reversed(plan["directories"]):
            tx.create_directory(directory, True)
        assert (root / "replace").read_bytes() == b"old"
        assert (root / "delete").read_bytes() == b"delete old"
        assert not (root / "new").exists() and not (root / "d").exists()
    finally:
        tx.close()


@pytest.mark.parametrize(
    "boundary", ["before", "after", "exchange_race", "parent", "root"]
)
def test_external_writer_preserved(tmp_path, monkeypatch, boundary):
    from agent_service import delivery_fs

    root, private, plan, tx = transaction_fixture(tmp_path)
    operation = next(
        f
        for f in plan["files"]
        if f["path"] == ("d/nested" if boundary == "parent" else "replace")
    )
    try:
        for directory in plan["directories"]:
            tx.create_directory(directory)
        if boundary == "before":
            (root / "replace").write_bytes(b"user")
            with pytest.raises(DeliveryConflict):
                tx.mutate(operation)
            assert (root / "replace").read_bytes() == b"user"
        elif boundary == "after":
            tx.mutate(operation)
            (root / "replace").write_bytes(b"later user")
            with pytest.raises(DeliveryConflict):
                tx.mutate(operation, True)
            assert (root / "replace").read_bytes() == b"later user"
        elif boundary == "exchange_race":
            original = delivery_fs.rename

            def race(*args):
                (root / "replace").write_bytes(b"concurrent user")
                return original(*args)

            monkeypatch.setattr(delivery_fs, "rename", race)
            with pytest.raises(DeliveryConflict):
                tx.mutate(operation)
            assert (
                private / plan["id"] / operation["slot"]
            ).read_bytes() == b"concurrent user"
            assert (root / "replace").read_bytes() == b"new"
            with pytest.raises(DeliveryConflict):
                tx.mutate(operation, True)
        elif boundary == "parent":
            outside = tmp_path / "outside"
            outside.mkdir()
            (root / "d").rename(root / "detached")
            (root / "d").symlink_to(outside, target_is_directory=True)
            with pytest.raises(OSError):
                tx.mutate(operation)
            assert not list(outside.iterdir())
        else:
            root.rename(tmp_path / "detached")
            root.mkdir()
            with pytest.raises(DeliveryConflict):
                tx.mutate(operation)
    finally:
        tx.close()


def test_lock_and_directory_rollback_do_not_remove_user_files(tmp_path):
    root, private, plan, tx = transaction_fixture(tmp_path)
    try:
        with private_lock(private, root):
            with pytest.raises(BlockingIOError):
                with private_lock(private, root):
                    pytest.fail("Second lock holder admitted")
        directory = plan["directories"][0]
        tx.create_directory(directory)
        (root / directory["path"] / "user").write_text("keep")
        with pytest.raises(OSError):
            tx.create_directory(directory, True)
        assert (root / directory["path"] / "user").read_text() == "keep"
    finally:
        tx.close()


@pytest.mark.parametrize("metadata", ["hardlink", "xattr", "symlink"])
def test_unsupported_original_metadata_is_rejected(tmp_path, metadata):
    root = tmp_path / "root"
    root.mkdir()
    git(root, "init", "-q")
    (root / "f").write_bytes(b"content")
    if metadata == "hardlink":
        os.link(root / "f", root / "link")
    elif metadata == "xattr":
        os.setxattr(root / "f", "user.test", b"keep")
    else:
        (root / "link").symlink_to(root / "f")
    with pytest.raises((ValueError, OSError)):
        observe(
            root,
            tmp_path / "snapshot",
            CodeRepairSettings().model_dump(),
            workspace_identity(root),
        )
    assert (root / "f").read_bytes() == b"content"


def test_user_provenance_allows_unrelated_model_changes(tmp_path):
    old = b"agent\n2\n3\n4\n5\n6\ntail\n"
    new = old.replace(b"tail", b"user")
    a = write_version(tmp_path / "a", {"f": old})
    m = write_version(tmp_path / "m", {"f": new})
    protected = protection(tmp_path / "a", a, tmp_path / "m", m, 1024)
    good = write_version(tmp_path / "good", {"f": new.replace(b"agent", b"fixed")})
    check_protection(protected, tmp_path / "good", good, m, m, 1024)
    bad = write_version(tmp_path / "bad", {"f": old})
    with pytest.raises(DeliveryConflict):
        check_protection(protected, tmp_path / "bad", bad, m, m, 1024)
    # The user's own subsequent revert supersedes the old provenance.
    check_protection(protected, tmp_path / "bad", bad, a, m, 1024)


def test_provenance_survives_second_install_and_unrelated_user_edit(tmp_path):
    from agent_service.delivery_merge import update_protection

    base = b"agent\n2\n3\n4\n5\n6\nuser old\n8\n9\n10\n11\n12\ntail\n"
    original = base.replace(b"user old", b"original")
    a = write_version(tmp_path / "a1", {"f": original})
    u = write_version(tmp_path / "u1", {"f": base})
    m = write_version(tmp_path / "m1", {"f": base})
    records = update_protection(
        {},
        protection(tmp_path / "a1", a, tmp_path / "m1", m, 1024),
        tmp_path / "m1",
        m,
        tmp_path / "u1",
        u,
        a,
        1024,
    )
    user_next = base.replace(b"tail", b"user new")
    u2 = write_version(tmp_path / "u2", {"f": user_next})
    # Changing another user region does not authorize reverting the first region.
    stale = write_version(tmp_path / "stale", {"f": original})
    with pytest.raises(DeliveryConflict):
        check_protection(
            records, tmp_path / "stale", stale, u2, u, 1024, tmp_path / "u2"
        )
    # An unrelated model edit shifts the old protected interval during install.
    a2_data = base.replace(b"agent", b"agent\nextra")
    a2 = write_version(tmp_path / "a2", {"f": a2_data})
    m2_data = a2_data.replace(b"tail", b"user new")
    m2 = write_version(tmp_path / "m2", {"f": m2_data})
    records2 = update_protection(
        records,
        protection(tmp_path / "a2", a2, tmp_path / "m2", m2, 1024),
        tmp_path / "m2",
        m2,
        tmp_path / "u2",
        u2,
        u,
        1024,
    )
    forbidden = write_version(
        tmp_path / "forbidden", {"f": m2_data.replace(b"user old", b"original")}
    )
    with pytest.raises(DeliveryConflict):
        check_protection(
            records2, tmp_path / "forbidden", forbidden, u2, u2, 1024, tmp_path / "u2"
        )
    # The user's own revert of that exact interval does supersede it.
    u3 = write_version(
        tmp_path / "u3", {"f": user_next.replace(b"user old", b"original")}
    )
    check_protection(
        records2, tmp_path / "forbidden", forbidden, u3, u2, 1024, tmp_path / "u3"
    )
