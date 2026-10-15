"""Run-owned Docker lifecycle. No tools or OS handles are persisted in checkpoints."""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import threading

from agent_service.execution_config import IMPLEMENTATION_VERSION
from agent_service.execution_errors import (
    EnvironmentUnavailable,
    EnvironmentLost,
    CleanupFailed,
)

LABEL = "io.agent-framework.resource"
FIXED_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "HOME": "/tmp",
    "TMPDIR": "/tmp",
}
HELPER = ["/usr/local/bin/python", "-I", "/opt/agent/helper.py"]


def docker_environment():
    # Used only by the host Docker client, never inherited by container processes.
    return {
        k: v
        for k, v in os.environ.items()
        if k
        in (
            "PATH",
            "HOME",
            "DOCKER_HOST",
            "DOCKER_CONTEXT",
            "DOCKER_TLS_VERIFY",
            "DOCKER_CERT_PATH",
            "XDG_RUNTIME_DIR",
        )
    }


class DockerClient:
    def __init__(self):
        self.processes = set()

    async def call(self, *args, stdin=None, timeout=10, maximum=65536, check=True):
        proc = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdin=asyncio.subprocess.PIPE
            if stdin is not None
            else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=docker_environment(),
            start_new_session=True,
        )
        self.processes.add(proc)

        async def drain(stream):
            output = bytearray()
            while chunk := await stream.read(4096):
                if len(output) < maximum:
                    output.extend(chunk[: maximum - len(output)])
            return bytes(output)

        async def feed():
            if stdin is not None:
                proc.stdin.write(stdin)
                await proc.stdin.drain()
                proc.stdin.close()

        try:
            async with asyncio.timeout(timeout):
                out, err, _ = await asyncio.gather(
                    drain(proc.stdout), drain(proc.stderr), feed()
                )
                await proc.wait()
            if check and proc.returncode:
                raise EnvironmentUnavailable(err.decode("utf-8", "replace")[:500])
            return proc.returncode, out, err
        finally:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()
            self.processes.discard(proc)

    async def inspect(self, identity):
        code, out, err = await self.call("container", "inspect", identity, check=False)
        if code:
            # A failed inspect is absence only when the daemon itself confirms absence.
            if b"No such container" in err or b"No such object" in err:
                await self.daemon()
                return None
            raise EnvironmentUnavailable(err.decode("utf-8", "replace")[:500])
        return json.loads(out)[0]

    async def daemon(self):
        _, out, _ = await self.call("info", "--format", "{{.ID}}")
        identity = out.decode().strip()
        if not identity:
            raise EnvironmentUnavailable("Docker daemon identity unavailable")
        return identity


class ExecutionScope:
    def __init__(self, store, row, settings):
        self.store, self.row, self.settings = store, row, settings
        self.client = DockerClient()
        self.resource = None
        self.container_id = None
        self.closing = False
        self.lock = asyncio.Lock()
        self.sync_lock = threading.Lock()
        self.shell_processes = set()
        self.native_tasks = set()
        self.session = None
        self.failure = None

    def check_open(self):
        if self.failure:
            raise self.failure
        if self.closing:
            raise EnvironmentLost("Execution scope is closing")

    async def update(self, **changes):
        self.check_open()
        result = await self.store.resource_update(
            self.row["id"], self.row["lease_token"], require_live=True, **changes
        )
        if result is None:
            raise EnvironmentLost("Execution resource ownership lost or Run cancelled")
        self.resource = result
        return result

    async def prepare(self):
        self.resource = await self.store.execution_resource(self.row["id"])
        if self.resource is None:
            raise EnvironmentUnavailable("Execution resource was not reserved")
        cfg = self.settings
        if not cfg.image:
            raise EnvironmentUnavailable("Immutable execution image is required")
        try:
            user = cfg.effective_user()
            daemon = await self.client.daemon()
            _, raw, _ = await self.client.call("image", "inspect", cfg.image)
            image = json.loads(raw)[0]
            if (
                image.get("Config", {})
                .get("Labels", {})
                .get("io.agent-framework.helper")
                != IMPLEMENTATION_VERSION
            ):
                raise EnvironmentUnavailable("Execution image helper version mismatch")
            root = self.resource["workspace_root"]
            if (
                str(Path(root).resolve(strict=True)) != root
                or not Path(root).is_dir()
                or "," in root
            ):
                raise EnvironmentUnavailable(
                    "Registered workspace path changed or is not mountable"
                )
            await self.update(daemon_id=daemon, phase="creating")
            name = self.resource["container_name"]
            readonly = not bool(
                set(self.row["execution_tools"]) & {"edit_file", "shell"}
            )
            mount = f"type=bind,source={root},target=/workspace" + (
                ",readonly" if readonly else ""
            )
            command = [
                "create",
                "--pull=never",
                "--name",
                name,
                "--label",
                f"{LABEL}={name}",
                "--label",
                f"io.agent-framework.lease={self.row['lease_token']}",
                "--network",
                "none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt",
                "no-new-privileges",
                "--user",
                user,
                "--cpus",
                str(cfg.cpus),
                "--memory",
                str(cfg.memory_bytes),
                "--memory-swap",
                str(cfg.memory_bytes),
                "--pids-limit",
                str(cfg.pids),
                "--tmpfs",
                f"/tmp:rw,nosuid,nodev,size={cfg.tmp_bytes},mode=1777",
                "--mount",
                mount,
                "--workdir",
                "/workspace",
                "--log-driver",
                "none",
            ]
            for key, value in FIXED_ENV.items():
                command += ["--env", f"{key}={value}"]
            command += [
                "--entrypoint",
                "/usr/local/bin/python",
                cfg.image,
                "-I",
                "/opt/agent/entrypoint.py",
                str(self.row["deadline_at"].timestamp()),
            ]
            _, out, _ = await self.client.call(*command, timeout=cfg.cleanup_timeout)
            identity = out.decode().strip()
            if len(identity) != 64:
                raise EnvironmentUnavailable(
                    "Container create acknowledgement is invalid"
                )
            # Persist ID before any start; late starts can only address this exact ID.
            await self.update(container_id=identity, phase="created")
            self.container_id = identity
            await self.update(phase="starting")
            self.check_open()
            await self.client.call("start", identity, timeout=cfg.cleanup_timeout)
            await self.update(state="active", phase="started")
            _, out, _ = await self.client.call(
                "exec",
                "-i",
                identity,
                *HELPER,
                stdin=b'{"operation":"version"}',
                timeout=min(cfg.command_timeout, self.remaining()),
            )
            if json.loads(out).get("version") != IMPLEMENTATION_VERSION:
                raise EnvironmentUnavailable("Execution helper version mismatch")
        except (OSError, ValueError, TimeoutError) as exc:
            raise EnvironmentUnavailable(str(exc)) from exc

    def remaining(self):
        from datetime import datetime, timezone

        return max(
            0.001,
            (self.row["deadline_at"] - datetime.now(timezone.utc)).total_seconds(),
        )

    async def healthy(self):
        self.check_open()
        result = await self.client.inspect(self.container_id)
        if not result or not result["State"]["Running"]:
            self.failure = EnvironmentLost(
                "Execution container stopped or was OOM killed"
            )
            raise self.failure

    async def file_call(self, name, args):
        self.check_open()
        code, out, err = await self.client.call(
            "exec",
            "-i",
            self.container_id,
            *HELPER,
            stdin=json.dumps(
                {
                    "operation": name,
                    "args": args,
                    "file_bytes": self.settings.file_bytes,
                }
            ).encode(),
            timeout=min(self.settings.command_timeout, self.remaining()),
            maximum=160000,
            check=False,
        )
        await self.healthy()
        if code:
            raise EnvironmentLost(
                "Execution helper exited unexpectedly: "
                + err.decode("utf-8", "replace")[:200]
            )
        try:
            return json.loads(out)
        except (ValueError, UnicodeError) as exc:
            raise EnvironmentLost("Invalid execution helper response") from exc

    def spawn_shell(self, command):
        # Called by the native Shell tool's executor thread. No docker run/restart here.
        with self.sync_lock:
            self.check_open()
            proc = subprocess.Popen(
                ["docker", "exec", "-i", self.container_id, *command],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=docker_environment(),
                start_new_session=True,
            )
            self.shell_processes.add(proc)
            return proc

    async def converge_local(self):
        # Even when Docker is unreachable, host clients/readers must stop. The container
        # remains reserved until a later recovery can confirm its removal.
        for proc in list(self.shell_processes):
            if proc.poll() is None:
                proc.kill()
            await asyncio.to_thread(proc.wait)
        if self.native_tasks:
            await asyncio.gather(*tuple(self.native_tasks), return_exceptions=True)
        if self.session:
            await asyncio.to_thread(self.session.stop, 0.2)
        self.shell_processes.clear()
        if self.client.processes:
            raise CleanupFailed("Docker client operations remain pending")

    async def close(self):
        with self.sync_lock:
            self.closing = True
        try:
            async with asyncio.timeout(self.settings.cleanup_timeout):
                self.resource = await self.store.execution_resource(self.row["id"])
                if self.resource is None or self.resource["state"] == "released":
                    await self.converge_local()
                    return True
                await self.store.resource_update(
                    self.row["id"], self.row["lease_token"], state="closing"
                )
                try:
                    await cleanup_resource(self.store, self.resource, self.client)
                finally:
                    await self.converge_local()
                await self.store.resource_update(
                    self.row["id"],
                    self.row["lease_token"],
                    state="released",
                    cleanup_error=None,
                )
                return True
        except Exception as exc:
            try:
                async with asyncio.timeout(self.settings.cleanup_timeout):
                    await self.converge_local()
                    current = await self.store.execution_resource(self.row["id"])
                    if current and current["state"] == "released":
                        return True  # Release commit acknowledgement was lost; now confirmed.
                    await self.store.resource_update(
                        self.row["id"],
                        self.row["lease_token"],
                        state="cleanup_failed",
                        cleanup_error=str(exc)[:500],
                    )
            except Exception:
                # The existing unreleased row remains authoritative if diagnostics fail.
                pass
            return False


async def cleanup_resource(store, resource, client):
    if resource["phase"] == "reserved" and resource["container_id"] is None:
        return
    daemon = await client.daemon()
    if resource["daemon_id"] and resource["daemon_id"] != daemon:
        raise CleanupFailed("Docker daemon identity mismatch; occupancy retained")
    identity = resource["container_id"] or resource["container_name"]
    inspected = await client.inspect(identity)
    if inspected:
        labels = inspected["Config"].get("Labels") or {}
        if labels.get(LABEL) != resource["container_name"] or labels.get(
            "io.agent-framework.lease"
        ) != str(resource["lease_token"]):
            raise CleanupFailed("Container ownership labels mismatch")
        identity = inspected["Id"]
        if resource["container_id"] is None:
            # Record positive evidence before removal, so lost removal acknowledgements
            # do not turn a resolved create back into an unknown operation.
            updated = await store.resource_update(
                resource["run_id"],
                resource["lease_token"],
                container_id=identity,
                phase="created",
            )
            if updated is None:
                current = await store.execution_resource(resource["run_id"])
                if current["state"] == "released":
                    return
                raise CleanupFailed("Container identity registration was not confirmed")
        # Positive observation resolves an uncertain create without repeating it.
        await client.call("rm", "--force", identity)
        if await client.inspect(identity) is not None:
            raise CleanupFailed("Container removal was not confirmed")
    elif resource["phase"] == "creating" and resource["container_id"] is None:
        raise CleanupFailed("Container creation result unknown; occupancy retained")


async def recover_resources(store, settings):
    for resource in await store.orphan_resources():
        client = DockerClient()
        try:
            async with asyncio.timeout(settings.cleanup_timeout):
                await store.resource_update(
                    resource["run_id"], resource["lease_token"], state="closing"
                )
                await cleanup_resource(store, resource, client)
                await store.resource_update(
                    resource["run_id"],
                    resource["lease_token"],
                    state="released",
                    cleanup_error=None,
                    recovered=True,
                )
        except Exception as exc:
            await store.resource_update(
                resource["run_id"],
                resource["lease_token"],
                state="cleanup_failed",
                cleanup_error=str(exc)[:500],
            )


async def check_environment(settings):
    """Worker-only preflight. API configuration parsing never calls this."""
    if not settings.image:
        raise EnvironmentUnavailable("Immutable execution image is required")
    try:
        settings.effective_user()
        client = DockerClient()
        daemon = await client.daemon()
        _, raw, _ = await client.call("image", "inspect", settings.image)
        image = json.loads(raw)[0]
        if (
            image.get("Config", {}).get("Labels", {}).get("io.agent-framework.helper")
            != IMPLEMENTATION_VERSION
        ):
            raise EnvironmentUnavailable("Execution image helper version mismatch")
        return daemon
    except (OSError, ValueError, TimeoutError) as exc:
        raise EnvironmentUnavailable(str(exc)) from exc
