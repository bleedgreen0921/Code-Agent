"""Compatibility boundary for LangChain 1.4.2. Native Shell framing/tool are retained."""

import asyncio
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.shell_tool import (
    DockerExecutionPolicy,
    ShellSession,
    ShellToolMiddleware,
    CommandExecutionResult,
)
from langchain_core.messages import ToolMessage

from agent_service.execution_errors import (
    CommandTimeout,
    EnvironmentLost,
)
from agent_service.failure import failure_info, is_control
from agent_service.output import limit_output
from agent_service.policy import active_call


class AttachedDockerPolicy(DockerExecutionPolicy):
    """Adapt the native strategy to an already registered, stopped-then-started container."""

    def __init__(self, scope):
        cfg = scope.settings
        super().__init__(
            image=cfg.image,
            network_enabled=False,
            read_only_rootfs=True,
            command_timeout=cfg.command_timeout,
            termination_timeout=cfg.cleanup_timeout,
            max_output_bytes=cfg.output_bytes,
            max_output_lines=cfg.output_lines,
            user=cfg.effective_user(),
            memory_bytes=cfg.memory_bytes,
            cpus=str(cfg.cpus),
        )
        self.scope = scope

    def spawn(self, *, workspace, env, command):
        return self.scope.spawn_shell(command)


class BoundedShellSession(ShellSession):
    """Keep native execute() and random completion markers; bound the reader and collector."""

    def __init__(self, *args, scope, **kwargs):
        self.scope = scope
        self.readers_closed = threading.Event()
        super().__init__(*args, **kwargs)

    @property
    def _queue(self):
        return self._bounded_queue

    @_queue.setter
    def _queue(self, value):
        self._bounded_queue = queue.Queue(maxsize=32)

    def start(self):
        self.scope.check_open()
        if self._terminated:
            raise EnvironmentLost("Shell cannot be restarted")
        super().start()

    def restart(self):
        self.scope.failure = CommandTimeout("Shell restart is disabled")
        raise self.scope.failure

    def _enqueue_stream(self, stream, label):
        import os

        try:
            while not self.readers_closed.is_set():
                chunk = os.read(stream.fileno(), 4096)
                if not chunk:
                    break
                self._put((label, chunk))
        except (OSError, ValueError):
            pass
        finally:
            self._put((label, None))

    def _put(self, value):
        while not self.readers_closed.is_set():
            try:
                self._queue.put(value, timeout=0.05)
                return
            except queue.Full:
                pass

    def stop(self, timeout):
        process = self._process
        self.readers_closed.set()
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
        super().stop(timeout)
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread:
                thread.join(timeout=max(timeout, 0.1))
        if any(
            thread and thread.is_alive()
            for thread in (self._stdout_thread, self._stderr_thread)
        ):
            from agent_service.execution_errors import CleanupFailed

            raise CleanupFailed("Shell reader threads have not stopped")
        for stream in (
            (process.stdin, process.stdout, process.stderr) if process else ()
        ):
            if stream:
                try:
                    stream.close()
                except OSError:
                    pass

    def execute(self, command, *, timeout):
        self.scope.check_open()
        return super().execute(command, timeout=min(timeout, self.scope.remaining()))

    def _drain_queue(self):
        # Background output may be continuous; never wait for an unbounded emptying.
        for _ in range(32):
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _collect_output(self, marker, deadline, timeout):
        # Fixed-size tail handles markers split between chunks or following a long line.
        token = marker.encode()
        pending = b""
        output = bytearray()
        total_bytes = total_lines = 0
        exit_code = None
        completed = False
        drain_deadline = None

        def retain(data):
            nonlocal total_bytes, total_lines
            total_bytes += len(data)
            total_lines += data.count(b"\n")
            if len(output) < self._policy.max_output_bytes:
                output.extend(data[: self._policy.max_output_bytes - len(output)])

        while True:
            if self.scope.closing:
                raise EnvironmentLost("Execution scope is closing")
            remaining = min(deadline, drain_deadline or deadline) - time.monotonic()
            if remaining <= 0:
                if completed:
                    break
                self.scope.failure = CommandTimeout("Shell command deadline expired")
                raise self.scope.failure
            try:
                source, data = self._queue.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                if not completed and self._process.poll() is not None:
                    self.scope.failure = EnvironmentLost(
                        "Shell exited before completing its command"
                    )
                    raise self.scope.failure
                continue
            if data is None:
                continue
            if source == "stderr":
                retain(data)
                continue
            if completed:
                retain(data)
                continue
            pending += data
            index = pending.find(token)
            if index >= 0:
                end = pending.find(b"\n", index)
                if end >= 0:
                    retain(pending[:index])
                    try:
                        exit_code = int(pending[index + len(token) : end].strip())
                    except ValueError:
                        self.scope.failure = EnvironmentLost(
                            "Invalid Shell completion marker"
                        )
                        raise self.scope.failure
                    pending = b""
                    completed = True
                    # Both pipes remain bounded while collecting trailing stderr.
                    drain_deadline = min(deadline, time.monotonic() + 0.05)
                    continue
            keep = len(token) + 32
            if len(pending) > keep:
                retain(pending[:-keep])
                pending = pending[-keep:]
        text = bytes(output).decode("utf-8", "replace")
        text, truncated = limit_output(
            text, self._policy.max_output_bytes, self._policy.max_output_lines
        )
        return CommandExecutionResult(
            text,
            exit_code,
            False,
            total_lines > self._policy.max_output_lines,
            total_bytes > self._policy.max_output_bytes or truncated,
            total_lines,
            total_bytes,
        )

    def _collect_output_after_exit(self, deadline):
        self.scope.failure = EnvironmentLost("Shell process exited unexpectedly")
        raise self.scope.failure


class ControlledShellMiddleware(ShellToolMiddleware):
    def __init__(self, scope=None):
        self.scope = scope
        self.resources = None
        super().__init__(
            workspace_root="/workspace",
            env={},
            execution_policy=AttachedDockerPolicy(scope)
            if scope
            else DockerExecutionPolicy(),
        )
        if scope is not None:

            async def tracked_shell(**kwargs):
                scope.check_open()
                task = asyncio.create_task(
                    asyncio.to_thread(self._shell_tool.func, **kwargs)
                )
                scope.native_tasks.add(task)

                def settled(done):
                    scope.native_tasks.discard(done)
                    if not done.cancelled():
                        done.exception()  # Observe errors even when the graph was cancelled.

                task.add_done_callback(settled)
                return await asyncio.shield(task)

            self._shell_tool.coroutine = tracked_shell

    def before_agent(self, state, runtime):
        return None

    async def abefore_agent(self, state, runtime):
        return None

    def after_agent(self, state, runtime):
        return None  # Worker scope owns cleanup on every graph exit path.

    async def aafter_agent(self, state, runtime):
        return None

    def _get_or_create_resources(self, state):
        self.scope.check_open()
        if self.resources is None:
            session = BoundedShellSession(
                Path("/workspace"),
                self._execution_policy,
                ("/bin/bash", "--noprofile", "--norc"),
                {},
                scope=self.scope,
            )
            self.scope.session = session
            session.start()
            self.resources = SimpleNamespace(
                session=session, policy=self._execution_policy
            )
        return self.resources

    def _run_shell_tool(self, resources, payload, *, tool_call_id):
        if payload.get("restart"):
            return self._format_tool_message(
                "Shell restart is disabled",
                tool_call_id,
                status="error",
                artifact={"timed_out": False, "exit_code": None},
            )
        return super()._run_shell_tool(resources, payload, tool_call_id=tool_call_id)

    def _format_tool_message(self, content, tool_call_id, *, status, artifact=None):
        artifact = dict(artifact or {})
        if status == "error":
            code = (
                "SHELL_COMMAND_FAILED"
                if type(artifact.get("exit_code")) is int and artifact["exit_code"] != 0
                else "INVALID_TOOL_ARGUMENTS"
            )
            artifact["failure_info"] = failure_info(code)
            call = active_call.get()
            if call:
                call.diagnostic = artifact["failure_info"]
        content, truncated = limit_output(
            content, self.scope.settings.output_bytes, self.scope.settings.output_lines
        )
        artifact["truncated"] = (
            truncated
            or artifact.get("truncated_by_lines", False)
            or artifact.get("truncated_by_bytes", False)
        )
        return super()._format_tool_message(
            content, tool_call_id, status=status, artifact=artifact
        )


class WorkspaceExecutionMiddleware(AgentMiddleware):
    def __init__(self, scope, names):
        self.scope, self.names = scope, set(names)

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in self.names:
            return await handler(request)
        scope = self.scope
        try:
            async with scope.lock:
                scope.check_open()
                await scope.healthy()
                if (
                    request.tool_call["name"] in {"edit_file", "shell"}
                    and scope.row.get("task_id")
                    and await scope.store.workspace_background(scope.row["task_id"])
                ):
                    return ToolMessage(
                        content='{"code":"BACKGROUND_WORKSPACE_READ_ONLY"}',
                        tool_call_id=request.tool_call["id"],
                        status="error",
                        artifact={
                            "failure_info": failure_info(
                                "BACKGROUND_WORKSPACE_READ_ONLY"
                            )
                        },
                    )
                run_limited = scope.remaining() <= scope.settings.command_timeout
                try:
                    async with asyncio.timeout(
                        min(scope.remaining(), scope.settings.command_timeout)
                    ):
                        result = await handler(request)
                except TimeoutError as exc:
                    scope.failure = (
                        TimeoutError("Run deadline expired")
                        if run_limited
                        else CommandTimeout("Tool command deadline expired")
                    )
                    raise scope.failure from exc
                if (
                    isinstance(result, ToolMessage)
                    and (result.artifact or {}).get("exit_code") == 137
                ):
                    raise EnvironmentLost(
                        "Execution process was killed, possibly by the container memory limit"
                    )
                await scope.healthy()
                return result
        except BaseException as exc:
            # Outer Hook/ToolError middleware cannot swallow fatal execution controls.
            if is_control(exc):
                call = active_call.get()
                if call:
                    call.failure = exc
                scope.failure = exc
                scope.closing = True
            raise
