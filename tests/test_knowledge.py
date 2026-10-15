import asyncio
import json

import httpx
import psycopg
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.store.base import GetOp, ListNamespacesOp, PutOp, SearchOp
from langgraph.store.memory import InMemoryStore
from starlette.exceptions import HTTPException

from agent_service.app import app
from agent_service.knowledge import (
    Entry,
    KnowledgeManager,
    KnowledgeSettings,
    ReadOnlyStore,
    knowledge_tools,
    namespace,
    native_store,
)
from agent_service.profiles import Profile, Registry, RuntimeFile, Workspace
from agent_service.runtime import RunContext, build_agent, run_config
from agent_service.worker import Worker
from tests.models import SyntheticModel
from tests.test_profiles_policy import call
from tests.test_runtime_bindings import execute, install


def registry(root, enabled=True):
    return Registry(
        RuntimeFile(
            version=1,
            knowledge=KnowledgeSettings(enabled=enabled),
            workspaces=[
                Workspace(id="project", root=str(root)),
                Workspace(id="other", root=str(root)),
            ],
            profiles=[
                Profile(id="knowledge", tools=("knowledge_get", "knowledge_search"))
            ]
            if enabled
            else [],
        )
    )


async def enable(r, root):
    install(r, registry(root))
    r.knowledge_store = await r.stack.enter_async_context(native_store(r.pool))
    return KnowledgeManager(r.pool, r.knowledge_store, r.registry)


def test_native_store_api_versions_pagination_filter_and_delete(database, tmp_path):
    async def check(r):
        manager = await enable(r, tmp_path)
        app.state.resources = r
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://127.0.0.1",
        ) as client:
            path = "/v1/workspaces/project/knowledge/design"
            body = {
                "title": "Build",
                "body": "Use the strict test entry.",
                "category": "testing",
                "source": "maintainer",
            }
            assert (await client.put(path, json=body)).status_code == 428
            created = await client.put(path, json=body, headers={"If-None-Match": "*"})
            assert created.status_code == 201
            etag = created.headers["etag"]
            assert (
                await client.put(path, json=body, headers={"If-None-Match": "*"})
            ).status_code == 412
            assert (await client.get(path)).headers["etag"] == etag
            assert (
                await client.get(path.replace("project", "other"))
            ).status_code == 404
            assert (
                await client.put(path, json=body, headers={"If-Match": '"stale"'})
            ).status_code == 412
            changed = await client.put(
                path, json={**body, "body": "Updated"}, headers={"If-Match": etag}
            )
            assert changed.status_code == 200 and changed.headers["etag"] != etag
            for n in range(2):
                await client.put(
                    path + str(n), json=body, headers={"If-None-Match": "*"}
                )
            listed = (
                await client.get(
                    path.rsplit("/", 1)[0],
                    params={"category": "testing", "source": "maintainer", "limit": 1},
                )
            ).json()
            assert len(listed["items"]) == 1 and listed["next_offset"] == 1
            assert "body" not in listed["items"][0] and "summary" in listed["items"][0]
            assert (
                await client.get(path.rsplit("/", 1)[0], params={"category": "missing"})
            ).json()["items"] == []
            assert (await client.delete(path)).status_code == 428
            assert (
                await client.delete(path, headers={"If-Match": etag})
            ).status_code == 412
            assert (
                await client.delete(path, headers={"If-Match": changed.headers["etag"]})
            ).status_code == 204
            assert (await client.get(path)).status_code == 404
            assert (
                await client.get(path.replace("project", "missing"))
            ).status_code == 404
            assert (
                await client.put(
                    path + "bad",
                    json={**body, "body": "中" * 3000},
                    headers={"If-None-Match": "*"},
                )
            ).status_code == 422
            assert (
                await client.put(
                    path + "bad",
                    json={**body, "body": "x\n" * 181},
                    headers={"If-None-Match": "*"},
                )
            ).status_code == 422
        assert (
            len(
                await r.knowledge_store.asearch(
                    manager.ns("project"), refresh_ttl=False
                )
            )
            == 2
        )

    database(check)


def test_knowledge_disabled_and_unknown_workspace_404(database, tmp_path):
    async def check(r):
        app.state.resources = r
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://127.0.0.1",
        ) as c:
            for method, path, kwargs in [
                ("GET", "", {}),
                ("GET", "/key", {}),
                (
                    "PUT",
                    "/key",
                    {
                        "json": {
                            "title": "t",
                            "body": "b",
                            "category": "c",
                            "source": "s",
                        },
                        "headers": {"If-None-Match": "*"},
                    },
                ),
                ("DELETE", "/key", {"headers": {"If-Match": '"a"'}}),
            ]:
                response = await c.request(
                    method, "/v1/workspaces/missing/knowledge" + path, **kwargs
                )
                assert (
                    response.status_code == 404
                    and response.json()["error"]["code"] == "NOT_FOUND"
                )

    database(check)


def test_knowledge_concurrent_updates_have_one_winner_and_rollback(
    database, tmp_path, monkeypatch
):
    async def check(r):
        m = await enable(r, tmp_path)
        body = Entry(title="t", body="initial", category="c", source="s")
        first = await m.mutate("project", "key", body, None, "*")

        async def change(text):
            try:
                return await m.mutate(
                    "project",
                    "key",
                    body.model_copy(update={"body": text}),
                    '"' + first["version"] + '"',
                    None,
                )
            except HTTPException as exc:
                return exc.status_code

        outcomes = await asyncio.gather(change("one"), change("two"))
        assert sum(isinstance(o, dict) for o in outcomes) == 1 and 412 in outcomes
        from langgraph.store.postgres.aio import AsyncPostgresStore

        original = AsyncPostgresStore.abatch

        async def broken(store, ops):
            ops = list(ops)
            result = await original(store, ops)
            if any(isinstance(op, PutOp) for op in ops):
                raise RuntimeError("after SQL before commit")
            return result

        monkeypatch.setattr(AsyncPostgresStore, "abatch", broken)
        current = await m.get("project", "key")
        with pytest.raises(RuntimeError, match="after SQL"):
            await m.mutate("project", "key", body, '"' + current["version"] + '"', None)
        assert await m.get("project", "key") == current

    database(check)


def test_readonly_store_rejects_writes_namespaces_queries_and_refreshes():
    async def check():
        native = InMemoryStore()
        ns = namespace("p", "identity")
        await native.aput(ns, "key", {"body": "read"})
        view = ReadOnlyStore(native, ns)
        with pytest.raises(AttributeError):
            view.namespace = ("forged",)
        assert (await view.aget(ns, "key")).value == {"body": "read"}
        assert (await asyncio.to_thread(view.get, ns, "key")).value == {"body": "read"}
        for op in [
            PutOp(ns, "key", {}),
            PutOp(ns, "key", None),
            GetOp(("other",), "key"),
            ListNamespacesOp(),
            SearchOp(ns[:-1]),
            SearchOp(ns, query="keyword"),
            SearchOp(ns, filter={"source": {"$ne": "a"}}),
            SearchOp(ns, limit=1000),
        ]:
            with pytest.raises(PermissionError):
                await view.abatch([op])
            with pytest.raises(PermissionError):
                await asyncio.to_thread(view.batch, [op])
        await native.aput((*ns, "private"), "nested", {"body": "not project knowledge"})
        assert len(await view.asearch(ns, refresh_ttl=True)) == 1
        assert len(await asyncio.to_thread(view.search, ns)) == 1
        assert native.get(ns, "key").value == {"body": "read"}
        tools = knowledge_tools(view)
        assert (
            "query" not in tools[1].tool_call_schema.model_json_schema()["properties"]
        )

    asyncio.run(check())


def test_knowledge_profile_binding_without_docker_and_root_isolation(
    database, tmp_path
):
    async def check(r):
        manager = await enable(r, tmp_path)
        assert not r.registry.execution.image
        with pytest.raises(ValueError, match="workspace"):
            r.registry.bind("knowledge")
        binding = r.registry.bind("knowledge", "project")
        item = await manager.mutate(
            "project",
            "key",
            Entry(
                title="Title", body="Persistent knowledge", category="c", source="human"
            ),
            None,
            "*",
        )
        assert binding == r.registry.bind("knowledge", "project")
        model = SyntheticModel(
            responses=[call("knowledge_get", {"key": "key"}), AIMessage("read")]
        )
        await r.store.submit("read", profile="knowledge", workspace_id="project")
        completed = await execute(r, Worker(r, model=model))
        assert completed["status"] == "completed"
        assert await r.store.execution_resource(completed["id"]) is None
        message = next(m for m in model.history[-1] if isinstance(m, ToolMessage))
        assert json.loads(message.content)["version"] == item["version"]
        # Recreate resources/Worker to exercise persistence without cached bodies.
        from agent_service.resources import Resources

        async with Resources(r.settings, registry=r.registry) as restarted:
            other = KnowledgeManager(
                restarted.pool, restarted.knowledge_store, restarted.registry
            )
            assert (await other.get("project", "key"))["body"] == "Persistent knowledge"
            new_model = SyntheticModel(
                responses=[call("knowledge_search", {}), AIMessage("found")]
            )
            await restarted.store.submit(
                "read", profile="knowledge", workspace_id="project"
            )
            result = await execute(restarted, Worker(restarted, model=new_model))
            assert result["status"] == "completed"
        root2 = tmp_path / "new-root"
        root2.mkdir()
        changed = KnowledgeManager(r.pool, r.knowledge_store, registry(root2))
        with pytest.raises(HTTPException) as exc:
            await changed.get("project", "key")
        assert exc.value.status_code == 404
        assert manager.ns("other") != manager.ns("project")

    database(check)


def test_knowledge_task_closed_with_service_lifecycle(database, tmp_path):
    async def check(r):
        from agent_service.resources import Resources

        async with Resources(r.settings, registry=registry(tmp_path)) as other:
            store = other.knowledge_store
            assert not store._task.done()
            await store.aget(namespace("p", "id"), "k", refresh_ttl=False)
        assert store._task.done() and store._ttl_sweeper_task is None

    database(check)


def test_deleted_knowledge_stays_deleted_with_inherited_history(database, tmp_path):
    async def check(r):
        m = await enable(r, tmp_path)
        item = await m.mutate(
            "project",
            "deleted",
            Entry(title="Past", body="Old snapshot", category="c", source="human"),
            None,
            "*",
        )
        conversation = await r.store.create_conversation(
            "knowledge", "knowledge", "project"
        )
        await r.store.submit("read", conversation_id=conversation["id"])
        first = await execute(
            r,
            Worker(
                r,
                model=SyntheticModel(
                    responses=[
                        call("knowledge_get", {"key": "deleted"}),
                        AIMessage("read"),
                    ]
                ),
            ),
        )
        assert first["status"] == "completed"
        await m.mutate("project", "deleted", None, '"' + item["version"] + '"', None)
        model = SyntheticModel(
            responses=[
                call("knowledge_get", {"key": "deleted"}, id="new"),
                AIMessage("deleted"),
            ]
        )
        await r.store.submit("read again", conversation_id=conversation["id"])
        second = await execute(r, Worker(r, model=model))
        assert second["status"] == "completed"
        messages = [x for x in model.history[-1] if isinstance(x, ToolMessage)]
        assert "Old snapshot" in messages[0].content
        assert json.loads(messages[-1].content)["code"] == "NOT_FOUND"
        assert (
            await r.knowledge_store.aget(m.ns("project"), "deleted", refresh_ttl=False)
            is None
        )

    database(check)


def test_native_subgraph_inherited_store_remains_readonly():
    async def check():
        from langchain.agents import create_agent
        from langchain.tools import ToolRuntime, tool
        from langgraph.checkpoint.memory import InMemorySaver

        native = InMemoryStore()
        ns = namespace("p", "identity")
        await native.aput(ns, "key", {"body": "protected"})
        view = ReadOnlyStore(native, ns)
        observed = []

        @tool
        async def inspect_store(runtime: ToolRuntime[RunContext]) -> str:
            """Inspect the native inherited runtime resource."""
            observed.append(runtime.store)
            if runtime.store is not None:
                with pytest.raises(PermissionError):
                    await runtime.store.aput(ns, "key", {"body": "overwritten"})
                with pytest.raises(PermissionError):
                    await runtime.store.aget(("another",), "key")
            return "guarded"

        child = create_agent(
            SyntheticModel(responses=[call("inspect_store"), AIMessage("safe")]),
            tools=[inspect_store],
            context_schema=RunContext,
            store=None,
        )

        @tool
        async def nested(runtime: ToolRuntime[RunContext]) -> str:
            """Run a native child graph."""
            result = await child.ainvoke(
                {"messages": [{"role": "user", "content": "inspect"}]},
                context=runtime.context,
            )
            return result["messages"][-1].text

        parent = build_agent(
            SyntheticModel(responses=[call("nested"), AIMessage("done")]),
            tools=[nested],
            store=view,
            checkpointer=InMemorySaver(),
        )
        await parent.ainvoke(
            {"messages": [{"role": "user", "content": "run"}]},
            run_config("inherit"),
            context=RunContext("inherit"),
        )
        assert observed and all(s is view or s is None for s in observed)
        assert native.get(ns, "key").value["body"] == "protected"

    asyncio.run(check())


def test_knowledge_committed_request_ack_loss_requires_read_not_replay(
    database, tmp_path, monkeypatch
):
    async def check(r):
        m = await enable(r, tmp_path)
        original = psycopg.AsyncTransaction.__aexit__
        armed = True

        async def lost_ack(transaction, *args):
            nonlocal armed
            result = await original(transaction, *args)
            if armed and not args[0] and not transaction._conn.info.transaction_status:
                armed = False
                raise psycopg.OperationalError("commit acknowledgement lost")
            return result

        monkeypatch.setattr(psycopg.AsyncTransaction, "__aexit__", lost_ack)
        with pytest.raises(psycopg.OperationalError):
            await m.mutate(
                "project",
                "ack",
                Entry(title="t", body="Committed once", category="c", source="s"),
                None,
                "*",
            )
        current = await m.get("project", "ack")
        assert current["body"] == "Committed once"
        with pytest.raises(HTTPException) as exc:
            await m.mutate(
                "project",
                "ack",
                Entry(title="t", body="Committed once", category="c", source="s"),
                None,
                "*",
            )
        assert exc.value.status_code == 412

    database(check)
