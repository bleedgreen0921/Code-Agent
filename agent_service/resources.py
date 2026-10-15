import os
from contextlib import AsyncExitStack

import httpx
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from agent_service.config import Settings
from agent_service.profiles import Registry


class Resources:
    def __init__(self, settings=None, *, registry=None):
        self.settings = settings or Settings.from_env()
        self.registry = registry or Registry.from_env()

    async def __aenter__(self):
        self.stack = AsyncExitStack()
        await self.stack.__aenter__()
        try:
            dsn = make_conninfo(
                os.environ["AGENT_DATABASE_URL"],
                options="-c search_path=agent -c statement_timeout=15000 -c lock_timeout=5000",
                connect_timeout=5,
            )
            self.pool = await self.stack.enter_async_context(
                AsyncConnectionPool(
                    dsn,
                    min_size=1,
                    max_size=6,
                    timeout=10,
                    kwargs={
                        "autocommit": True,
                        "row_factory": dict_row,
                        "prepare_threshold": 0,
                    },
                    open=False,
                )
            )
            await self.pool.wait(timeout=10)
            self.http = await self.stack.enter_async_context(
                httpx.AsyncClient(timeout=self.settings.model_timeout)
            )
            self.sync_http = self.stack.enter_context(
                httpx.Client(timeout=self.settings.model_timeout)
            )
            self.checkpointer = AsyncPostgresSaver(self.pool)
            self.knowledge_store = None
            if self.registry.knowledge.enabled:
                from agent_service.knowledge import native_store

                self.knowledge_store = await self.stack.enter_async_context(
                    native_store(self.pool)
                )
            from agent_service.storage import Store

            self.store = Store(self.pool, self.settings, self.registry)
            await self.store.ready()
            return self
        except BaseException:
            await self.stack.aclose()
            raise

    async def __aexit__(self, *args):
        return await self.stack.__aexit__(*args)
