import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest
from agent_service.config import Settings
from agent_service.runtime import model_from_resources


def test_chatopenai_async_shared_client_configuration_and_no_sdk_retry(monkeypatch):
    async def check():
        monkeypatch.setenv("AGENT_MODEL", "synthetic-openai")
        monkeypatch.setenv("AGENT_MODEL_URL", "http://synthetic.invalid/v1")
        monkeypatch.setenv("AGENT_MODEL_KEY", "test-only")
        requests = []
        status = 200

        def handle(request):
            requests.append(request)
            if status != 200:
                return httpx.Response(
                    status,
                    json={
                        "error": {
                            "message": "synthetic failure",
                            "type": "server_error",
                        }
                    },
                )
            return httpx.Response(
                200,
                json={
                    "id": "test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "synthetic-openai",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "SDK answer"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with httpx.Client(transport=httpx.MockTransport(handle)) as sync_client:
                resources = SimpleNamespace(
                    settings=Settings(), http=client, sync_http=sync_client
                )
                model = model_from_resources(resources)
                assert (await model.ainvoke("hello")).text == "SDK answer"
                assert model.max_retries == 0 and model.request_timeout == 120
                assert model.max_tokens == 4096
                assert model.root_async_client._client is client
                status = 500
                with pytest.raises(openai.InternalServerError):
                    await model.ainvoke("failure")
                assert len(requests) == 2
                assert not client.is_closed
        assert client.is_closed and sync_client.is_closed

    asyncio.run(check())
