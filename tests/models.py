from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class SyntheticModel(BaseChatModel):
    responses: list[AIMessage] = []
    history: list[Any] = []
    reply: Any = None

    @property
    def _llm_type(self):
        return "synthetic"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("Execution must use async model calls")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self.history.append(list(messages))
        if self.reply:
            message = self.reply(messages)
            if hasattr(message, "__await__"):
                message = await message
        elif self.responses:
            message = self.responses.pop(0)
        else:
            message = AIMessage(content="synthetic answer")
        return ChatResult(generations=[ChatGeneration(message=message)])
