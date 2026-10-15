"""JSON projection of native checkpoints, including queued LangGraph tool sends."""

from fastapi.encoders import jsonable_encoder
from langgraph.types import Send


def encode_trace(value):
    return jsonable_encoder(
        value,
        custom_encoder={
            Send: lambda send: encode_trace(
                {"node": send.node, "arg": send.arg, "timeout": send.timeout}
            ),
        },
    )


def trace_config(run_id, namespace=None):
    config = {"configurable": {"thread_id": str(run_id)}}
    if namespace is not None:
        config["configurable"]["checkpoint_ns"] = namespace
    return config
