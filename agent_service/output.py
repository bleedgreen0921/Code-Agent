"""Bound content including all result wrapping, with standard message artifacts."""


def limit_output(content, max_bytes=16384, max_lines=200):
    encoded = content.encode("utf-8", "replace")
    by_bytes = len(encoded) > max_bytes
    by_lines = len(content.splitlines()) > max_lines
    if not by_bytes and not by_lines:
        return content, False
    suffix = "\n[output truncated]"
    text = encoded[: max_bytes - len(suffix)].decode("utf-8", "ignore")
    text = "\n".join(text.splitlines()[: max_lines - 1])
    return text + suffix, True


def safe_content(content):
    """Only text and bounded JSON reach model history, never inline binary data."""
    import json

    if isinstance(content, str):
        return content
    parts = []
    for block in content if isinstance(content, list) else [content]:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            kind = block.get("type", "json")
            if kind == "text":
                parts.append(str(block.get("text", "")))
            elif kind in {"json", "structured"}:
                parts.append(
                    json.dumps(block.get("data", {}), ensure_ascii=False, default=str)
                )
            else:
                parts.append(f"[{kind} content omitted]")
        else:
            parts.append("[unsupported content omitted]")
    return "\n".join(parts)


def bounded_artifact(value, max_bytes=16384):
    import json

    if value is None:
        return None

    def clean(item, depth=0):
        if depth > 12:
            return "[depth limit]"
        if isinstance(item, dict):
            kind = item.get("type")
            if kind in {"image", "audio", "resource", "resource_link", "file"}:
                return {"type": kind, "omitted": True}
            return {
                str(k): clean(v, depth + 1)
                for k, v in list(item.items())[:200]
                if k not in {"data", "blob", "base64", "image_url", "audio_url"}
            }
        if isinstance(item, (list, tuple)):
            return [clean(v, depth + 1) for v in item[:200]]
        if isinstance(item, str):
            return limit_output(item, max_bytes, 200)[0]
        if isinstance(item, (int, float, bool)) or item is None:
            return item
        return "[unsupported artifact omitted]"

    # Reserve trusted diagnosis before trimming unrelated artifact data.
    diagnosis = None
    if isinstance(value, dict) and value.get("failure_info") is not None:
        from agent_service.failure import validated_info

        diagnosis = validated_info(value["failure_info"])
    result = clean(value)
    if len(json.dumps(result, ensure_ascii=False, default=str).encode()) > max_bytes:
        fallback = {"truncated": True, "message": "Artifact exceeded output limit"}
        if diagnosis is not None:
            fallback["failure_info"] = diagnosis
        return fallback
    return (
        value
        if isinstance(value, (dict, list, tuple, str, int, float, bool))
        and result == value
        else result
    )
