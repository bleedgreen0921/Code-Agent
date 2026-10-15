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
