"""Supported API entry point, bound to this machine without forwarded headers."""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", choices=("agent",), default="agent")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    uvicorn.run(
        "agent_service.app:app", host="127.0.0.1", port=args.port, proxy_headers=False
    )


if __name__ == "__main__":
    main()
