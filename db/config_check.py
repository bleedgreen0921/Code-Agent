import argparse
import os

from agent_service.config import Settings
from agent_service.profiles import Registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("agent-api", "agent-worker"), required=True)
    args = parser.parse_args()
    Settings.from_env()
    Registry.from_env()
    names = ["AGENT_DATABASE_URL"]
    if args.role == "agent-worker":
        names += ["AGENT_MODEL_URL", "AGENT_MODEL"]
    for name in names:
        if not os.environ.get(name):
            raise ValueError(f"{name} required")
    print("Configuration valid")


if __name__ == "__main__":
    main()
