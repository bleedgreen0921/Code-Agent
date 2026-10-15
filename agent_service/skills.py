"""Server-registered, startup-only Agent Skills snapshots. No executable resources."""

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

SKILL_VERSION = "skills-v1"
CATALOG_BYTES = 8192


class SkillLoader(yaml.SafeLoader):
    """Keep safe YAML types while rejecting ambiguous duplicate metadata keys."""

    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if any(key in keys[:i] for i, key in enumerate(keys)):
            raise ValueError("Duplicate skill metadata key")
        return super().construct_mapping(node, deep=deep)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    sha256: str
    metadata_json: str

    def manifest(self):
        return {
            "name": self.name,
            "description": self.description,
            "sha256": self.sha256,
        }


def catalog(skills):
    return json.dumps(
        [{"name": s.name, "description": s.description} for s in skills],
        ensure_ascii=False,
    )


def load_skills(directories, output_bytes=16384, output_lines=200):
    skills = {}
    for directory in directories:
        root = Path(directory)
        if not root.is_absolute() or not root.is_dir():
            raise ValueError("Skill directory must be an existing absolute directory")
        root = root.resolve(strict=True)
        for child in sorted(root.iterdir()):
            resolved = child.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise ValueError(f"Skill link escapes registered directory: {child}")
            if not resolved.is_dir():
                if not resolved.is_file():
                    raise ValueError(f"Special skill entry: {child}")
                continue
            path = child / "SKILL.md"
            try:
                target = path.resolve(strict=True)
            except FileNotFoundError as exc:
                raise ValueError(f"Missing SKILL.md: {child}") from exc
            if not target.is_relative_to(root):
                raise ValueError(f"Skill link escapes registered directory: {path}")
            # O_NONBLOCK prevents a malicious special file from blocking startup.
            fd = os.open(target, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError(f"Skill must be a regular file: {path}")
                try:
                    text = stream.read().decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"Skill must be UTF-8: {path}") from exc
            lines = text.splitlines(keepends=True)
            if not lines or lines[0].strip() != "---":
                raise ValueError(f"Missing YAML frontmatter: {path}")
            end = next(
                (i for i in range(1, len(lines)) if lines[i].strip() == "---"), None
            )
            if end is None:
                raise ValueError(f"Unclosed YAML frontmatter: {path}")
            try:
                meta = yaml.load("".join(lines[1:end]), Loader=SkillLoader)
                # JSON also rejects YAML dates and other nonstandard metadata objects.
                metadata_json = json.dumps(meta, ensure_ascii=False, sort_keys=True)
            except (yaml.YAMLError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid skill metadata: {path}") from exc
            if not isinstance(meta, dict):
                raise ValueError(f"Skill metadata must be a mapping: {path}")
            name, description = meta.get("name"), meta.get("description")
            if (
                not isinstance(name, str)
                or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
                or len(name) > 64
                or name != child.name
                or not isinstance(description, str)
                or not description.strip()
                or len(description) > 1024
            ):
                raise ValueError(f"Invalid skill name or description: {path}")
            for key in ("license", "compatibility", "allowed-tools"):
                if key in meta and not isinstance(meta[key], str):
                    raise ValueError(f"Invalid skill {key}: {path}")
            if "compatibility" in meta and not 1 <= len(meta["compatibility"]) <= 500:
                raise ValueError(f"Invalid skill compatibility: {path}")
            if "metadata" in meta and (
                not isinstance(meta["metadata"], dict)
                or any(
                    not isinstance(k, str) or not isinstance(v, str)
                    for k, v in meta["metadata"].items()
                )
            ):
                raise ValueError(f"Invalid skill metadata mapping: {path}")
            body = "".join(lines[end + 1 :])
            if (
                len(body.encode()) > output_bytes
                or len(body.splitlines()) > output_lines
            ):
                raise ValueError(
                    f"Skill body exceeds tool output limits; shorten SKILL.md: {path}"
                )
            if name in skills:
                raise ValueError(f"Duplicate skill name: {name}")
            skills[name] = Skill(
                name,
                description,
                body,
                hashlib.sha256(body.encode()).hexdigest(),
                metadata_json,
            )
    return MappingProxyType(skills)


class ReadSkillInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)


def read_skill_tool(skills):
    allowed = MappingProxyType(dict(skills))

    async def read_skill(name):
        skill = allowed.get(name)
        if skill is None:
            raise ToolException(
                "SKILL_NOT_AVAILABLE: Skill is not available for this profile."
            )
        return skill.body, {"name": skill.name, "sha256": skill.sha256}

    return StructuredTool.from_function(
        coroutine=read_skill,
        name="read_skill",
        description="Read the complete instructions of a registered skill by name from the current profile's skill catalog.",
        args_schema=ReadSkillInput,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )
