"""Frozen local-only JSON Schema validation at the final execution boundary."""

from copy import deepcopy

from jsonschema.validators import validator_for


def local_schema(schema):
    schema = deepcopy(schema)

    def check(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"$ref", "$dynamicRef", "$recursiveRef"} and (
                    not isinstance(child, str) or not child.startswith("#")
                ):
                    raise ValueError("Only local JSON Schema references are permitted")
                if key == "$id" and child:
                    raise ValueError(
                        "JSON Schema resource identifiers are not permitted"
                    )
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(schema)
    cls = validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)
