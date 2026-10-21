"""Validate explicit AMD portable-route evidence without weakening logit gates."""
from functools import cache
import hashlib
import json
from pathlib import Path

RECORD = Path(__file__).with_name("amd_matmul_decisions.json")


@cache
def measurements():
    document = json.loads(RECORD.read_text(encoding="utf-8"))
    if document["schema"] != 1:
        raise ValueError("unsupported AMD matmul measurement schema")
    for name, digest in document["shader_sha256"].items():
        if Path(name).name != name:
            raise ValueError("invalid AMD module name")
        artifact = RECORD.parent.parent / "shaders" / name
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
            raise ValueError(f"AMD matmul shader changed since measurement: {name}")
    return document["decisions"]


def validates_fallback(evidence, family):
    """A missing vendor dispatch needs a real, auditable shipping route."""
    required = ("device", "family", "rows", "input_dim", "output_dim", "parameter_storage", "reason")
    if any(key not in evidence for key in required):
        return False
    if evidence["family"] != family or evidence["parameter_storage"] != "fp32-compute":
        return False
    device = evidence["device"]
    if not isinstance(device, str) or "[pci=1002:" not in device or ";driver=" not in device:
        return False
    if any(type(evidence[key]) is not int or evidence[key] <= 0
           for key in ("rows", "input_dim", "output_dim")):
        return False
    found = [row for row in measurements()
             if all(row[key] == evidence[key] for key in required[:-1])]
    if evidence["reason"] == "unmeasured-portable":
        return not found
    return (evidence["reason"] == "measured-portable" and len(found) == 1
            and found[0]["route"] == "portable" and found[0]["bit_exact"])
