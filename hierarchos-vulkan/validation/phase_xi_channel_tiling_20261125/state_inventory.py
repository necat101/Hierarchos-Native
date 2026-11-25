#!/usr/bin/env python3
"""Per-artifact state inventory derived from the artifact, not from the source.

`spirv-dis` is run over every SPIR-V this phase produces and over the production
key shader, and the report prints what the module actually declares:

  * function-scope array variables and their element counts/bytes, which is the
    per-invocation state the shader asks the compiler to allocate;
  * access chains into those arrays with a *variable* index, which is what keeps
    the array in memory instead of in registers (a constant index can be
    scalar-replaced, a dynamic one cannot);
  * loop count and total instruction count.

These are source-derived structural facts. They are not VGPR counts, spill
counts, occupancy or hardware utilisation, and the report says so.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ARRAY = re.compile(r"%(\S+) = OpTypeArray %float %uint_(\d+)")
PTR_FUNCTION_ARRAY = re.compile(r"%(\S+) = OpTypePointer Function %(\S+)")
VAR = re.compile(r"%(\S+) = OpVariable %(\S+) Function")
ACCESS = re.compile(r"%(\S+) = OpAccessChain %(\S+) %(\S+) %(\S+)")


def inventory(path: Path, spirv_dis: str) -> dict:
    text = subprocess.run(
        [spirv_dis, str(path)], capture_output=True, text=True, check=True
    ).stdout
    lines = text.splitlines()

    array_lengths: dict[str, int] = {}
    for line in lines:
        match = ARRAY.search(line)
        if match:
            array_lengths[match.group(1)] = int(match.group(2))

    pointer_targets: dict[str, str] = {}
    for line in lines:
        match = PTR_FUNCTION_ARRAY.search(line)
        if match:
            pointer_targets[match.group(1)] = match.group(2)

    arrays = []
    for line in lines:
        match = VAR.search(line)
        if not match:
            continue
        variable, pointer = match.groups()
        target = pointer_targets.get(pointer)
        if target is None or target not in array_lengths:
            continue
        arrays.append(
            {
                "variable": variable,
                "elements": array_lengths[target],
                "bytes": array_lengths[target] * 4,
            }
        )

    array_variables = {entry["variable"] for entry in arrays}
    dynamic = 0
    static = 0
    for line in lines:
        match = ACCESS.search(line)
        if not match:
            continue
        _result, _pointer, base, index = match.groups()
        # The base operand names the variable, so only chains rooted at one of
        # the function-scope arrays are counted. A constant index shows up as
        # `%uint_3` (an OpConstant); a dynamic one as a plain result id.
        if base not in array_variables:
            continue
        if re.fullmatch(r"%uint_\d+", index):
            static += 1
        else:
            dynamic += 1

    return {
        "artifact": path.name,
        "function_array_variables": len(arrays),
        "function_array_bytes": sum(entry["bytes"] for entry in arrays),
        "arrays": arrays,
        "dynamic_function_array_access_chains": dynamic,
        "constant_function_array_access_chains": static,
        "loops": sum(1 for line in lines if "OpLoopMerge" in line),
        "instructions": sum(1 for line in lines if re.search(r"\bOp[A-Z]", line)),
        "scalar_float_storage_arrays": sum(
            1 for length in array_lengths.values() if length > 1
        ),
    }


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: state_inventory.py SPIRV_DIS ARTIFACT...", file=sys.stderr)
        return 2
    spirv_dis = argv[1]
    entries = [inventory(Path(name), spirv_dis) for name in argv[2:]]
    print(
        json.dumps(
            {
                "note": (
                    "Structural facts read out of each artifact. Function-scope "
                    "arrays with dynamic indices are the per-invocation state "
                    "that cannot be scalar-replaced; their size is not a VGPR "
                    "count, and none of this is a measurement of occupancy or "
                    "hardware utilisation."
                ),
                "artifacts": entries,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
