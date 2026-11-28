#!/usr/bin/env python3
"""Phase XVIII: inspect what the dQ accumulator looks like in SPIR-V.

The claim under audit is *not* "the word `for` disappeared" but "the specialized
body no longer indexes the 128-float accumulator with a runtime value".  This
script reads `spirv-dis` output and reports, per artifact:

* every Function-storage variable whose type is a float array (the candidate
  shape of the dQ accumulator),
* for each such variable, every `OpAccessChain` that addresses it, split into
  accesses whose every index is an `OpConstant` and accesses that use a
  runtime value,
* the distinct index ids/values used by the constant accesses,
* Private-storage variables (what a driver may create *after* this stage; the
  SPIR-V itself cannot prove scratch usage either way), and
* basic totals so the size change is attributable.

This is static SPIR-V evidence only.  It says what the module asks the IHV
compiler for.  It does not prove VGPR residency: only the vendor ISA dump can,
and that toolchain is not present here (see README section 3).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DIS_RE = re.compile(r"^\s*(%[\w]+)\s*=\s*(.*)$")
ID_RE = re.compile(r"%[\w]+")


def parse(text: str) -> dict:
    """Return id -> (result_type, opcode, operands) for every defining line.

    `spirv-dis` writes `%id = OpName <result-type> <operands...>`; type and
    constant declarations have no result type, so their first operand is the
    element/literal instead.
    """
    defs: dict[str, tuple[str | None, str, list[str]]] = {}
    for line in text.splitlines():
        line = line.split(";", 1)[0].strip()
        match = DIS_RE.match(line)
        if not match:
            continue
        result_id, body = match.group(1), match.group(2)
        tokens = body.split()
        if tokens[0].startswith("OpType") or tokens[0].startswith("OpConstant"):
            defs[result_id] = (None, tokens[0], tokens[1:])
            continue
        if len(tokens) < 2:
            continue
        opcode, result_type = tokens[0], tokens[1]
        defs[result_id] = (result_type, opcode, tokens[2:])
    return defs


def array_of_float(defs: dict) -> set[str]:
    """Ids of OpTypeArray whose element type resolves to float."""
    floats = {i for i, (_, op, args) in defs.items() if op == "OpTypeFloat"}
    return {
        i
        for i, (_, op, args) in defs.items()
        if op == "OpTypeArray" and args and args[0] in floats
    }


def pointers_to(defs: dict, target: set[str], storage: str) -> set[str]:
    out = set()
    for i, (_, op, args) in defs.items():
        if op == "OpTypePointer" and len(args) >= 2 and args[0] == storage and args[1] in target:
            out.add(i)
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: analyze_spirv.py <dump.txt> [label]", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else path.stem
    text = path.read_text(encoding="utf-8", errors="replace")
    defs = parse(text)

    arrays = array_of_float(defs)
    fn_ptrs = pointers_to(defs, arrays, "Function")
    priv_ptrs = pointers_to(defs, arrays, "Private")

    variables = {
        i: defs[i][2]
        for i, (rtype, op, _args) in defs.items()
        if op == "OpVariable" and rtype in (fn_ptrs | priv_ptrs)
    }

    # Resolve array lengths for reporting.
    lengths = {}
    for array_id in arrays:
        args = defs[array_id][2]
        if len(args) >= 2 and args[1] in defs:
            const_args = defs[args[1]][2]
            if const_args:
                lengths[array_id] = const_args[-1]

    def pointee(pointer_id: str) -> str:
        args = defs.get(pointer_id, (None, "", []))[2]
        return args[1] if len(args) >= 2 else "?"

    chains = {
        i: (rtype, args)
        for i, (rtype, op, args) in defs.items()
        if op == "OpAccessChain" and args and args[0] in variables
    }

    per_variable: dict[str, dict] = {}
    for var_id, var_args in variables.items():
        pointer_type = defs[var_id][0] or "?"
        per_variable[var_id] = {
            "storage": var_args[0] if var_args else "?",
            "array": pointee(pointer_type),
            "constant": 0,
            "dynamic": 0,
            "constant_indices": [],
            "dynamic_index_sources": [],
        }

    for chain_id, (rtype, args) in chains.items():
        var = args[0]
        entry = per_variable.get(var)
        if entry is None:
            continue
        indices = args[1:]
        resolved = [defs.get(i) for i in indices]
        kinds = [d[1] if d else "?" for d in resolved]
        if all(kind == "OpConstant" for kind in kinds):
            entry["constant"] += 1
            values = [d[2][-1] for d in resolved if d]
            entry["constant_indices"].append(",".join(values))
        else:
            entry["dynamic"] += 1
            entry["dynamic_index_sources"].append(
                ",".join(f"{i}:{d[1] if d else '?'}" for i, d in zip(indices, resolved))
            )

    total_dynamic_chains = sum(
        1
        for _, (rtype, op, args) in defs.items()
        if op == "OpAccessChain" and any(defs.get(i, ("", "", []))[1] != "OpConstant" for i in args[1:])
    )

    print(f"artifact={label}")
    print(f"file={path.name}")
    print(f"instructions_total={sum(1 for line in text.splitlines() if ' = ' in line)}")
    print(f"function_storage_float_arrays={len(arrays)}")
    print(f"private_storage_float_array_pointers={len(priv_ptrs)}")
    print(f"function_storage_array_variables={len(variables)}")
    print(f"access_chains_with_any_runtime_index_module_wide={total_dynamic_chains}")
    for var_id, entry in sorted(per_variable.items(), key=lambda kv: -kv[1]["dynamic"]):
        length = lengths.get(entry["array"], "?")
        print(
            f"  variable={var_id} storage={entry['storage']} array={entry['array']}"
            f"[{length}] constant_access_chains={entry['constant']}"
            f" runtime_index_access_chains={entry['dynamic']}"
        )
        if entry["constant"]:
            distinct = sorted({int(v) for v in entry["constant_indices"]})
            print(
                f"    constant_index_count={len(entry['constant_indices'])}"
                f" distinct_constants={len(distinct)}"
                f" range={distinct[0]}..{distinct[-1]}" if distinct else ""
            )
        for source in sorted(set(entry["dynamic_index_sources"]))[:6]:
            print(f"    runtime_index_sources={source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
