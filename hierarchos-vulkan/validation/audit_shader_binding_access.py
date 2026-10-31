#!/usr/bin/env python3
"""Prove per-binding storage usage from SPIR-V and compare it with the access
qualifiers the GLSL source declares.

The dispatch scheduler can only elide a compute-to-compute barrier when it
knows a binding cannot be written (or cannot be read). That knowledge comes
either from an explicit GLSL `readonly`/`writeonly` qualifier, which the
compiler turns into a `NonWritable`/`NonReadable` decoration, or from nothing at
all -- an unqualified block is conservatively read-write and forces a barrier
against every earlier dispatch that touched the same buffer region.

This tool reads the compiled module as ground truth: a binding that no
`OpStore`, atomic, or `OpImageWrite` can reach is provably never written, and a
binding no `OpLoad`, atomic, or `OpImageRead` reaches is provably never read.
Reporting the gap between proof and declaration points at exactly the shaders
where a qualifier would let the scheduler skip work.

Usage:
  audit_shader_binding_access.py [--shader-dir DIR] [--json OUT] [--check]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from collections import defaultdict

STORAGE_UNIFORM = 2
STORAGE_STORAGE_BUFFER = 12
STORAGE_CLASSES = {STORAGE_UNIFORM, STORAGE_STORAGE_BUFFER}

DECORATION_NON_WRITABLE = 24
DECORATION_NON_READABLE = 25
DECORATION_BINDING = 33
DECORATION_SET = 34

OP_FUNCTION = 54
OP_FUNCTION_PARAMETER = 55
OP_FUNCTION_CALL = 57
OP_VARIABLE = 59
OP_LOAD = 61
OP_STORE = 62
OP_ACCESS_CHAIN = 65
OP_IN_BOUNDS_ACCESS_CHAIN = 66
OP_PTR_ACCESS_CHAIN = 67
OP_DECORATE = 71
OP_MEMBER_DECORATE = 72
OP_COPY_OBJECT = 83
OP_IMAGE_READ = 98
OP_IMAGE_WRITE = 99
OP_BITCAST = 124
OP_SELECT = 169
OP_ATOMIC_LOAD = 227
OP_ATOMIC_STORE = 228
ATOMIC_POINTER_AT_INDEX_2 = set(range(229, 243))
OP_PHI = 245

DERIVING_OPS = {
    OP_ACCESS_CHAIN,
    OP_IN_BOUNDS_ACCESS_CHAIN,
    OP_PTR_ACCESS_CHAIN,
    OP_COPY_OBJECT,
    OP_BITCAST,
    OP_SELECT,
    OP_PHI,
}


def read_words(path: str) -> list[int]:
    with open(path, "rb") as handle:
        data = handle.read()
    if len(data) % 4:
        raise ValueError(f"{path}: SPIR-V byte length is not a multiple of four")
    return list(struct.unpack(f"<{len(data) // 4}I", data))


def instructions(words: list[int]):
    if len(words) < 5 or words[0] != 0x0723_0203:
        raise ValueError("not a SPIR-V module")
    offset = 5
    while offset < len(words):
        header = words[offset]
        count = header >> 16
        opcode = header & 0xFFFF
        if count == 0 or offset + count > len(words):
            raise ValueError("malformed SPIR-V instruction stream")
        yield opcode, words[offset + 1 : offset + count]
        offset += count


class ModuleUsage:
    """Per-binding read/write proof plus the decorations the module carries."""

    def __init__(self, path: str):
        words = read_words(path)
        decorated_non_writable: set[int] = set()
        decorated_non_readable: set[int] = set()
        bindings: dict[int, int] = {}
        sets: dict[int, int] = {}
        variables: dict[int, int] = {}  # id -> storage class
        derivations: list[tuple[int, int]] = []  # derived id -> base id
        function_parameters: dict[int, list[int]] = defaultdict(list)
        function_calls: list[tuple[int, list[int]]] = []
        read_ids: set[int] = set()
        written_ids: set[int] = set()
        current_function = None

        for opcode, operands in instructions(words):
            if opcode == OP_FUNCTION:
                # result type, result, control, function type
                current_function = operands[1]
            elif opcode == OP_FUNCTION_PARAMETER and current_function is not None:
                # type, result
                function_parameters[current_function].append(operands[1])
            elif opcode == OP_VARIABLE and len(operands) >= 3:
                if operands[2] in STORAGE_CLASSES:
                    variables[operands[1]] = operands[2]
            elif opcode == OP_DECORATE and len(operands) >= 2:
                target = operands[0]
                decoration = operands[1]
                if decoration == DECORATION_NON_WRITABLE:
                    decorated_non_writable.add(target)
                elif decoration == DECORATION_NON_READABLE:
                    decorated_non_readable.add(target)
                elif decoration == DECORATION_BINDING and len(operands) >= 3:
                    bindings[target] = operands[2]
                elif decoration == DECORATION_SET and len(operands) >= 3:
                    sets[target] = operands[2]
            elif opcode == OP_LOAD and len(operands) >= 3:
                read_ids.add(operands[2])
            elif opcode == OP_STORE and len(operands) >= 1:
                written_ids.add(operands[0])
            elif opcode == OP_ATOMIC_LOAD and len(operands) >= 3:
                read_ids.add(operands[2])
            elif opcode == OP_ATOMIC_STORE and len(operands) >= 1:
                written_ids.add(operands[0])
            elif opcode in ATOMIC_POINTER_AT_INDEX_2 and len(operands) >= 3:
                read_ids.add(operands[2])
                written_ids.add(operands[2])
            elif opcode == OP_IMAGE_READ and len(operands) >= 4:
                read_ids.add(operands[2])
            elif opcode == OP_IMAGE_WRITE and len(operands) >= 1:
                written_ids.add(operands[0])
            elif opcode in DERIVING_OPS and len(operands) >= 3:
                base = operands[2]
                if opcode == OP_PHI:
                    for index in range(2, len(operands), 2):
                        derivations.append((operands[1], operands[index]))
                else:
                    derivations.append((operands[1], base))
            elif opcode == OP_FUNCTION_CALL and len(operands) >= 3:
                function_calls.append((operands[2], list(operands[3:])))

        # Bind call arguments to callee parameters so a helper that stores
        # through a pointer parameter still attributes the write to the caller's
        # variable.
        for callee, arguments in function_calls:
            parameters = function_parameters.get(callee, [])
            for parameter, argument in zip(parameters, arguments):
                derivations.append((parameter, argument))

        # Propagate each access to every base it can reach. The derivation graph
        # is acyclic in practice, but iterate to a fixpoint so a chain of
        # access chains and helper calls resolves in one pass anyway.
        base_of = {derived: base for derived, base in derivations}
        reachable: dict[int, set[int]] = {}

        def roots(identifier: int, seen: frozenset[int] = frozenset()) -> set[int]:
            cached = reachable.get(identifier)
            if cached is not None:
                return cached
            base = base_of.get(identifier)
            if base is None or identifier in seen:
                result = {identifier}
            else:
                result = roots(base, seen | {identifier})
            reachable[identifier] = result
            return result

        self.read_roots: set[int] = set()
        self.write_roots: set[int] = set()
        for identifier in read_ids:
            self.read_roots |= roots(identifier)
        for identifier in written_ids:
            self.write_roots |= roots(identifier)

        self.bindings: dict[int, dict[str, object]] = {}
        for identifier, storage in variables.items():
            if sets.get(identifier) != 0:
                continue
            binding = bindings.get(identifier)
            if binding is None:
                continue
            entry = self.bindings.setdefault(
                binding,
                {
                    "storage_class": storage,
                    "read": False,
                    "write": False,
                    "non_writable": False,
                    "non_readable": False,
                },
            )
            entry["read"] = bool(entry["read"]) or identifier in self.read_roots
            entry["write"] = bool(entry["write"]) or identifier in self.write_roots
            entry["non_writable"] = bool(entry["non_writable"]) or (
                identifier in decorated_non_writable
            )
            entry["non_readable"] = bool(entry["non_readable"]) or (
                identifier in decorated_non_readable
            )

    def declared_access(self, binding: int) -> str:
        entry = self.bindings.get(binding)
        if entry is None:
            return "unbound"
        if entry["non_writable"] and not entry["non_readable"]:
            return "readonly"
        if entry["non_readable"] and not entry["non_writable"]:
            return "writeonly"
        return "may-write"

    def proven_access(self, binding: int) -> str:
        entry = self.bindings.get(binding)
        if entry is None:
            return "unbound"
        read, write = bool(entry["read"]), bool(entry["write"])
        if read and not write:
            return "readonly"
        if write and not read:
            return "writeonly"
        if read and write:
            return "may-write"
        return "unused"


GLSL_DECLARATION = re.compile(
    r"layout\s*\((?P<layout>[^)]*)\)\s*"
    r"(?P<qualifiers>(?:\w+\s+)*?)"
    r"(?:buffer|uniform)\s+(?P<name>\w+)",
    re.MULTILINE,
)
BINDING_INDEX = re.compile(r"binding\s*=\s*(\d+)")
SET_INDEX = re.compile(r"set\s*=\s*(\d+)")
ACCESS_QUALIFIERS = ("readonly", "writeonly", "restrict")


def source_declarations(path: str) -> dict[int, str]:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    declarations: dict[int, str] = {}
    for match in GLSL_DECLARATION.finditer(text):
        binding = BINDING_INDEX.search(match.group("layout"))
        set_index = SET_INDEX.search(match.group("layout"))
        if binding is None or (set_index is not None and set_index.group(1) != "0"):
            continue
        qualifiers = [word for word in match.group("qualifiers").split() if word in ACCESS_QUALIFIERS]
        declarations[int(binding.group(1))] = " ".join(qualifiers) or "may-write"
    return declarations


def audit(shader_dir: str) -> dict[str, object]:
    modules = []
    gaps = []
    for name in sorted(os.listdir(shader_dir)):
        if not name.endswith(".spv"):
            continue
        stem = name[: -len(".spv")]
        spv_path = os.path.join(shader_dir, name)
        comp_path = os.path.join(shader_dir, f"{stem}.comp")
        try:
            module = ModuleUsage(spv_path)
        except ValueError as error:
            modules.append({"shader": stem, "error": str(error)})
            continue
        declared = source_declarations(comp_path) if os.path.exists(comp_path) else {}
        shader_gaps = []
        for binding in sorted(module.bindings):
            proven = module.proven_access(binding)
            decorated = module.declared_access(binding)
            declared_source = declared.get(binding, "may-write")
            if proven in ("readonly", "writeonly") and decorated != proven:
                shader_gaps.append(
                    {
                        "binding": binding,
                        "proven": proven,
                        "spirv_decorated": decorated,
                        "source_declared": declared_source,
                    }
                )
        modules.append(
            {
                "shader": stem,
                "bindings": {
                    str(binding): {
                        "proven": module.proven_access(binding),
                        "spirv_decorated": module.declared_access(binding),
                        "source_declared": declared.get(binding, "may-write"),
                    }
                    for binding in sorted(module.bindings)
                },
            }
        )
        if shader_gaps:
            gaps.append({"shader": stem, "bindings": shader_gaps})
    return {
        "shader_dir": shader_dir,
        "modules": modules,
        "gaps": gaps,
        "gap_count": sum(len(entry["bindings"]) for entry in gaps),
        "gap_shader_count": len(gaps),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shader-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shaders"),
    )
    parser.add_argument("--json", help="write the full report here")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero when a binding could declare narrower access",
    )
    args = parser.parse_args()
    report = audit(os.path.abspath(args.shader_dir))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    print(
        f"{report['gap_count']} provably narrower binding(s) over "
        f"{report['gap_shader_count']} shader(s) out of {len(report['modules'])} modules"
    )
    for entry in sorted(
        report["gaps"], key=lambda entry: (-len(entry["bindings"]), entry["shader"])
    ):
        summary = ", ".join(
            f"{binding['binding']}:{binding['proven']}(declared {binding['source_declared']})"
            for binding in entry["bindings"]
        )
        print(f"  {entry['shader']}: {summary}")
    if args.check and report["gap_count"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
