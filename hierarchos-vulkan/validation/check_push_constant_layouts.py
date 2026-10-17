"""Compare every compute kernel's push-constant block across the three descriptions
of it: the SPIR-V that runs, the GLSL it was built from, and the host struct whose
bytes are pushed.

A push block is the one kernel ABI the compiler cannot check. Vulkan only requires
the declared push range to cover the bytes the shader reads, so a module built
before a member was added still binds and still runs -- it just ignores the new
word. `transformer_qwen_gdn_params.spv` was exactly that: an 8-byte block behind a
12-byte host struct, silently dropping OLMo Hybrid's `beta_scale` from the forward
kernel while the backward kernel applied it.

The check reports three independent disagreements:

* SPIR-V vs GLSL -- the committed module must declare the members its source does.
  This is the one that catches a stale artifact.
* declared range vs SPIR-V -- a kernel created for fewer bytes than its shader
  reads is invalid at dispatch time.
* host struct vs GLSL -- the struct whose `bytes_of` is pushed must line up with
  the shader's members. A struct wider than the block is legal (the shared head
  trainer declares the full loss push block and both vendor and portable modules
  read only its leading geometry words), so the comparison is a prefix match.

Run from `hierarchos-vulkan/`:

    python validation/check_push_constant_layouts.py            # whole crate
    python validation/check_push_constant_layouts.py --verbose  # per-kernel rows
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation" / "training_shader_manifest.json"

OP_MEMBER_NAME = 6
OP_TYPE_INT = 21
OP_TYPE_FLOAT = 22
OP_TYPE_VECTOR = 23
OP_TYPE_MATRIX = 24
OP_TYPE_ARRAY = 28
OP_TYPE_STRUCT = 30
OP_TYPE_POINTER = 32
OP_CONSTANT = 43
OP_VARIABLE = 59
OP_MEMBER_DECORATE = 72

STORAGE_PUSH_CONSTANT = 9
DECORATION_OFFSET = 35

# GLSL type -> (size, alignment) under std430. Vectors of 3 occupy 12 bytes but
# align to 16, and a matNx4 is N vec4 columns.
GLSL_TYPES: dict[str, tuple[int, int]] = {
    "float": (4, 4),
    "int": (4, 4),
    "uint": (4, 4),
    "bool": (4, 4),
    "vec2": (8, 8),
    "ivec2": (8, 8),
    "uvec2": (8, 8),
    "vec3": (12, 16),
    "ivec3": (12, 16),
    "uvec3": (12, 16),
    "vec4": (16, 16),
    "ivec4": (16, 16),
    "uvec4": (16, 16),
    "mat2": (32, 16),
    "mat3": (48, 16),
    "mat4": (64, 16),
}

SPIRV_WIDTHS = {
    OP_TYPE_INT: lambda kind: kind[0] // 8,
    OP_TYPE_FLOAT: lambda kind: kind[0] // 8,
}


@dataclass(frozen=True)
class Member:
    name: str
    offset: int
    size: int

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass(frozen=True)
class PushBlock:
    members: tuple[Member, ...]

    @property
    def size(self) -> int:
        return max((member.end for member in self.members), default=0)

    def describe(self) -> str:
        return ", ".join(f"{member.name}@{member.offset}" for member in self.members)


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class SpirvModule:
    """Just enough of a SPIR-V reader to recover one push-constant block."""

    def __init__(self, data: bytes) -> None:
        self.types: dict[int, tuple[int, ...]] = {}
        self.constants: dict[int, int] = {}
        self.pointee: dict[int, int] = {}
        self.member_names: dict[tuple[int, int], str] = {}
        self.offsets: dict[tuple[int, int], int] = {}
        self.push_variables: list[int] = []
        self._parse(list(struct.unpack(f"<{len(data) // 4}I", data)))

    def _parse(self, words: list[int]) -> None:
        index = 5
        while index < len(words):
            header = words[index]
            count, opcode = header >> 16, header & 0xFFFF
            if count == 0:
                break
            operands = words[index + 1 : index + count]
            index += count
            if opcode in (
                OP_TYPE_INT,
                OP_TYPE_FLOAT,
                OP_TYPE_VECTOR,
                OP_TYPE_MATRIX,
                OP_TYPE_ARRAY,
                OP_TYPE_STRUCT,
            ):
                self.types[operands[0]] = (opcode, *operands[1:])
            elif opcode == OP_TYPE_POINTER:
                self.pointee[operands[0]] = operands[2]
            elif opcode == OP_CONSTANT:
                self.constants[operands[1]] = operands[2] if len(operands) > 2 else 0
            elif opcode == OP_VARIABLE:
                if len(operands) >= 3 and operands[2] == STORAGE_PUSH_CONSTANT:
                    # Keep the variable's pointer *type*: ids are not types.
                    self.push_variables.append(operands[0])
            elif opcode == OP_MEMBER_NAME:
                raw = b"".join(struct.pack("<I", word) for word in operands[2:])
                self.member_names[(operands[0], operands[1])] = raw.split(b"\0", 1)[0].decode(
                    "utf-8", "replace"
                )
            elif opcode == OP_MEMBER_DECORATE:
                if operands[2] == DECORATION_OFFSET:
                    self.offsets[(operands[0], operands[1])] = operands[3]

    def size_of(self, type_id: int, seen: frozenset[int] = frozenset()) -> int | None:
        entry = self.types.get(type_id)
        if entry is None or type_id in seen:
            return None
        opcode, *kind = entry
        if opcode in SPIRV_WIDTHS:
            return SPIRV_WIDTHS[opcode](kind)
        if opcode == OP_TYPE_VECTOR:
            component = self.size_of(kind[0], seen | {type_id})
            return component * kind[1] if component is not None else None
        if opcode == OP_TYPE_MATRIX:
            column = self.size_of(kind[0], seen | {type_id})
            return column * kind[1] if column is not None else None
        if opcode == OP_TYPE_ARRAY:
            element = self.size_of(kind[0], seen | {type_id})
            length = self.constants.get(kind[1])
            return element * length if element is not None and length is not None else None
        if opcode == OP_TYPE_STRUCT:
            sizes = [self.size_of(member, seen | {type_id}) for member in kind]
            return None if any(size is None for size in sizes) else sum(sizes)  # type: ignore[arg-type]
        return None

    def push_block(self) -> PushBlock | None:
        for variable in self.push_variables:
            # `push_variables` holds the variable's pointer type id, which only the
            # pointer table knows about (a pointer id is not a type id).
            struct_id = self.pointee.get(variable)
            entry = self.types.get(struct_id) if struct_id is not None else None
            if entry is None or entry[0] != OP_TYPE_STRUCT:
                continue
            members: list[Member] = []
            for index, member_type in enumerate(entry[1:]):
                offset = self.offsets.get((struct_id, index))
                size = self.size_of(member_type)
                if offset is None or size is None:
                    return None
                members.append(Member(self.member_names.get((struct_id, index), ""), offset, size))
            return PushBlock(tuple(members))
        return None


DIRECTIVE = re.compile(r"^\s*#\s*(\w+)(.*)$")
INCLUDE = re.compile(r'#\s*include\s+"([^"]+)"')


def define_table(flags: list[str]) -> dict[str, str]:
    """The macros a manifest entry's `-D` flags define, as preprocessor values."""
    table: dict[str, str] = {}
    for flag in flags:
        if not flag.startswith("-D"):
            continue
        name, _, value = flag[2:].partition("=")
        table[name] = value or "1"
    return table


def evaluate(expression: str, macros: dict[str, str]) -> bool | None:
    """Evaluate the small subset of `#if` expressions the shaders use.

    Anything outside that subset returns None so the caller can report an
    unreadable block instead of guessing which branches were compiled.
    """
    expression = re.sub(
        r"defined\s*\(\s*(\w+)\s*\)",
        lambda m: "1" if m.group(1) in macros else "0",
        expression,
    )
    identifiers = set(re.findall(r"[A-Za-z_]\w*", expression))
    # An undefined macro is 0 in GLSL preprocessor expressions, which is also how
    # a variant pins a conditional member off: the flag simply is not passed.
    for name in identifiers:
        expression = re.sub(rf"\b{name}\b", macros.get(name, "0") or "0", expression)
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)
    # Only digits, parentheses, comparisons and the three logical operators can
    # remain; anything else means this is not an expression we understand.
    for part in re.split(r"\b(and|or|not)\b", expression):
        if part in ("and", "or", "not", ""):
            continue
        if not re.fullmatch(r"[0-9\s()<>=!&|+\-*/]+", part):
            return None
    try:
        return bool(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 - digits and operators only
    except (SyntaxError, TypeError, ZeroDivisionError):
        return None


def preprocess(path: Path, macros: dict[str, str], stack: tuple[Path, ...] = ()) -> str | None:
    """Expand `#include`s and resolve `#if` branches for one compile.

    A push block can carry conditional members, and its source can live in an
    included file; reading the file as written would compare against whichever
    branch happens to be in the text rather than the one the artifact was built
    with.
    """
    if path in stack:
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    out: list[str] = []
    active = [True]
    # `taken[-1]` records whether some branch of the current `#if` already matched,
    # so `#elif`/`#else` do not re-enable a block a previous branch owned.
    taken = [True]
    local = dict(macros)
    for line in text.splitlines():
        include = INCLUDE.match(line.strip())
        if include is not None and active[-1]:
            target = (path.parent / include.group(1)).resolve()
            if not target.is_file():
                return None
            expanded = preprocess(target, local, stack + (path,))
            if expanded is None:
                return None
            out.append(expanded)
            continue
        directive = DIRECTIVE.match(line)
        if directive is None:
            if active[-1]:
                out.append(line)
            continue
        keyword, rest = directive.group(1), directive.group(2).strip()
        if keyword in ("ifdef", "ifndef", "if"):
            parent = active[-1]
            if keyword == "ifdef":
                value = rest in local
            elif keyword == "ifndef":
                value = rest not in local
            else:
                evaluated = evaluate(rest, local) if parent else False
                if evaluated is None and parent:
                    return None
                value = bool(evaluated)
            active.append(parent and value)
            taken.append(parent and value)
        elif keyword == "elif":
            if len(active) < 2:
                return None
            parent, previous = active[-2], taken[-1]
            evaluated = evaluate(rest, local) if parent and not previous else False
            if evaluated is None and parent and not previous:
                return None
            active[-1] = parent and not previous and bool(evaluated)
            taken[-1] = previous or (parent and bool(evaluated))
        elif keyword == "else":
            if len(active) < 2:
                return None
            active[-1] = active[-2] and not taken[-1]
            taken[-1] = True
        elif keyword == "endif":
            if len(active) < 2:
                return None
            active.pop()
            taken.pop()
        elif keyword == "define" and active[-1]:
            name, _, value = rest.partition(" ")
            local[name.split("(")[0]] = value.strip() or "1"
        elif keyword == "undef" and active[-1]:
            local.pop(rest, None)
        # #version, #extension, #pragma and line markers are irrelevant here.
    return "\n".join(out)


@dataclass(frozen=True)
class GlslMember:
    name: str
    type: str
    count: int


def parse_glsl_push_block(source: str) -> tuple[str, tuple[GlslMember, ...]] | None:
    match = re.search(
        r"layout\s*\(\s*push_constant\s*\)\s*uniform\s+(\w+)\s*\{(?P<body>[^}]*)\}",
        source,
        re.S,
    )
    if match is None:
        return None
    members: list[GlslMember] = []
    for declaration in match.group("body").split(";"):
        declaration = re.sub(r"//[^\n]*", "", declaration).strip()
        if not declaration:
            continue
        tokens = declaration.split()
        if len(tokens) < 2:
            return None
        type_name, name = tokens[0], tokens[-1]
        count = 1
        array = re.match(r"(\w+)\s*\[\s*(\d+)\s*\]$", name)
        if array is not None:
            name, count = array.group(1), int(array.group(2))
        members.append(GlslMember(name, type_name, count))
    return match.group(1), tuple(members)


def glsl_block(members: tuple[GlslMember, ...]) -> tuple[PushBlock, str] | None:
    placed: list[Member] = []
    cursor = 0
    for member in members:
        if member.type not in GLSL_TYPES:
            return None
        size, alignment = GLSL_TYPES[member.type]
        if member.count > 1:
            size, alignment = size * member.count, alignment
        offset = align_up(cursor, alignment)
        placed.append(Member(member.name, offset, size))
        cursor = offset + size
    return PushBlock(tuple(placed)), ""


def host_structs(sources: dict[Path, str]) -> dict[str, list[tuple[str, str]]]:
    structs: dict[str, list[tuple[str, str]]] = {}
    for path, text in sources.items():
        for match in re.finditer(r"(?:#\[repr\(C\)\]\s*)?(?:#\[derive\([^)]*\)\]\s*)*struct\s+(\w+)\s*\{([^}]*)\}", text):
            fields = []
            for field in match.group(2).split(","):
                field = re.sub(r"//[^\n]*", "", field).strip()
                if not field:
                    continue
                name, _, kind = field.partition(":")
                fields.append((name.strip(), kind.strip()))
            if fields:
                structs[match.group(1)] = fields
    return structs


RUST_WIDTHS = {"u32": 4, "i32": 4, "f32": 4, "u64": 8, "i64": 8, "f64": 8, "u16": 2, "i16": 2}


def rust_block(fields: list[tuple[str, str]]) -> tuple[PushBlock, str] | None:
    placed: list[Member] = []
    cursor = 0
    for name, kind in fields:
        if kind not in RUST_WIDTHS:
            return None
        size = RUST_WIDTHS[kind]
        offset = align_up(cursor, size)
        placed.append(Member(name, offset, size))
        cursor = offset + size
    return PushBlock(tuple(placed)), ""


def glsl_for(spv: Path) -> Path | None:
    """The committed source beside an artifact, if it has one."""
    for candidate in (spv.with_suffix(".comp"), spv.with_suffix(".glsl")):
        if candidate.is_file():
            return candidate
    return None


def shader_problems(
    label: str,
    spv: Path,
    glsl: Path,
    macros: dict[str, str] | None = None,
) -> tuple[list[str], bool]:
    """Compare one artifact's push block with the block its GLSL declares.

    Returns `(problems, compared)`: `compared` is False when either side has no
    readable push block, which callers count separately. Member names are the
    descriptive half of the check: glslc emits `OpMemberName` for some artifacts
    and not others, so a module is only compared by name when it carries names at
    all. Count, offsets and widths are unconditional -- they are what a stale
    rebuild actually breaks.
    """
    text = preprocess(glsl, macros or {})
    parsed = parse_glsl_push_block(text) if text is not None else None
    actual = SpirvModule(spv.read_bytes()).push_block()
    if parsed is None and actual is None:
        # A kernel with no push block at all, like the device-side optimizer
        # clock: nothing to compare and nothing at risk.
        return [], False
    if parsed is None or actual is None:
        return [
            f"{label}: {'the module has a push block but ' + glsl.name + ' declares none' if actual else 'the source declares a push block but the module has none'}"
        ], False
    expected = glsl_block(parsed[1])
    if expected is None:
        return [f"{label}: unreadable push block in {glsl.name}"], False
    want, _ = expected
    named = all(member.name for member in actual.members)
    mismatch = len(want.members) != len(actual.members)
    mismatch = mismatch or (named and [m.name for m in want.members] != [m.name for m in actual.members])
    mismatch = mismatch or [m.offset for m in want.members] != [m.offset for m in actual.members]
    mismatch = mismatch or [m.size for m in want.members] != [m.size for m in actual.members]
    if not mismatch:
        return [], True
    return [
        f"{label}: SPIR-V declares [{actual.describe()}] but {glsl.name} declares "
        f"[{want.describe()}]"
    ], True


def host_problems(
    label: str,
    struct_name: str,
    fields: list[tuple[str, str]],
    spv_name: str,
    block: PushBlock,
) -> tuple[list[str], bool]:
    """Compare the host struct whose bytes are pushed with the block it fills.

    A struct wider than the block is legal and documented -- the shared head
    trainer declares the whole loss push block and the vendor and portable modules
    read only its leading geometry words -- so the rule is a prefix match: the
    shader's members must be the first fields of the struct, with the same widths
    and in the same order. The host carrying *fewer* words than the shader reads is
    never legal.
    """
    built = rust_block(fields)
    if built is None:
        return [], False
    host, _ = built
    shader_widths = [member.size for member in block.members]
    host_widths = [member.size for member in host.members]
    if len(host_widths) < len(shader_widths):
        return [
            f"{label}: {struct_name} is {host.size} bytes but {spv_name} reads "
            f"{block.size} -- the host pushes fewer words than the kernel reads"
        ], True
    if host_widths[: len(shader_widths)] != shader_widths:
        return [
            f"{label}: {struct_name} starts with word widths {host_widths[: len(shader_widths)]}, "
            f"but {spv_name} declares {shader_widths}; the leading fields must match in "
            f"order (a wider struct is fine, a different prefix is not)"
        ], True
    return [], True


CREATE_RE = re.compile(
    r"ComputeKernel::new(?:_with_access)?\(\s*device\s*,\s*(?P<spv>[A-Z0-9_]+)\s*,"
    r"(?P<between>[^;]*?)\)\s*\?",
    re.S,
)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    sources = {
        path: path.read_text(encoding="utf-8", errors="replace")
        for path in sorted((ROOT / "src").rglob("*.rs"))
    }
    all_text = "\n".join(sources.values())

    embedded: dict[str, Path] = {}
    for match in re.finditer(r'const\s+([A-Z0-9_]+)_SPV\s*:\s*&\[u8\]\s*=\s*include_bytes!\("([^"]+\.spv)"\)', all_text):
        embedded[match.group(1) + "_SPV"] = (ROOT / "src" / match.group(2)).resolve()

    # A push block can be conditional, so the members are only known for the
    # macros the artifact was pinned with.
    flags_by_artifact: dict[str, list[str]] = {}
    if MANIFEST.is_file():
        for entry in json.loads(MANIFEST.read_text(encoding="utf-8"))["shaders"]:
            flags_by_artifact[entry["artifact"]] = entry.get("flags", [])

    structs = host_structs(sources)
    problems: list[str] = []
    rows = 0

    for spv in sorted((ROOT / "shaders").rglob("*.spv")):
        glsl = glsl_for(spv)
        if glsl is None:
            continue
        key = spv.relative_to(ROOT / "shaders").as_posix()
        found, compared = shader_problems(
            spv.relative_to(ROOT).as_posix(),
            spv,
            glsl,
            define_table(flags_by_artifact.get(key, [])),
        )
        problems.extend(found)
        rows += int(compared)

    typed_sites = 0
    for path, text in sources.items():
        for match in CREATE_RE.finditer(text):
            spv = embedded.get(match.group("spv"))
            if spv is None or not spv.is_file():
                continue
            block = SpirvModule(spv.read_bytes()).push_block()
            if block is None:
                continue
            tail = match.group("between")
            size_match = re.search(r"size_of::<(\w+)>\(\)", tail)
            literal_match = re.search(r",\s*(\d+)\s*$", tail.strip())
            label = path.relative_to(ROOT).as_posix()
            if size_match is not None:
                struct_name = size_match.group(1)
                fields = structs.get(struct_name)
                if fields is None or glsl_for(spv) is None:
                    continue
                found, compared = host_problems(label, struct_name, fields, spv.name, block)
                problems.extend(found)
                typed_sites += int(compared)
            elif literal_match is not None:
                declared = int(literal_match.group(1))
                if declared < block.size:
                    problems.append(
                        f"{label}: {match.group('spv')} declares {declared} push bytes "
                        f"but reads {block.size}"
                    )

    if args.json:
        print(json.dumps({"checked": rows, "typed_sites": typed_sites, "problems": problems}, indent=2))
    else:
        for problem in problems:
            print(f"FAIL {problem}")
        print(f"checked {rows} shader push blocks, {typed_sites} host structs")
    if problems:
        print(f"FAIL: {len(problems)} push-constant layout disagreements")
        return 1
    print("PASS: push-constant blocks agree across SPIR-V, GLSL and host structs")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
