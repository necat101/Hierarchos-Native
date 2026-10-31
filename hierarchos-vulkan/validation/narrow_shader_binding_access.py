#!/usr/bin/env python3
"""Narrow storage-buffer access qualifiers where SPIR-V already proves them.

`audit_shader_binding_access.py` reports bindings whose compiled module never
writes (or never reads) a buffer while the GLSL source still declares the block
read-write. Those declarations are what make the dispatch scheduler insert a
barrier it does not need, because an unqualified block is conservatively
read-write.

This tool closes that gap safely:

  1. it resolves each affected artifact to its GLSL source and to the build
     command that produced it, and proves that command reproduces the checked-in
     artifact byte-for-byte before touching anything,
  2. it inserts `readonly`/`writeonly` into the affected declaration once per
     source, covering every variant artifact built from that source,
  3. it rebuilds every affected artifact, requires `spirv-val` to accept it, and
     proves the rebuild changed only decorations: the set of buffers the module
     reads and writes must be identical to the previous artifact, and the
     reported declaration must now match the proof.

Commands come from the docs that record them (`README.md`, `VENDOR_TUNING.md`).
For an artifact no document covers, the tool probes the compiler invocations the
repository normally uses and accepts only one that reproduces the artifact
byte-for-byte, so compiler and flags are never guessed silently.

Without `--apply` the tool reports what it would change and exits.

Usage:
  narrow_shader_binding_access.py [--shader-dir DIR] [--apply] [--only SHADER ...]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit_shader_binding_access import ModuleUsage, audit  # noqa: E402

DOCUMENTED_COMMAND = re.compile(
    r"^\s*(?P<tool>[\w./\\-]*(?:glslc|glslangValidator))\s+"
    r"(?P<args>.*?)-o\s+(?P<output>shaders/[^\s]+\.spv)\s*$"
)
SOURCE_IN_COMMAND = re.compile(r"([\w./\\-]+\.comp)")
DOCUMENTED_SOURCE = re.compile(r"hs/[\w./\\-]+\.comp")

TARGET_ENVS = ("", "--target-env=vulkan1.0", "--target-env=vulkan1.1", "--target-env=vulkan1.2")
DEFINE = re.compile(r"-D([A-Za-z_][A-Za-z0-9_]*)(?:=(\S+))?")
NUMBER_IN_NAME = {
    "tile": re.compile(r"_tile(\d+)"),
    "rows": re.compile(r"reuse(\d+)"),
    "wg": re.compile(r"_wg(\d+)"),
    "lanes": re.compile(r"lanes(\d+)"),
}
VERSION_TAG = re.compile(r"(?:reuse|tile|wg|lanes)\d+")

DECLARATION = re.compile(
    r"^(?P<indent>\s*)layout\s*\((?P<layout>[^)]*)\)\s*"
    r"(?P<qualifiers>(?:\w+\s+)*?)(?P<kind>buffer|uniform)\s+"
    r"(?P<name>\w+)\s*(?P<terminator>[;{])",
    re.MULTILINE,
)
BINDING_INDEX = re.compile(r"binding\s*=\s*(\d+)")
SET_INDEX = re.compile(r"set\s*=\s*(\d+)")


class BuildRecipe:
    """How one artifact is produced from one GLSL source."""

    def __init__(self, artifact: str, source: str, command: str, documented: bool):
        self.artifact = artifact
        self.source = source
        self.command = command
        self.documented = documented


def documented_recipes(crate_dir: str, docs: list[str]) -> dict[str, tuple[str, str]]:
    recipes: dict[str, tuple[str, str]] = {}
    for doc in docs:
        path = os.path.join(crate_dir, doc)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = DOCUMENTED_COMMAND.match(line.rstrip("\n"))
                if match is None:
                    continue
                sources = SOURCE_IN_COMMAND.findall(match.group("args"))
                if len(sources) != 1:
                    continue
                artifact = os.path.basename(match.group("output"))
                command = (
                    f"{match.group('tool')} {match.group('args').strip()} "
                    f"-o {match.group('output')}"
                )
                recipes[artifact] = (sources[0].replace("\\", "/"), command)
    return recipes


def run_command(crate_dir: str, command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, cwd=crate_dir, shell=True, capture_output=True, text=True
    )


def artifact_path(crate_dir: str, artifact: str) -> str:
    return os.path.join(crate_dir, "shaders", artifact)


def reproduces(crate_dir: str, recipe: BuildRecipe) -> bool:
    """True when the recipe rebuilds the checked-in artifact byte-for-byte."""
    with tempfile.TemporaryDirectory() as scratch:
        output = os.path.join(scratch, recipe.artifact)
        command = f"{recipe.command.rsplit('-o ', 1)[0]}-o \"{output}\""
        completed = run_command(crate_dir, command)
        if completed.returncode != 0:
            return False
        if completed.stderr.strip():
            return False
        with open(output, "rb") as rebuilt, open(
            artifact_path(crate_dir, recipe.artifact), "rb"
        ) as checked_in:
            return rebuilt.read() == checked_in.read()


def define_for(macro: str, artifact: str, fallback: str | None) -> str | None:
    """Pick the value an artifact-name version tag implies for a macro."""
    upper = macro.upper()
    if "TILE" in upper:
        match = NUMBER_IN_NAME["tile"].search(artifact)
        return match.group(1) if match else fallback
    if "ROWS" in upper or "REUSE" in upper:
        match = NUMBER_IN_NAME["rows"].search(artifact)
        return match.group(1) if match else fallback
    if "WG" in upper or "WORKGROUP" in upper or "LANE" in upper:
        for key in ("wg", "lanes"):
            match = NUMBER_IN_NAME[key].search(artifact)
            if match:
                return match.group(1)
        return fallback
    return fallback


CONDITIONAL_MACRO = re.compile(r"#\s*if(?:ndef)?\s+(?:defined\s*\()?([A-Za-z_]\w*)")


def source_macros(crate_dir: str, source: str) -> list[str]:
    """Tunable macros a shader source defaults with a preprocessor guard."""
    path = os.path.join(crate_dir, source)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    return sorted(set(CONDITIONAL_MACRO.findall(text)) - {"defined"})


def candidate_commands(
    crate_dir: str, artifact: str, source: str, documented: dict[str, tuple[str, str]]
) -> list[str]:
    """Bounded set of plausible build commands for one artifact."""
    siblings = sorted(
        {command for _, (sibling, command) in documented.items() if sibling == source}
    )
    if not siblings:
        siblings = [
            f"glslc {source}",
            f"glslangValidator -V --target-env vulkan1.0 -Ishaders {source}",
        ]
    commands: list[str] = []
    for command in siblings:
        prefix = command.partition("-o ")[0].strip()
        defines = DEFINE.findall(prefix)
        # A variant artifact is often the same source with a different tunable.
        # Try the sibling's own defines first, then every name-implied value for
        # the macros the source guards, since those are the knobs a variant can
        # only have changed.
        extra_names = sorted(
            {name for name in source_macros(crate_dir, source)}
            | {macro for macro, _ in defines}
        )
        for use_extras in (False, True):
            for target_env in TARGET_ENVS:
                variant = prefix
                for macro, value in defines:
                    narrow = define_for(macro, artifact, value)
                    if narrow != value:
                        variant = variant.replace(
                            f"-D{macro}={value}", f"-D{macro}={narrow}"
                        )
                if use_extras:
                    for macro in extra_names:
                        if re.search(rf"-D{macro}(?:=|\s|$)", variant):
                            continue
                        narrow = define_for(macro, artifact, None)
                        if narrow:
                            variant = f"{variant} -D{macro}={narrow}"
                variant = re.sub(r"--target-env=\S+", target_env, variant)
                if target_env and "--target-env" not in variant:
                    variant = f"{variant} {target_env}"
                commands.append(f"{variant} -o shaders/{artifact}")
    return commands


def resolve_recipe(
    crate_dir: str, artifact: str, documented: dict[str, tuple[str, str]]
) -> BuildRecipe:
    name = artifact[: -len(".spv")]
    # A variant artifact shares a source whose own name is a prefix of it, with
    # any version tag (`reuse8`, `tile64`, `wg32`) stripped from that prefix.
    def source_prefix(source: str) -> str:
        stem = source.rsplit("/", 1)[-1]
        if stem.endswith(".comp"):
            stem = stem[: -len(".comp")]
        return VERSION_TAG.split(stem)[0]

    sources = sorted(
        {
            source
            for source, _ in documented.values()
            if name.startswith(source_prefix(source))
        },
        key=len,
        reverse=True,
    )
    seen: set[str] = set()
    for source in sources + [f"shaders/{name}.comp"]:
        if source in seen:
            continue
        seen.add(source)
        if not os.path.exists(os.path.join(crate_dir, source)):
            continue
        for command in candidate_commands(crate_dir, artifact, source, documented):
            recipe = BuildRecipe(artifact, source, command, artifact in documented)
            if reproduces(crate_dir, recipe):
                if artifact in documented and command != documented[artifact][1]:
                    # The docs name a command that no longer reproduces the
                    # artifact; keep the working one but say so.
                    recipe.documented = False
                return recipe
    raise RuntimeError(
        f"no compiler invocation reproduces {artifact}; its build command is not recorded"
    )


def usage_signature(path: str) -> tuple[frozenset[int], frozenset[int]]:
    module = ModuleUsage(path)
    reads = frozenset(
        binding
        for binding in module.bindings
        if module.proven_access(binding) in ("readonly", "may-write")
    )
    writes = frozenset(
        binding
        for binding in module.bindings
        if module.proven_access(binding) in ("writeonly", "may-write")
    )
    return reads, writes


def validation_target_env(path: str) -> str:
    """The spirv-val environment that accepts this module's SPIR-V version."""
    import struct

    with open(path, "rb") as handle:
        version = struct.unpack("<I", handle.read(8)[4:8])[0]
    if version <= 0x0001_0000:
        return "vulkan1.0"
    if version <= 0x0001_0300:
        return "vulkan1.1"
    if version <= 0x0001_0500:
        return "vulkan1.2"
    return "vulkan1.3"


def narrow_source(path: str, binding: int, qualifier: str) -> bool:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        text = handle.read()
    for match in DECLARATION.finditer(text):
        declared = BINDING_INDEX.search(match.group("layout"))
        set_index = SET_INDEX.search(match.group("layout"))
        if declared is None or int(declared.group(1)) != binding:
            continue
        if set_index is not None and set_index.group(1) != "0":
            continue
        existing = match.group("qualifiers").split()
        if qualifier in existing:
            return False
        qualifiers = " ".join([qualifier, *existing])
        replacement = (
            f"{match.group('indent')}layout({match.group('layout')}) {qualifiers} "
            f"{match.group('kind')} {match.group('name')} {match.group('terminator')}"
        )
        text = text[: match.start()] + replacement + text[match.end() :]
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        return True
    raise RuntimeError(f"{path}: no set-0 declaration for binding {binding}")


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shader-dir", default=os.path.join(here, "..", "shaders"))
    parser.add_argument("--crate-dir", default=os.path.join(here, ".."))
    parser.add_argument(
        "--docs",
        nargs="*",
        default=["README.md", "VENDOR_TUNING.md", "COMPATIBILITY.md"],
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="limit to these shaders")
    args = parser.parse_args()
    shader_dir = os.path.abspath(args.shader_dir)
    crate_dir = os.path.abspath(args.crate_dir)
    documented = documented_recipes(crate_dir, args.docs)

    report = audit(shader_dir)
    gaps = [
        entry
        for entry in report["gaps"]
        if args.only is None or entry["shader"] in set(args.only)
    ]
    if not gaps:
        print("every binding already declares access the module proves")
        return 0

    # Group the requested narrowings by the source file that owns them: one edit
    # has to serve every variant artifact compiled from that source.
    by_source: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for entry in sorted(gaps, key=lambda entry: entry["shader"]):
        artifact = f"{entry['shader']}.spv"
        try:
            recipe = resolve_recipe(crate_dir, artifact, documented)
        except RuntimeError as error:
            failures.append(f"{entry['shader']}: {error}")
            continue
        record = by_source.setdefault(
            recipe.source,
            {"edits": {}, "recipes": [], "bindings": {}},
        )
        record["recipes"].append(recipe)  # type: ignore[union-attr]
        for binding in entry["bindings"]:
            index = binding["binding"]
            previous = record["bindings"].get(index)  # type: ignore[union-attr]
            if previous is not None and previous != binding["proven"]:
                failures.append(
                    f"{entry['shader']}: binding {index} proves {binding['proven']} "
                    f"but another variant of {recipe.source} proves {previous}"
                )
                continue
            record["bindings"][index] = binding["proven"]  # type: ignore[index]

    print(
        f"{len(by_source)} source file(s) carry the {len(gaps) - len(failures)} "
        f"requested shader narrowing(s)"
    )
    applied = 0
    for source, record in sorted(by_source.items()):
        recipes: list[BuildRecipe] = record["recipes"]  # type: ignore[assignment]
        bindings: dict[int, str] = record["bindings"]  # type: ignore[assignment]
        source_path = os.path.join(crate_dir, source)
        if not os.path.exists(source_path):
            failures.append(f"{source}: missing source file")
            continue
        if not args.apply:
            detail = ", ".join(f"{index}:{kind}" for index, kind in sorted(bindings.items()))
            print(f"  would narrow {source}: {detail}")
            applied += 1
            continue
        before = {
            recipe.artifact: usage_signature(artifact_path(crate_dir, recipe.artifact))
            for recipe in recipes
        }
        backup = f"{source_path}.bak"
        shutil.copyfile(source_path, backup)
        try:
            edits = [
                f"binding {index} -> {kind}"
                for index, kind in sorted(bindings.items())
                if narrow_source(source_path, index, kind)
            ]
            if not edits:
                os.remove(backup)
                continue
            for recipe in recipes:
                completed = run_command(crate_dir, recipe.command)
                if completed.returncode != 0 or completed.stderr.strip():
                    raise RuntimeError(
                        f"rebuilding {recipe.artifact} failed:\n"
                        f"{completed.stdout}{completed.stderr}"
                    )
            for recipe in recipes:
                path = artifact_path(crate_dir, recipe.artifact)
                validated = run_command(
                    crate_dir,
                    "spirv-val --target-env "
                    f"{validation_target_env(path)} shaders/{recipe.artifact}",
                )
                if validated.returncode != 0:
                    raise RuntimeError(
                        f"spirv-val rejected {recipe.artifact}:\n"
                        f"{validated.stdout}{validated.stderr}"
                    )
                if before[recipe.artifact] != usage_signature(path):
                    raise RuntimeError(
                        f"{recipe.artifact}: rebuild changed buffer usage, not only "
                        f"access decorations"
                    )
            rebuilt = ModuleUsage(artifact_path(crate_dir, recipes[0].artifact))
            unclosed = [
                index
                for index, kind in bindings.items()
                if rebuilt.declared_access(index) != kind
            ]
            if unclosed:
                raise RuntimeError(
                    f"rebuilt module still declares read-write access for {unclosed}"
                )
            print(f"  {source}: {', '.join(edits)} across {len(recipes)} artifact(s)")
            applied += 1
        except RuntimeError as error:
            shutil.copyfile(backup, source_path)
            for recipe in recipes:
                run_command(crate_dir, recipe.command)
            failures.append(f"{source}: {error}")
        finally:
            if os.path.exists(backup):
                os.remove(backup)

    print(f"{applied} source file(s) {'updated' if args.apply else 'pending'}")
    for failure in failures:
        print(f"  FAILED {failure}")
    if args.apply:
        remaining = audit(shader_dir)
        print(
            f"remaining provably narrower binding(s): {remaining['gap_count']} "
            f"over {remaining['gap_shader_count']} shader(s)"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
