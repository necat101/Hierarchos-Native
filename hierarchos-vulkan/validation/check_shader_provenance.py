"""Fail when a committed SPIR-V artifact no longer rebuilds from its source.

`training_shader_manifest.json` records the compiler, that compiler's release and
the flags for every committed artifact, so this check needs no per-shader
arguments: it recompiles each recorded source into scratch, compares the result
byte-for-byte with the artifact the crate embeds, and exits non-zero on any
difference.

Recording the release per artifact is what makes a tree spanned by more than one
toolchain release expressible. The candidates for a compiler name are the
repeatable `--compiler-path NAME=PATH` overrides, else what PATH resolves: every
artifact is compiled by the candidate that reports the release it records, so one
run verifies artifacts that different releases produced. `--require-version
NAME=VERSION` still pins entries that record no release of their own; naming a
release that a recorded pin disagrees with is a `pin_conflict` (the recorded pin
wins, loudly) rather than a silent override.

`--record-versions` writes the release that reproduced each artifact back into the
manifest, so the pins come from a verified rebuild instead of hand editing.

A toolchain that is installed but is not the pinned release is reported as
`wrong_version` rather than as a byte difference, because a one-release compiler
bump can change an artifact without a source change (see VENDOR_TUNING.md,
*Toolchain pin*): glslang 16.6.0 emits an extra instruction for both attention
kernels, and shaderc 2026.4 rewrites the index arithmetic of two RMSNorm
input-gradient kernels.

Two statuses mean "not checked" and are distinct from a mismatch:
`missing_compiler` fails the run unless that compiler is named in
`--allow-missing`, and `skipped` (an allowed missing compiler) is reported loudly
but does not fail. Anything else -- `drifted`, `compile_failed`, `missing_source`,
`missing_artifact`, `wrong_version`, `pin_conflict` -- fails the run.

    python validation/check_shader_provenance.py
    python validation/check_shader_provenance.py \
        --compiler-path glslangValidator=/opt/glslang-16.5.0/bin/glslangValidator \
        --compiler-path glslc=/opt/shaderc-2023.8/bin/glslc
    python validation/check_shader_provenance.py --record-versions
"""
from __future__ import annotations

import argparse
from collections import namedtuple
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation" / "training_shader_manifest.json"
EXCEPTIONS = ROOT / "validation" / "shader_provenance_exceptions.json"
EMBEDDED = re.compile(r'include_bytes!\(\s*"([^"]+\.spv)"')

# A pin is the release its own compiler reports, so read it from the first line of
# `--version` that carries one: a dotted triple (`Glslang Version: 11:16.5.0`), a
# `v`-prefixed release (`shaderc v2023.8 unknown hash, 2025-08-31`), or a bare pair
# (`SPIRV-Tools 2025.5 2025.5, 0`). Later lines name components rather than the
# driver -- an embedded glslang release inside glslc, say -- and are not considered.
RELEASE_PATTERNS = (
    re.compile(r"\b(\d+\.\d+\.\d+)\b"),
    re.compile(r"\bv(\d+\.\d+(?:\.\d+)?)\b"),
    re.compile(r"\b(\d+\.\d+)\b"),
)

# Statuses that do not fail the run.
OK_STATUSES = frozenset({"identical", "skipped"})

# Manifest key order, so recording a release does not reshuffle the file.
KEY_ORDER = (
    "artifact",
    "source",
    "compiler",
    "compiler_version",
    "flags",
    "target_env",
    "storage_policy",
)

Toolchain = namedtuple("Toolchain", "path version reported")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_version(reported: str) -> str | None:
    """Extract the release a compiler reports about itself.

    Only the first line that carries a release is considered: that is the line each
    driver here prints its own release on, and stopping there keeps an embedded
    component's release from being mistaken for the toolchain's.
    """
    for line in reported.splitlines():
        if not line.strip():
            continue
        for pattern in RELEASE_PATTERNS:
            matches = pattern.findall(line)
            if matches:
                return matches[-1]
        return None
    return None


def probe(path: str) -> Toolchain:
    """Ask one binary what it is."""
    try:
        banner = subprocess.check_output([path, "--version"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.SubprocessError) as error:  # pragma: no cover - environment
        return Toolchain(path, None, f"<unavailable: {error}>")
    first = next((line.strip() for line in banner.splitlines() if line.strip()), "")
    return Toolchain(path, parse_version(banner), first)


def parse_compiler_paths(pairs, error) -> dict[str, list[str]]:
    """Turn repeated `NAME=PATH` arguments into per-name candidate lists."""
    overrides: dict[str, list[str]] = {}
    for pair in pairs:
        name, _, path = pair.partition("=")
        if not path:
            error(f"expected NAME=PATH, got {pair!r}")
        overrides.setdefault(name, []).append(path)
    return overrides


def embedded_artifacts(root: Path) -> set[str]:
    """Every committed SPIR-V module the crate pulls in through `include_bytes!`.

    Paths come back relative to the crate root with forward slashes, e.g.
    `shaders/linear_forward.spv`, which is also how the manifest addresses them
    (minus the `shaders/` prefix, which the checker adds).
    """
    found: set[str] = set()
    for source in (root / "src").rglob("*.rs"):
        text = source.read_text(encoding="utf-8", errors="ignore")
        for match in EMBEDDED.finditer(text):
            target = (source.parent / match.group(1)).resolve()
            found.add(Path(os.path.relpath(target, root)).as_posix())
    return found


def read_exceptions(path: Path) -> dict[str, str]:
    """Artifact -> reason for the modules no committed source reproduces."""
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    return {
        entry["artifact"]: entry.get("reason", "")
        for entry in document.get("modules", [])
        if "artifact" in entry
    }


def discover_local(root: Path) -> dict[str, list[str]]:
    """Release binaries installed by `install_shader_toolchain.sh`.

    A tree can honestly span more than one compiler release (the manifest records
    the release that reproduces each artifact), so the check must offer every
    installed release as a candidate. Scanning `.tools/` keeps the documented
    command a bare `python validation/check_shader_provenance.py` even when PATH
    holds only one of them.
    """
    tools = root.parent / ".tools"
    found: dict[str, list[str]] = {}
    glslang: list[str] = []
    for pattern in ("glslang-*/bin/glslangValidator*", "glslang-*/bin/glslang.exe"):
        for entry in sorted(tools.glob(pattern)):
            if entry.is_file():
                glslang.append(str(entry))
    if glslang:
        found["glslangValidator"] = glslang
    glslc = tools / "env" / "Library" / "bin" / "glslc.exe"
    if glslc.is_file():
        found["glslc"] = [str(glslc)]
    return found


def gather(names, overrides: dict[str, list[str]], lookup=shutil.which) -> dict[str, list[Toolchain]]:
    """Resolve every manifest compiler name to the candidate binaries on offer.

    An explicit `--compiler-path` for a name replaces the PATH lookup for that name,
    so a caller can offer releases PATH cannot express.
    """
    toolchains: dict[str, list[Toolchain]] = {}
    for name in names:
        paths = overrides.get(name)
        if paths is None:
            found = lookup(name)
            paths = [found] if found else []
        toolchains[name] = [probe(path) for path in paths]
    return toolchains


def choose(
    entry: dict,
    candidates: list[Toolchain],
    required_versions: dict[str, str] | None = None,
    *,
    allow_missing: frozenset[str] = frozenset(),
) -> tuple[Toolchain | None, dict]:
    """Pick the candidate that reproduces one artifact.

    Returns `(toolchain, row_update)`: a toolchain to compile with, or None plus the
    status explaining why none fits. The release an entry records beats a
    `--require-version` pin, and the two disagreeing is reported rather than
    resolved silently, because a stale command line should not override the file.
    """
    name = entry["compiler"]
    recorded = entry.get("compiler_version")
    flagged = (required_versions or {}).get(name)
    update: dict = {}
    if recorded and flagged is not None and flagged != recorded:
        update.update(
            status="pin_conflict",
            expected_version=recorded,
            conflicting_version=flagged,
            version_source="manifest",
        )
        return None, update
    pin = recorded or flagged
    update["version_source"] = "manifest" if recorded else ("flag" if flagged else None)
    if pin is not None:
        update["expected_version"] = pin
    if not candidates:
        update["status"] = "skipped" if name in allow_missing else "missing_compiler"
        return None, update
    if pin is None:
        chosen = candidates[0]
        if len(candidates) > 1:
            update["toolchain_ambiguous"] = True
    else:
        chosen = next((candidate for candidate in candidates if candidate.version == pin), None)
        if chosen is None:
            update.update(
                status="wrong_version",
                observed_versions=[candidate.version for candidate in candidates],
            )
            return None, update
    update.update(version=chosen.version, compiler_path=chosen.path)
    return chosen, update


def evaluate(
    entries: list[dict],
    rebuild,
    *,
    toolchains: dict[str, list[Toolchain]],
    allow_missing: frozenset[str],
    required_versions: dict[str, str],
    root: Path,
) -> tuple[list[dict], list[dict]]:
    """Rebuild every manifest entry with the toolchain it records and classify it.

    `rebuild(entry, compiler, out)` compiles one entry and returns
    `(exit_code, message)`. Returning `(rows, failures)` keeps the comparison logic
    testable without a compiler on the machine.
    """
    rows: list[dict] = []
    for entry in entries:
        artifact = entry["artifact"]
        source = entry["source"]
        name = entry["compiler"]
        row = {
            "artifact": artifact,
            "source": source,
            "compiler": name,
            "flags": list(entry.get("flags", [])),
        }
        toolchain, update = choose(
            entry,
            toolchains.get(name, []),
            required_versions,
            allow_missing=allow_missing,
        )
        row.update(update)
        if toolchain is None:
            rows.append(row)
            continue
        source_path = root / "shaders" / source
        embedded = root / "shaders" / artifact
        if not source_path.is_file():
            row["status"] = "missing_source"
            rows.append(row)
            continue
        if not embedded.is_file():
            row["status"] = "missing_artifact"
            rows.append(row)
            continue
        with tempfile.TemporaryDirectory(prefix="shader-provenance-") as scratch:
            rebuilt = Path(scratch) / artifact
            # A nested artifact rebuilds into the same relative path, so its
            # directory has to exist in the scratch tree first.
            rebuilt.parent.mkdir(parents=True, exist_ok=True)
            code, message = rebuild(entry, toolchain.path, rebuilt)
            row["compile_exit"] = code
            if code != 0:
                row.update(status="compile_failed", error=message[-2000:])
            else:
                row["embedded_sha256"] = sha256(embedded)
                row["rebuilt_sha256"] = sha256(rebuilt)
                row["status"] = (
                    "identical"
                    if row["embedded_sha256"] == row["rebuilt_sha256"]
                    else "drifted"
                )
        rows.append(row)
    failures = [row for row in rows if row["status"] not in OK_STATUSES]
    return rows, failures


def build_rebuild(root: Path):
    def rebuild(entry: dict, compiler: str, out: Path) -> tuple[int, str]:
        command = [
            compiler,
            *entry.get("flags", []),
            f"shaders/{entry['source']}",
            "-o",
            str(out),
        ]
        result = subprocess.run(
            command, cwd=root, text=True, capture_output=True, check=False
        )
        return result.returncode, result.stdout + result.stderr

    return rebuild


def read_manifest(path: Path, root: Path) -> tuple[list[dict], list[str]]:
    """Return the manifest entries and a list of structural problems."""
    if not path.is_file():
        return [], [f"manifest not found: {path}"]
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != 1:
        return [], [f"unsupported manifest schema: {document.get('schema')!r}"]
    entries = document.get("shaders")
    if not isinstance(entries, list) or not entries:
        return [], ["manifest records no shaders"]
    problems = []
    # Artifacts are addressed by their path under `shaders/`, so a module frozen in
    # a nested directory (the pre-tuning scalar fallbacks) is pinned the same way a
    # top-level one is; a bare name is its own path because top-level artifacts live
    # directly under `shaders/`.
    committed = {
        item.relative_to(root / "shaders").as_posix()
        for item in (root / "shaders").rglob("*")
        if item.is_file()
    }
    for entry in entries:
        missing = {"artifact", "source", "compiler", "flags"} - set(entry)
        if missing:
            problems.append(f"{entry.get('artifact', '?')}: missing keys {sorted(missing)}")
        elif entry["artifact"] not in committed:
            problems.append(f"{entry['artifact']}: not committed under shaders/")
        recorded = entry.get("compiler_version")
        if "compiler_version" in entry and not (isinstance(recorded, str) and recorded.strip()):
            problems.append(f"{entry['artifact']}: compiler_version must be a non-empty string")
    return entries, problems


def ordered(entry: dict) -> dict:
    """Re-emit one entry so recording a release does not reshuffle the manifest."""
    return {key: entry[key] for key in KEY_ORDER if key in entry} | {
        key: value for key, value in entry.items() if key not in KEY_ORDER
    }


def record_versions(path: Path, rows: list[dict]) -> tuple[int, int, int]:
    """Write the release that reproduced each artifact back into the manifest.

    Only artifacts that rebuilt byte-for-byte are recorded; anything that did not
    reproduce keeps whatever it recorded before, so a partial toolchain cannot
    overwrite a verified pin with a guess.
    """
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    observed = {row["artifact"]: row for row in rows}
    written = unchanged = unrecorded = 0
    entries = []
    for entry in document["shaders"]:
        row = observed.get(entry["artifact"], {})
        version = row.get("version")
        if row.get("status") == "identical" and version:
            if entry.get("compiler_version") == version:
                unchanged += 1
            else:
                entry["compiler_version"] = version
                written += 1
        else:
            unrecorded += 1
        entries.append(ordered(entry))
    document["shaders"] = entries
    text = json.dumps(document, indent=2) + "\n"
    if b"\r\n" in raw:
        text = text.replace("\n", "\r\n")
    path.write_bytes(text.encode("utf-8"))
    return written, unchanged, unrecorded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--compiler-path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="a candidate binary for a manifest compiler name (repeatable; replaces PATH for that name)",
    )
    parser.add_argument(
        "--require-version",
        action="append",
        default=[],
        metavar="NAME=VERSION",
        help="pin entries that record no release of their own (repeatable)",
    )
    parser.add_argument(
        "--allow-missing",
        action="append",
        default=[],
        help="report NAME as skipped instead of failing when it is not installed",
    )
    parser.add_argument(
        "--record-versions",
        action="store_true",
        help="write the release that reproduced each artifact back into the manifest",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    entries, problems = read_manifest(args.manifest, ROOT)
    if not entries:
        for problem in problems:
            print(f"error: {problem}")
        return 1

    overrides = parse_compiler_paths(args.compiler_path, parser.error)
    # Installed releases are candidates too, so an artifact pinned to a second
    # release reproduces without spelling every path out on the command line.
    for name, paths in discover_local(ROOT).items():
        known = overrides.setdefault(name, [])
        known.extend(path for path in paths if path not in known)
    required_versions = {}
    for pair in args.require_version:
        name, _, version = pair.partition("=")
        if not version:
            parser.error(f"expected NAME=VERSION, got {pair!r}")
        required_versions[name] = version

    names = sorted({entry["compiler"] for entry in entries})
    toolchains = gather(names, overrides)
    rows, failures = evaluate(
        entries,
        build_rebuild(ROOT),
        toolchains=toolchains,
        allow_missing=frozenset(args.allow_missing),
        required_versions=required_versions,
        root=ROOT,
    )

    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    notes = [
        f"{name}: more than one candidate and no recorded release; the first was used"
        for name in sorted({row["compiler"] for row in rows if row.get("toolchain_ambiguous")})
    ]
    report = {
        "manifest": str(args.manifest),
        "compilers": {
            name: [
                {"path": candidate.path, "version": candidate.version, "reported": candidate.reported}
                for candidate in toolchains[name]
            ]
            for name in names
        },
        "status_counts": dict(sorted(by_status.items())),
        "manifest_problems": problems,
        "notes": notes,
        "shaders": rows,
        "pass": not failures and not problems,
    }
    content = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")

    for row in rows:
        detail = ""
        if row["status"] == "drifted":
            detail = f" embedded={row['embedded_sha256'][:12]} rebuilt={row['rebuilt_sha256'][:12]}"
        elif row["status"] == "wrong_version":
            observed = ", ".join(str(item) for item in row["observed_versions"]) or "nothing"
            detail = f" expected {row['expected_version']}, candidates report {observed}"
        elif row["status"] == "pin_conflict":
            detail = (
                f" records {row['expected_version']} but --require-version says "
                f"{row['conflicting_version']}; drop the flag, the manifest wins"
            )
        print(f"{row['status']:16} {row['artifact']}{detail}")
    for problem in problems:
        print(f"manifest        {problem}")
    for note in notes:
        print(f"note            {note}")
    if args.record_versions:
        written, unchanged, unrecorded = record_versions(args.manifest, rows)
        print(
            f"recorded        {written} written, {unchanged} already recorded, "
            f"{unrecorded} left alone"
        )
    print(json.dumps(report["status_counts"]))
    if not report["pass"]:
        print(
            f"FAIL: {len(failures)} artifact(s) do not reproduce under the manifest toolchain"
        )
        return 1
    print("PASS: every checked artifact reproduces byte-for-byte")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
