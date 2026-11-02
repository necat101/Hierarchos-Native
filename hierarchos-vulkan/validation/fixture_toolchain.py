"""Record fixture provenance only after a generator reproduces its frozen values.

Pins live in a sidecar bound to each file's SHA-256 so fixtures whose literal
bytes are hashed by Rust remain unchanged. Legacy entries are hash-bound too:
neither a new unpinned fixture nor an edited legacy fixture can enter silently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SIDECAR = "deepseek_v41_toolchain.json"
LEGACY = "deepseek_v41_toolchain_legacy.json"


def current():
    import numpy
    import torch
    return {"torch": torch.__version__, "numpy": numpy.__version__,
            "cpu": torch.backends.cpu.get_cpu_capability()}


def payload(value):
    return {k: v for k, v in value.items() if k != "toolchain"} if isinstance(value, dict) else value


def variant_path(path):
    path = Path(path)
    variant = os.environ.get("HIERARCHOS_FIXTURE_VARIANT", "")
    if not variant:
        return path
    if not variant.replace("_", "").replace("-", "").isalnum():
        raise ValueError("fixture variant must contain only letters, digits, '-' or '_'")
    return path.with_name(f"{path.stem}.{variant}{path.suffix}")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path):
    path = Path(path)
    value = _read(path) if path.exists() else {}
    inline = value.get("toolchain") if isinstance(value, dict) else None
    entry = _read(path.parent / SIDECAR).get(path.name)
    if entry is not None:
        if entry.get("sha256") != _sha(path):
            raise ValueError(f"{path.name}: stale toolchain pin (fixture SHA-256 differs)")
        if inline is not None and inline != entry["toolchain"]:
            raise ValueError(f"{path.name}: inline and sidecar pins disagree")
        return entry["toolchain"]
    return inline


def _allow():
    return ("--allow-toolchain-mismatch" in sys.argv
            or os.environ.get("HIERARCHOS_ALLOW_TOOLCHAIN_MISMATCH") == "1")


def check_file(path, *, actual=None, allow=None):
    path = Path(path)
    if not path.exists():
        return
    expected = pin(path)
    if expected is None:
        warnings.warn(f"{path.name}: predates toolchain pinning; reproduction is not certified")
        return
    actual = current() if actual is None else actual
    mismatch = any(expected.get(k) != actual.get(k) for k in ("torch", "numpy", "cpu") if k in expected)
    if mismatch:
        message = (f"{path.name}: fixture toolchain {expected} differs from interpreter {actual}; "
                   "use the pinned interpreter/CPU, a host variant, or --allow-toolchain-mismatch")
        if _allow() if allow is None else allow:
            warnings.warn(message)
        else:
            raise SystemExit(message)


def write_fixture(path, text, *args, **kwargs):
    """A generator's write: enforce provenance and refuse to replace old values.

    Successful legacy reproduction gets a sidecar pin; the original JSON bytes
    stay intact. A mismatched environment may compare under an explicit override
    but cannot silently replace an existing pin.
    """
    path = Path(path)
    generated = json.loads(text)
    if not isinstance(generated, (dict, list)):
        raise ValueError(f"{path}: expected a fixture object or array")
    check_file(path)
    if path.exists():
        if payload(_read(path)) != payload(generated):
            raise ValueError(f"{path.name}: refusing to overwrite frozen CPU fixture values; use a host variant")
    else:
        path.write_text(text, *args, **kwargs)
    sidecar = path.parent / SIDECAR
    pins = _read(sidecar)
    existing = pin(path)
    pins[path.name] = {"toolchain": existing or current(), "sha256": _sha(path),
                       "evidence": "generator reproduced the complete fixture payload"}
    sidecar.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    baseline = path.parent / LEGACY
    if baseline.exists():
        legacy = _read(baseline)
        if legacy.pop(path.name, None) is not None:
            baseline.write_text(json.dumps(legacy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(text)


def audit(directory, expected, *, strict=False, historical=None):
    directory = Path(directory)
    legacy = _read(directory / LEGACY)
    files = sorted(p for p in directory.glob("deepseek_v41_*.json")
                   if p.name not in {SIDECAR, LEGACY, "deepseek_v41_flash_1017_config.json"})
    problems, pinned, old = [], 0, 0
    if not files:
        problems.append("empty fixture directory")
    names = {p.name for p in files}
    for name in legacy:
        if name not in names:
            problems.append(f"{name}: stale legacy entry (file missing)")
        if historical is not None:
            original = historical.get("hierarchos-vulkan/validation/fixtures/" + name)
            if original is None or original.get("sha256") != legacy[name].get("sha256"):
                problems.append(f"{name}: legacy baseline may only contain original handoff hashes")
    for path in files:
        try:
            toolchain = pin(path)
        except ValueError as error:
            problems.append(str(error))
            continue
        if toolchain is None:
            if legacy.get(path.name, {}).get("sha256") != _sha(path):
                problems.append(f"{path.name}: missing verified pin or legacy SHA-256")
            else:
                old += 1
                if strict:
                    problems.append(f"{path.name}: missing verified toolchain pin")
        else:
            pinned += 1
            if any(not toolchain.get(k) or toolchain[k] != expected.get(k) for k in ("torch", "numpy")):
                problems.append(f"{path.name}: library pin {toolchain} differs from {expected}")
            if not toolchain.get("cpu"):
                problems.append(f"{path.name}: CPU capability pin missing")
            if path.name in legacy:
                problems.append(f"{path.name}: verified fixture must leave the legacy baseline")
    return {"pinned": pinned, "legacy": old, "total": len(files), "problems": problems}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("fixture", type=Path)
    check.add_argument("--allow-toolchain-mismatch", action="store_true")
    all_check = sub.add_parser("check-all")
    all_check.add_argument("--fixtures", type=Path, default=FIXTURES)
    all_check.add_argument("--expect-from", type=Path, required=True)
    all_check.add_argument("--strict", action="store_true")
    all_check.add_argument("--historical-manifest", type=Path,
                           default=Path(__file__).resolve().parents[2] / "HANDOFF-MANIFEST.json")
    args = parser.parse_args()
    if args.command == "check":
        if not args.fixture.is_file() or pin(args.fixture) is None:
            parser.error("fixture has no verified toolchain pin")
        check_file(args.fixture, allow=args.allow_toolchain_mismatch)
        print(f"PASS {args.fixture.name}")
    else:
        expected = pin(args.expect_from)
        if expected is None:
            parser.error("--expect-from fixture has no verified toolchain pin")
        if not args.historical_manifest.is_file():
            parser.error("historical manifest is required to enforce the legacy ratchet")
        report = audit(args.fixtures, expected, strict=args.strict,
                       historical=_read(args.historical_manifest))
        print(json.dumps(report, indent=2))
        return int(bool(report["problems"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
