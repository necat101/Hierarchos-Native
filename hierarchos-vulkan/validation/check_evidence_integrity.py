#!/usr/bin/env python3
"""Offline digest guard for qualification evidence roots.

Every qualification receipt in this repository declares digests for the files it
rests on: an ``evidence`` map, ``retained`` family hashes, ``components``, or a
``source_files`` inventory. A digest only means something if the bytes it names are
present and unchanged. This guard resolves every declared binding and reports what it
actually finds, separating three questions that receipts usually blur:

* **Integrity** - do the bytes on disk hash to the declared digest?
* **Availability** - is the referenced artifact present at all (working tree and, for
  historical roots, the Git object database)?
* **Provenance** - is there an execution record that establishes a producer ran?

A matching SHA-256 proves the first. It says nothing about the third, so provenance is
reported separately and never folded into a pass/fail digest result.

Modes
-----
``--root DIR``        audit one evidence root
``--release FILE``    audit a unified release and every component it binds (strict scope)
``--audit-all``       discover every digest-declaring root under ``release_parity``

``--strict`` fails closed on any in-scope mismatch, missing artifact, unsafe path,
malformed digest or unsupported schema. Provenance findings stay advisory: a receipt
without a preserved log is reported, not failed, because no repository convention
guarantees one. Historical roots are audited in report mode; they are not silently
promoted to passing.

Usage
-----
    python hierarchos-vulkan/validation/check_evidence_integrity.py --release \
      hierarchos-vulkan/validation/release_parity/multimodal_training/\
20261115-phase25-current-source/release-qualification.json --strict
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_PARITY = REPO_ROOT / "hierarchos-vulkan" / "validation" / "release_parity"

# Binding statuses. Integrity and provenance are deliberately separate spaces.
VALID = "valid"
MISMATCH = "mismatch"
MISSING = "missing"
UNSAFE = "unsafe-path"
INVALID = "invalid-digest"
UNSUPPORTED = "unsupported-schema"
DUPLICATE = "ambiguous-reference"
SOURCE_DRIFT = "source-drift"
PROVENANCE = "provenance-unestablished"

FAILURES = {MISMATCH, MISSING, UNSAFE, INVALID, UNSUPPORTED, DUPLICATE, SOURCE_DRIFT}

# Candidate prefixes for a receipt's relative evidence keys. Phase XV keys such as
# ``gemma3/qualification.json`` are relative to ``retained/`` while ``isolation.json``
# is relative to ``gates/``; both conventions are real and in use.
PREFIXES = ("", "evidence", "retained", "gates", "multimodal", "multimodal/stages")

HEX = set("0123456789abcdef")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value.lower()) <= HEX


def safe_join(base: Path, relative: str) -> Path:
    """Join ``relative`` under ``base``, rejecting absolute and escaping paths.

    Both the lexical form and the resolved form are checked so a symlink cannot move
    the target outside the root after the traversal check passes.
    """
    pure = PurePosixPath(relative.replace("\\", "/"))
    if pure.is_absolute() or any(part == ".." for part in pure.parts):
        raise ValueError(f"path escapes the evidence root: {relative}")
    candidate = (base / pure).resolve()
    root = base.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path resolves outside the evidence root: {relative}")
    return candidate


def candidates(root: Path, relative: str) -> list[Path]:
    """Real on-disk candidates for a receipt-relative key, in convention order."""
    out: list[Path] = []
    pure = PurePosixPath(relative.replace("\\", "/"))
    for prefix in PREFIXES:
        parts = [p for p in pure.parts]
        if prefix:
            parts = prefix.split("/") + parts
        try:
            out.append(safe_join(root, "/".join(parts)))
        except ValueError:
            continue
        # Evidence logs are sometimes filed one level down under ``logs/``.
        if len(parts) > 1:
            parent, name = parts[:-1], parts[-1]
            try:
                out.append(safe_join(root, "/".join(parent + ["logs", name])))
            except ValueError:
                pass
    seen, unique = set(), []
    for path in out:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


@dataclass
class Finding:
    kind: str
    label: str
    status: str
    detail: str = ""
    path: str = ""


@dataclass
class Report:
    root: Path
    receipt: Path
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, kind, label, status, detail="", path=""):
        self.findings.append(Finding(kind, label, status, detail, path))

    def count(self, status: str) -> int:
        return sum(1 for f in self.findings if f.status == status)

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.status in FAILURES]

    @property
    def integrity_failures(self) -> list[Finding]:
        """Failures that are not explained by the tree having moved on."""
        return [f for f in self.failures if f.status != SOURCE_DRIFT]

    @property
    def flags(self) -> list[str]:
        flags = []
        if self.integrity_failures:
            flags.append("integrity_mismatch")
        if self.count(MISSING):
            flags.append("evidence_missing")
        if self.count(SOURCE_DRIFT):
            flags.append("source_drift")
        if self.count(PROVENANCE):
            flags.append("provenance_unsupported")
        if not self.findings:
            flags.append("no-digest-bindings")
        return flags


def inside(root: Path, path: Path) -> bool:
    """True when ``path`` resolves to something within ``root`` (symlinks included)."""
    try:
        resolved, base = path.resolve(), root.resolve()
    except OSError:
        return False
    return resolved == base or base in resolved.parents


def index_by_name(root: Path) -> dict[str, list[Path]]:
    """Map basename -> files under the root, for evidence filed in a stage subtree."""
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and inside(root, path):
            index.setdefault(path.name, []).append(path)
    return index


class LazyIndex:
    """Build the basename index only when a binding actually misses its prefixes."""

    def __init__(self, root: Path):
        self.root = root
        self._table: dict[str, list[Path]] | None = None

    def get(self) -> dict[str, list[Path]]:
        if self._table is None:
            self._table = index_by_name(self.root)
        return self._table


def resolve_binding(root: Path, relative: str, digest: str,
                    by_name: "dict[str, list[Path]] | LazyIndex | None" = None) -> tuple[str, str, str]:
    """Classify one relative-path binding. Returns (status, detail, resolved path)."""
    if not is_digest(digest):
        return INVALID, f"not a sha256 digest: {digest!r}", ""
    try:
        options = candidates(root, relative)
    except ValueError as exc:  # defensive: candidates() already filters
        return UNSAFE, str(exc), ""
    if not options:
        return UNSAFE, f"unsafe relative path: {relative}", ""
    present = [p for p in options if p.is_file()]
    if present:
        for path in present:
            if sha256_file(path) == digest:
                return VALID, "", str(path)
        actual = sha256_file(present[0])
        mismatched = present[0]
    else:
        actual, mismatched = "", options[0]

    # Evidence is often filed under a dated stage subtree that the flat prefix list
    # cannot name. Match by basename and declared digest before calling it a miss.
    table = by_name.get() if isinstance(by_name, LazyIndex) else by_name
    if table is not None:
        for path in table.get(PurePosixPath(relative.replace("\\", "/")).name, []):
            if sha256_file(path) == digest:
                return VALID, "", str(path)
        present = [p for p in table.get(
            PurePosixPath(relative.replace("\\", "/")).name, []) if p.is_file()]
        if not present:
            return MISSING, "no file with this name exists under the root", str(mismatched)
        if actual and actual != digest:
            return MISMATCH, f"declared {digest[:12]} actual {actual[:12]}", str(mismatched)
        return MISMATCH, (f"declared {digest[:12]}; {len(present)} same-named file(s) "
                          "under the root, none matching"), str(mismatched)
    if actual:
        return MISMATCH, f"declared {digest[:12]} actual {actual[:12]}", str(mismatched)
    return MISSING, "no candidate path exists", str(mismatched)


def check_map(report: Report, root: Path, kind: str, mapping: dict,
              by_name: "dict[str, list[Path]] | LazyIndex | None" = None):
    for label, digest in mapping.items():
        if not isinstance(label, str):
            report.add(kind, str(label), INVALID, "non-string evidence key")
            continue
        status, detail, path = resolve_binding(root, label, digest, by_name)
        report.add(kind, label, status, detail, path)


def as_digest_map(value):
    """A digest map has one string per entry; a metadata block has counts/flags.

    Older receipts use ``evidence`` for both: a path->sha256 map, or a summary such
    as ``{"file": "evidence.zip", "bytes": 21372426, "files": 3241}``. Treating the
    latter as digests reports four failures that say nothing about integrity.
    """
    if not isinstance(value, dict) or not value:
        return None
    if not all(isinstance(v, str) for v in value.values()):
        return None
    return value


def extract(doc: dict) -> dict:
    out = {"evidence": {}, "components": {}, "retained": {}, "sources": {},
           "binaries": {}, "producer": None, "evidence_files": {}}
    if as_digest_map(doc.get("evidence")):
        out["evidence"].update(doc["evidence"])
    if as_digest_map(doc.get("evidence_files")):
        out["evidence_files"].update(doc["evidence_files"])
    elif isinstance(doc.get("evidence_files"), list):
        for row in doc["evidence_files"]:
            if isinstance(row, dict) and "path" in row and "sha256" in row:
                out["evidence_files"][row["path"]] = row["sha256"]
    if isinstance(doc.get("components"), dict):
        for name, row in doc["components"].items():
            if isinstance(row, dict) and row.get("path") and row.get("sha256"):
                out["components"][name] = (row["path"], row["sha256"])
    if isinstance(doc.get("retained"), dict):
        for name, row in doc["retained"].items():
            digest = row.get("qualification_sha256") if isinstance(row, dict) else row
            if digest:
                out["retained"][name] = digest
    out["duplicates"] = []
    for key in ("sources", "source_files"):
        if not isinstance(doc.get(key), dict):
            continue
        for path, digest in doc[key].items():
            if path in out["sources"] and out["sources"][path] != digest:
                out["duplicates"].append((path, out["sources"][path], digest))
            out["sources"].setdefault(path, digest)
    if isinstance(doc.get("binaries"), dict):
        out["binaries"] = doc["binaries"]
    if isinstance(doc.get("producer_sha256"), str):
        out["producer"] = doc["producer_sha256"]
    return out


def history_digests(relative: str, cwd: Path, limit: int = 200) -> set[str]:
    """sha256 of every committed version of ``relative``, newest first, capped.

    A declared evidence digest is a hash of file *content*; a Git object id is a hash
    of the object header plus content, so ``git cat-file`` cannot answer this. The
    question that matters is whether the declared content ever existed at the path the
    receipt names, which means hashing each revision's blob.
    """
    out: set[str] = set()
    try:
        revs = subprocess.run(["git", "log", "--all", "--format=%H", "--", relative],
                              capture_output=True, text=True, cwd=str(cwd)).stdout.split()
    except OSError:
        return out
    for rev in revs[:limit]:
        blob = subprocess.run(["git", "show", f"{rev}:{relative}"],
                              capture_output=True, cwd=str(cwd)).stdout
        if blob:
            out.add(hashlib.sha256(blob).hexdigest())
    return out


_history_cache: dict[tuple[str, str], set[str]] = {}


def history_has(relative: str, digest: str, cwd: Path) -> bool:
    key = (str(cwd), relative)
    if key not in _history_cache:
        _history_cache[key] = history_digests(relative, cwd)
    return digest in _history_cache[key]


def audit_receipt(receipt: Path, root: Path, repo: Path, depth: int = 0,
                  nested: bool = False, git_objects: bool = True,
                  verify_sources: bool = True) -> Report:
    report = Report(root=root, receipt=receipt)
    try:
        doc = json.loads(receipt.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - corrupt receipt is itself a finding
        report.add("receipt", receipt.name, UNSUPPORTED, f"unreadable JSON: {exc}")
        return report
    if not isinstance(doc, dict):
        report.add("receipt", receipt.name, UNSUPPORTED, "receipt is not an object")
        return report

    parts = extract(doc)
    declared = sum(len(parts[k]) for k in ("evidence", "components", "retained", "sources"))
    if parts["sources"] and not verify_sources:
        report.notes.append(f"{len(parts['sources'])} source binding(s) not re-hashed "
                            "in audit mode; run --verify-sources to include them")
    if not declared and not parts["binaries"] and not parts["producer"]:
        report.add("schema", receipt.name, UNSUPPORTED,
                   "receipt declares no digest bindings this guard understands")
        return report

    for path, first, second in parts["duplicates"]:
        report.add("duplicate", path, DUPLICATE,
                   f"declared twice with different digests: {first[:12]} vs {second[:12]}")
    by_name = LazyIndex(root) if (parts["evidence"] or parts["evidence_files"]
                               or parts["retained"]) else None
    if parts["evidence"]:
        check_map(report, root, "evidence", parts["evidence"], by_name)
    if parts["evidence_files"]:
        check_map(report, root, "evidence_files", parts["evidence_files"], by_name)
    for name, (relative, digest) in sorted(parts["components"].items()):
        status, detail, path = resolve_binding(root, relative, digest)
        report.add("component", name, status, detail, path)
        # Only aggregate receipts carry their own digest maps; a component's raw
        # evidence file is verified by its digest and is not re-parsed as a root.
        if status == VALID and depth < 1 and Path(relative).name == "current-qualification.json":
            child = Path(path)
            report.notes.append(f"nested: {child.relative_to(root)}")
            audit_nested(report, child, child.parent, repo, depth + 1)
    for name, digest in sorted(parts["retained"].items()):
        relative = f"retained/{name}/qualification.json"
        status, detail, path = resolve_binding(root, relative, digest, by_name)
        if status == MISSING:
            # A family may file its receipt directly under the root.
            alt = f"{name}/qualification.json"
            status2, detail2, path2 = resolve_binding(root, alt, digest, by_name)
            if status2 != MISSING:
                status, detail, path = status2, detail2, path2
        report.add("retained", name, status, detail, path)
        if status in (MISMATCH, VALID) and path:
            # Cross-check the aggregate's copy against the family receipt's own view.
            try:
                family = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                family = None
            if isinstance(family, dict) and family.get("pass") is False:
                report.add("retained", name, MISMATCH,
                           "family receipt reports pass=false", path)
    for relative, digest in (sorted(parts["sources"].items()) if verify_sources else []):
        status, detail, path = resolve_binding(repo, relative, digest)
        if status == MISMATCH:
            status = SOURCE_DRIFT
            detail = "bound source differs from the receipt's revision: " + detail
        report.add("source", relative, status, detail, path)
    if parts["binaries"]:
        for name in sorted(parts["binaries"]):
            report.add("binary", name, PROVENANCE,
                       "digest declared; no executable artifact is tracked, so the "
                       "digest cannot be re-derived offline")
    if parts["producer"]:
        report.add("producer", "producer_sha256", PROVENANCE,
                   f"{parts['producer'][:12]}: no producer artifact is tracked under "
                   "this digest, so the producing run cannot be identified")

    if git_objects:
        sources_of_truth = {"evidence": parts["evidence"],
                            "evidence_files": parts["evidence_files"],
                            "retained": parts["retained"]}
        checked = never = 0
        for finding in report.findings:
            if finding.status != MISMATCH or finding.kind not in sources_of_truth:
                continue
            digest = sources_of_truth[finding.kind].get(finding.label)
            if not is_digest(digest) or not finding.path:
                continue
            try:
                relative = str(Path(finding.path).resolve().relative_to(repo.resolve()))
            except ValueError:
                continue
            checked += 1
            if not history_has(relative, digest, repo):
                never += 1
        if checked:
            detail = (f"{never} of {checked} mismatched binding(s) name content that no "
                      "committed revision of that path ever held")
            if checked > never:
                detail += (f"; {checked - never} hold it at an earlier revision, a stale "
                           "path binding rather than reconstructed evidence")
            report.add("history", "declared-evidence-content", PROVENANCE, detail)
    return report


def audit_nested(parent: Report, receipt: Path, root: Path, repo: Path, depth: int):
    """Fold a component receipt's own bindings into the parent report."""
    child = audit_receipt(receipt, root, repo, depth=depth, nested=True)
    for finding in child.findings:
        if finding.status in (VALID,):
            continue
        finding.label = f"{receipt.name}:{finding.label}"
        parent.findings.append(finding)


def discover_roots(base: Path) -> list[tuple[Path, Path]]:
    """Every (root directory, receipt) pair that declares digest bindings."""
    pairs: dict[Path, Path] = {}
    for path in sorted(base.rglob("*.json")):
        if ".git" in path.parts or path.stat().st_size > 8 << 20:
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        parts = extract(doc)
        if sum(len(parts[k]) for k in ("evidence", "components", "retained", "sources")):
            pairs.setdefault(path.parent, path)
    return sorted(pairs.items())


def pick_receipt(root: Path) -> Path | None:
    """The receipt in ``root`` that declares the most digest bindings."""
    best, score = None, -1
    for path in sorted(root.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        parts = extract(doc)
        value = sum(len(parts[k]) for k in ("evidence", "components", "retained", "sources"))
        if value > score or (value == score and best is not None
                             and path.name == "current-qualification.json"):
            best, score = path, value
    return best if score > 0 else None


def release_roots(release_file: Path) -> list[Path]:
    """Roots inside the unified release's dependency closure."""
    release_root = release_file.parent
    doc = json.loads(release_file.read_text(encoding="utf-8"))
    roots = {release_root}
    for row in (doc.get("components") or {}).values():
        if not isinstance(row, dict) or not row.get("path"):
            continue
        try:
            target = safe_join(release_root, row["path"])
        except ValueError:
            continue
        if target.name in ("current-qualification.json", "qualification.json"):
            roots.add(target.parent)
    return sorted(roots)


def summarise(reports: list[Report], release_root: Path | None) -> dict:
    keys = ("root", "receipt", "flags", "failures", "counts")
    out = {"roots": [], "totals": {}}
    counters = {"roots": 0, "verified": 0, "mismatch": 0, "missing": 0,
                "drift": 0, "provenance": 0, "release_referenced": 0}
    for report in reports:
        counters["roots"] += 1
        release_ref = bool(release_root) and (
            release_root == report.root or release_root in report.root.parents)
        if release_ref:
            counters["release_referenced"] += 1
        if report.integrity_failures:
            counters["mismatch"] += 1
        if report.count(MISSING):
            counters["missing"] += 1
        if report.count(SOURCE_DRIFT):
            counters["drift"] += 1
        if report.count(PROVENANCE):
            counters["provenance"] += 1
        if not report.integrity_failures:
            counters["verified"] += 1
        out["roots"].append({
            "root": str(report.root.relative_to(REPO_ROOT)) if REPO_ROOT in report.root.parents
            else str(report.root),
            "receipt": report.receipt.name,
            "flags": report.flags,
            "integrity_failures": len(report.integrity_failures),
            "release_referenced": release_ref,
            "failures": len(report.failures),
            "counts": {s: report.count(s) for s in
                       (VALID, MISMATCH, MISSING, SOURCE_DRIFT, PROVENANCE, INVALID,
                        UNSAFE, UNSUPPORTED, DUPLICATE)},
        })
    out["totals"] = counters
    return out


def print_report(report: Report, verbose: bool):
    rel = report.root.relative_to(REPO_ROOT) if REPO_ROOT in report.root.parents else report.root
    bad = report.failures
    print(f"\n{rel}")
    print(f"  receipt: {report.receipt.name}")
    print(f"  findings: {len(report.findings)}  failures: {len(bad)}  "
          f"valid: {report.count(VALID)}  missing: {report.count(MISSING)}  "
          f"provenance: {report.count(PROVENANCE)}")
    for finding in (report.findings if verbose else bad)[:40]:
        print(f"    [{finding.status:22s}] {finding.kind}/{finding.label} {finding.detail}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", action="append", default=[], help="evidence root to audit")
    parser.add_argument("--release", help="unified release-qualification.json to audit")
    parser.add_argument("--audit-all", action="store_true",
                        help="discover and report every digest-declaring root")
    parser.add_argument("--strict", action="store_true",
                        help="fail closed on any integrity finding in scope")
    parser.add_argument("--json", action="store_true", help="machine-readable summary")
    parser.add_argument("--verbose", action="store_true", help="print valid bindings too")
    parser.add_argument("--latest-release", action="store_true",
                        help="audit the newest sealed release root")
    parser.add_argument("--verify-sources", action="store_true",
                        help="re-hash bound source files too (slow; audit mode skips them)")
    parser.add_argument("--no-git", action="store_true",
                        help="skip the Git object-database provenance cross-check")
    parser.add_argument("--documented-exception", action="append", default=[],
                        metavar="PATH",
                        help="report this root's failures as documented, not silent")
    args = parser.parse_args(argv)

    if not (args.root or args.release or args.audit_all or args.latest_release):
        parser.error("choose --root, --release, --latest-release or --audit-all")

    if args.latest_release:
        sealed = sorted(RELEASE_PARITY.rglob("release-qualification.json"))
        if not sealed:
            print("no sealed release found", file=sys.stderr)
            return 2
        args.release = str(sealed[-1])

    reports: list[Report] = []
    scope_failures = 0
    release_root = None

    for root in args.root:
        root_path = Path(root).resolve()
        receipt = pick_receipt(root_path)
        if receipt is None:
            print(f"no digest-declaring receipt in {root}", file=sys.stderr)
            return 2
        report = audit_receipt(receipt, root_path, REPO_ROOT, git_objects=not args.no_git,
                               verify_sources=args.verify_sources)
        reports.append(report)
        scope_failures += len(report.failures)

    if args.release:
        release_file = Path(args.release).resolve()
        release_root = release_file.parent
        report = audit_receipt(release_file, release_root, REPO_ROOT,
                               git_objects=not args.no_git, verify_sources=True)
        reports.append(report)
        scope_failures += len(report.failures)
        for component_root in release_roots(release_file):
            if component_root == release_root:
                continue
            receipt = pick_receipt(component_root)
            if receipt is None:
                continue
            nested_report = audit_receipt(receipt, component_root, REPO_ROOT,
                                          git_objects=not args.no_git)
            reports.append(nested_report)
            scope_failures += len(nested_report.failures)

    if args.audit_all:
        for root_path, receipt in discover_roots(RELEASE_PARITY):
            reports.append(audit_receipt(receipt, root_path, REPO_ROOT,
                                         git_objects=not args.no_git,
                                         verify_sources=args.verify_sources))

    exceptions = [e.rstrip("/\\").replace("\\", "/") for e in args.documented_exception]
    documented = 0
    for report in reports:
        key = str(report.root.relative_to(REPO_ROOT)).replace("\\", "/") \
            if REPO_ROOT in report.root.parents else str(report.root)
        if report.failures and any(key.endswith(e) or e in key for e in exceptions):
            documented += 1
            report.notes.append("documented exception: reported, not suppressed")
            if args.strict and report in reports[:len(args.root) + (1 if args.release else 0)]:
                scope_failures -= len(report.failures)

    if args.json:
        print(json.dumps(summarise(reports, release_root), indent=2))
    else:
        for report in reports:
            if report.failures or args.verbose or report.count(PROVENANCE):
                print_report(report, args.verbose)
        totals = summarise(reports, release_root)["totals"]
        print(f"\nroots: {totals['roots']}  release-referenced: {totals['release_referenced']}  "
              f"mismatch: {totals['mismatch']}  missing: {totals['missing']}  "
              f"provenance: {totals['provenance']}  documented exceptions: {documented}")

    if args.strict and args.audit_all:
        # Audit mode never fails the build; only strict scope does.
        return 0
    if args.strict:
        return 1 if scope_failures else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
