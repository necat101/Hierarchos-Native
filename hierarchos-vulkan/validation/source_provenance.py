#!/usr/bin/env python3
"""Verify a qualification receipt's source bindings at the revision it declares.

A sealed receipt describes the bytes of one git revision. Hashing the working
tree answers a different question: a later commit that touches a bound file
leaves the receipt's evidence intact while making the checkout disagree with it.
Reporting that as integrity drift hides the defect the guard exists to find, and
makes every historical receipt expire as soon as current source moves.

This module asks the receipt's own question - do the committed blobs at
``source_revision`` hash to the declared digests - and keeps the working-tree
comparison as a separate, advisory statement about the checkout.

Fail-closed cases: an unusable or unavailable revision, a path that escapes the
repository, a malformed digest, a file absent from the declared revision, and
content that differs at the declared revision. Only :func:`drift_paths` is
advisory, and callers must never fold it into the historical verdict.

This is deliberately not a waiver. A receipt that declares no revision cannot
reach the historical path at all: ``verify_bindings`` rejects anything that is
not a 40-character commit id, and then holds every binding to the exact bytes of
that commit. Changing ``source_revision`` to another real commit moves the
comparison to that commit and the declared digests stop matching, so a forged
revision fails rather than bypassing the check.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")

# Status names match check_evidence_integrity's vocabulary so a report can be
# folded into that audit without translation.
VALID = "valid"
MISMATCH = "mismatch"
MISSING = "missing"
UNSAFE = "unsafe-path"
INVALID = "invalid-digest"
UNAVAILABLE = "revision-unavailable"

FAILURES = frozenset({MISMATCH, MISSING, UNSAFE, INVALID, UNAVAILABLE})


@dataclass
class Finding:
    path: str
    status: str
    detail: str = ""


@dataclass
class Report:
    """Outcome of holding a receipt's source map to its declared revision."""

    revision: str
    available: bool = False
    findings: list[Finding] = field(default_factory=list)

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.status in FAILURES]

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def checked(self) -> int:
        return sum(1 for f in self.findings if f.status == VALID)

    def describe(self) -> str:
        """A short, caller-facing reason; empty when the report passes."""
        problems = self.failures
        if not problems:
            return ""
        text = "; ".join(message(f) for f in problems[:3])
        if len(problems) > 3:
            text += f"; and {len(problems) - 3} more binding(s)"
        return text


def message(finding: Finding) -> str:
    if finding.status == MISMATCH:
        return "bound source changed: " + finding.path
    if finding.status == MISSING:
        return "missing bound source: " + finding.path
    if finding.status == UNSAFE:
        return "unsafe source path: " + finding.path
    if finding.status == INVALID:
        return "invalid source digest for " + finding.path
    return finding.detail or ("source binding failed: " + finding.path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(repo: Path, args, **kwargs):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, **kwargs)


def commit_available(repo: Path, revision: str) -> bool:
    """True only for a 40-hex id that resolves to a commit present in ``repo``."""
    if not isinstance(revision, str) or HEX40.fullmatch(revision) is None:
        return False
    return _git(repo, ["cat-file", "-e", revision + "^{commit}"]).returncode == 0


def relative_path(raw) -> str | None:
    """Normalise a receipt-relative path, rejecting anything that escapes it."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    if "\n" in raw or "\0" in raw:
        return None
    pure = PurePosixPath(raw.replace("\\", "/"))
    if pure.is_absolute() or any(part == ".." for part in pure.parts):
        return None
    return pure.as_posix()


def _blob_digests(repo: Path, revision: str, relatives: list[str]) -> dict[str, str | None]:
    """sha256 of each path's committed content, from one ``git cat-file --batch``.

    A declared evidence digest is a hash of file *content*, while a git object id
    hashes an object header plus that content, so object ids cannot answer this
    question. Reading the blobs themselves can.

    A path that does not exist at ``revision``, or that names a tree or a
    submodule rather than a blob, maps to ``None`` and is reported as missing.
    """
    if not relatives:
        return {}
    query = "".join(f"{revision}:{path}\n" for path in relatives).encode("utf-8")
    result = _git(repo, ["cat-file", "--batch"], input=query)
    if result.returncode != 0:
        raise RuntimeError(
            "git cat-file --batch failed: "
            + result.stderr.decode("utf-8", "replace").strip()[:300])
    data, position, digests = result.stdout, 0, {}
    for path in relatives:
        newline = data.find(b"\n", position)
        if newline < 0:
            raise RuntimeError(f"truncated git cat-file output at {path}")
        header = data[position:newline].decode("utf-8", "replace")
        position = newline + 1
        fields = header.split(" ")
        if len(fields) == 2 and fields[1] == "missing":
            digests[path] = None
            continue
        if len(fields) != 3:
            digests[path] = None
            continue
        size = int(fields[2])
        if fields[1] == "blob":
            digests[path] = hashlib.sha256(data[position:position + size]).hexdigest()
        else:
            digests[path] = None
        position += size + 1
    return digests


def _join(prefix: str, path: str) -> str:
    """Repository-relative lookup path for a receipt key under ``prefix``."""
    base = relative_path(prefix) if prefix else None
    if prefix and base is None:
        return ""
    return f"{base}/{path}" if base else path


def verify_bindings(repo: Path, revision: str, sources: dict,
                    prefix: str = "") -> Report:
    """Hold every declared source digest to the bytes committed at ``revision``.

    ``sources`` maps a receipt-relative path to its declared sha256. A receipt
    whose keys are relative to a crate subdirectory passes that directory as
    ``prefix``; findings keep the receipt's own keys so callers' messages are
    unchanged.

    The returned report contains historical-source findings only. Working-tree
    drift is production of :func:`drift_paths`, not of this function, because it
    is a property of the checkout rather than of the receipt.
    """
    sources = sources if isinstance(sources, dict) else {}
    report = Report(revision=str(revision))
    if not isinstance(revision, str) or HEX40.fullmatch(revision) is None:
        report.findings.append(Finding(
            "source_revision", UNAVAILABLE,
            f"declared source revision {revision!r} is not a 40-character commit id"))
        return report
    if not commit_available(repo, revision):
        report.findings.append(Finding(
            "source_revision", UNAVAILABLE,
            f"declared source revision {revision[:12]} is unavailable: that commit "
            "object is not in this clone, so its bytes cannot be verified"))
        return report
    report.available = True

    pairs: dict[str, tuple[str, str]] = {}
    for raw, digest in sources.items():
        path = relative_path(raw)
        full = _join(prefix, path) if path is not None else ""
        if path is None or not full:
            report.findings.append(Finding(
                str(raw), UNSAFE, "path escapes the repository"))
            continue
        if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
            report.findings.append(Finding(
                path, INVALID, f"not a sha256 digest: {digest!r}"))
            continue
        pairs[path] = (full, digest)

    digests = _blob_digests(repo, revision, sorted({full for full, _ in pairs.values()}))
    for path in sorted(pairs):
        full, declared = pairs[path]
        actual = digests.get(full)
        if actual is None:
            report.findings.append(Finding(
                path, MISSING, f"absent at the declared revision {revision[:12]}"))
        elif actual != declared:
            report.findings.append(Finding(
                path, MISMATCH,
                f"declared {declared[:12]} actual {actual[:12]} at {revision[:12]}"))
        else:
            report.findings.append(Finding(path, VALID))
    return report


def drift_paths(repo: Path, sources: dict, prefix: str = "") -> list[str]:
    """Bound files whose bytes in this checkout differ from the receipt's.

    Advisory only: this says the checkout has moved beyond the receipt's
    revision, never that the receipt is wrong. A bound file that is absent from
    the checkout counts as drift because the checkout then cannot answer for the
    binding at all.
    """
    drift: list[str] = []
    if not isinstance(sources, dict):
        return drift
    for raw, digest in sources.items():
        path = relative_path(raw)
        full = _join(prefix, path) if path is not None else ""
        if path is None or not full:
            continue
        if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
            continue
        candidate = Path(repo) / full
        if not candidate.is_file() or sha256_file(candidate) != digest:
            drift.append(path)
    return sorted(drift)
