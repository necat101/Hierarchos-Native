#!/usr/bin/env python3
"""Independent byte-integrity gate for the Phase LXXXI release evidence.

``phase_lxxxi_evidence/SHA256SUMS_phase_lxxxi.txt`` binds 612 evidence files by
SHA-256. Nothing else in the repository re-derives those digests, so a checkout
that rewrites the evidence bytes still reads as passing. Phase LXXXI was
committed before ``phase_lxxxi_evidence/** -text`` existed: Git normalised the
producer's CRLF evidence to LF, 397 of the 612 digests stopped resolving, and
``*.log`` kept the two retained logs out of the tree entirely.

This gate re-derives every digest twice, from two independent sources:

* the checked-out working tree, read as raw bytes; and
* the committed blob, read through ``git cat-file --batch`` plumbing.

A tree can therefore only pass when the committed bytes *and* the bytes on disk
both reproduce the frozen manifest. Working-tree and blob results are reported
separately so a divergence names which side moved.

The gate fails closed. A missing manifest, a malformed line, a duplicate path, an
absolute or escaping path, a symlink, an unreachable Git object, an unreadable
file, an inventory mismatch and a missing ``-text`` attribute are all failures,
never warnings.

Standard library only. It never imports the Phase LXXXI producer scripts and
never consults the producing worktree, so it runs unchanged on a clean clone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

EVIDENCE_DIR = "phase_lxxxi_evidence"
MANIFEST_NAME = "SHA256SUMS_phase_lxxxi.txt"
MANIFEST_REPO_PATH = f"{EVIDENCE_DIR}/{MANIFEST_NAME}"

# Lowercase hex only: the frozen manifest is GNU sha256sum output, and accepting
# an uppercase or short form would let a hand-edited line pass as canonical.
DIGEST_RE = re.compile(r"\A[0-9a-f]{64}\Z")
DRIVE_RE = re.compile(r"\A[A-Za-z]:")
RESERVED_RE = re.compile(r"\A(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?\Z", re.IGNORECASE)


class ManifestError(Exception):
    """The manifest itself is unusable."""


class GitError(Exception):
    """A Git plumbing command could not answer."""


@dataclass(frozen=True)
class Entry:
    digest: str
    path: str


@dataclass
class Report:
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    worktree_verified: int = 0
    worktree_failed: int = 0
    blob_verified: int = 0
    blob_failed: int = 0
    attributes_verified: int = 0
    attributes_failed: int = 0
    entries: int = 0
    tracked_evidence: int = 0
    missing_from_tree: list[str] = field(default_factory=list)
    unlisted_in_tree: list[str] = field(default_factory=list)
    extra_on_disk: list[str] = field(default_factory=list)
    manifest_sha256: str = ""

    def fail(self, message: str) -> None:
        self.failures.append(message)

    @property
    def ok(self) -> bool:
        return not self.failures


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_rel_path(rel: str) -> None:
    """Reject anything that is not a plain, canonical, repository-relative path."""
    if not rel:
        raise ManifestError("empty path")
    if "\x00" in rel:
        raise ManifestError("NUL byte in path")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in rel):
        raise ManifestError("control character in path")
    if "\\" in rel:
        raise ManifestError("backslash separator")
    if rel.startswith("/"):
        raise ManifestError("absolute path")
    if DRIVE_RE.match(rel):
        raise ManifestError("drive-letter path")
    pure = PurePosixPath(rel)
    if pure.is_absolute():
        raise ManifestError("absolute path")
    parts = rel.split("/")
    for part in parts:
        if part in ("", ".", ".."):
            raise ManifestError("empty, '.' or '..' path component")
        if RESERVED_RE.match(part):
            raise ManifestError("reserved device name")
        if part != part.strip():
            raise ManifestError("leading or trailing whitespace in path component")
        if part.endswith("."):
            raise ManifestError("trailing dot in path component")
    if str(pure) != rel:
        raise ManifestError("non-canonical path")


def parse_manifest(raw: bytes) -> list[Entry]:
    if not raw:
        raise ManifestError("manifest is empty")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"manifest is not valid UTF-8: {exc}") from exc
    if text.endswith("\n"):
        text = text[:-1]
    if not text:
        raise ManifestError("manifest has no entries")

    entries: list[Entry] = []
    seen: dict[str, int] = {}
    for number, line in enumerate(text.split("\n"), start=1):
        if line.endswith("\r"):
            line = line[:-1]
        if not line:
            raise ManifestError(f"line {number}: blank line")
        digest, separator, path = line.partition("  ")
        if not separator:
            raise ManifestError(f"line {number}: expected two spaces between digest and path")
        if not DIGEST_RE.match(digest):
            raise ManifestError(f"line {number}: malformed SHA-256 digest {digest!r}")
        try:
            validate_rel_path(path)
        except ManifestError as exc:
            raise ManifestError(f"line {number}: unsafe path {path!r}: {exc}") from exc
        if path in seen:
            raise ManifestError(
                f"line {number}: duplicate path {path!r} (first listed on line {seen[path]})"
            )
        seen[path] = number
        entries.append(Entry(digest=digest, path=path))
    return entries


def run_git(repo: Path, args: list[str], stdin: bytes | None = None) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            input=stdin,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise GitError(f"cannot run git: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise GitError(f"git {' '.join(args)} exited {completed.returncode}: {detail}")
    return completed.stdout


def resolve_revision(repo: Path, revision: str) -> str:
    out = run_git(repo, ["rev-parse", "--verify", f"{revision}^{{commit}}"])
    return out.decode("ascii").strip()


def tracked_paths(repo: Path, revision: str) -> set[str]:
    out = run_git(repo, ["ls-tree", "-r", "-z", "--name-only", revision, "--", EVIDENCE_DIR])
    result: set[str] = set()
    prefix = f"{EVIDENCE_DIR}/"
    for chunk in out.split(b"\x00"):
        if not chunk:
            continue
        name = chunk.decode("utf-8", "surrogateescape")
        if not name.startswith(prefix):
            raise GitError(f"unexpected ls-tree entry outside {EVIDENCE_DIR}: {name!r}")
        result.add(name[len(prefix):])
    return result


def blob_bytes(repo: Path, revision: str, paths: list[str]) -> dict[str, bytes | None]:
    """Read every committed blob as raw bytes through one ``cat-file --batch``."""
    if not paths:
        return {}
    specs = [f"{revision}:{EVIDENCE_DIR}/{path}".encode("utf-8") for path in paths]
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "--batch"],
            input=b"\n".join(specs) + b"\n",
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise GitError(f"cannot run git cat-file: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise GitError(f"git cat-file --batch exited {completed.returncode}: {detail}")

    out = completed.stdout
    position = 0
    result: dict[str, bytes | None] = {}
    for path in paths:
        newline = out.find(b"\n", position)
        if newline < 0:
            raise GitError(f"truncated cat-file output at {path!r}")
        header = out[position:newline]
        position = newline + 1
        if header.endswith(b" missing") or header.endswith(b" ambiguous"):
            result[path] = None
            continue
        fields = header.split(b" ")
        if len(fields) != 3 or fields[1] != b"blob":
            raise GitError(f"unexpected cat-file header for {path!r}: {header!r}")
        size = int(fields[2])
        body = out[position:position + size]
        if len(body) != size:
            raise GitError(f"truncated cat-file body for {path!r}")
        position += size + 1  # the record's own trailing newline
        result[path] = body
    return result


def attribute_values(repo: Path, repo_paths: list[str]) -> dict[str, str]:
    """Effective ``text`` attribute for every path, resolved by Git itself."""
    if not repo_paths:
        return {}
    payload = b"\x00".join(path.encode("utf-8") for path in repo_paths) + b"\x00"
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "check-attr", "-z", "--stdin", "text"],
            input=payload,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise GitError(f"cannot run git check-attr: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise GitError(f"git check-attr exited {completed.returncode}: {detail}")

    fields = completed.stdout.split(b"\x00")
    values: dict[str, str] = {}
    index = 0
    while index + 2 < len(fields):
        name, _attribute, value = fields[index], fields[index + 1], fields[index + 2]
        if not name:
            break
        values[name.decode("utf-8", "surrogateescape")] = value.decode("utf-8", "replace")
        index += 3
    return values


def _inside(root: Path, candidate: Path) -> bool:
    try:
        return os.path.commonpath([str(root), str(candidate)]) == str(root)
    except ValueError:  # different drives on Windows
        return False


def _link_reason(evidence: Path, resolved_evidence: Path, full: Path, rel: str) -> str | None:
    """Name the symlink or junction on the way to ``full``, if any."""
    current = evidence
    for part in rel.split("/"):
        current = current / part
        try:
            if current.is_symlink():
                return "symlink"
        except OSError:
            pass
        is_junction = getattr(os.path, "isjunction", None)
        if is_junction is not None:
            try:
                if is_junction(current):
                    return "junction"
            except OSError:
                pass
    try:
        resolved = Path(os.path.realpath(full))
    except OSError as exc:
        return f"unresolvable path ({exc})"
    if not _inside(resolved_evidence, resolved):
        return f"resolves outside {EVIDENCE_DIR} to {resolved}"
    return None


def check(repo_root: Path, revision: str, expect_manifest_sha256: str | None) -> Report:
    report = Report()
    repo = repo_root.resolve()
    evidence = repo / EVIDENCE_DIR
    manifest = evidence / MANIFEST_NAME

    if not (repo / ".git").exists() and not (repo / ".git").is_file():
        report.fail(f"{repo} is not a Git work tree")
        return report
    if not evidence.is_dir():
        report.fail(f"missing evidence directory {EVIDENCE_DIR}/")
        return report
    if not manifest.is_file():
        report.fail(f"missing manifest {MANIFEST_REPO_PATH}")
        return report

    manifest_raw = manifest.read_bytes()
    report.manifest_sha256 = sha256_bytes(manifest_raw)
    if expect_manifest_sha256 and report.manifest_sha256 != expect_manifest_sha256:
        report.fail(
            "manifest was replaced: expected "
            f"{expect_manifest_sha256}, found {report.manifest_sha256}"
        )

    try:
        entries = parse_manifest(manifest_raw)
    except ManifestError as exc:
        report.fail(f"manifest: {exc}")
        return report
    report.entries = len(entries)
    listed = {entry.path for entry in entries}

    try:
        commit = resolve_revision(repo, revision)
        tracked = tracked_paths(repo, commit)
    except GitError as exc:
        report.fail(f"Git inventory: {exc}")
        return report

    tracked_evidence = tracked - {MANIFEST_NAME}
    report.tracked_evidence = len(tracked_evidence)
    report.missing_from_tree = sorted(listed - tracked_evidence)
    report.unlisted_in_tree = sorted(tracked_evidence - listed)
    for path in report.missing_from_tree:
        report.fail(f"manifest lists a path absent from the committed tree: {path}")
    for path in report.unlisted_in_tree:
        report.fail(f"committed evidence file is absent from the manifest: {path}")

    # --- path safety and symlink/junction escapes -----------------------------
    resolved_evidence = Path(os.path.realpath(evidence))
    for entry in entries:
        full = evidence / entry.path.replace("/", os.sep)
        reason = _link_reason(evidence, resolved_evidence, full, entry.path)
        if reason is not None:
            report.fail(f"unsafe evidence path {entry.path}: {reason}")

    # --- working tree, raw bytes ---------------------------------------------
    for entry in entries:
        full = evidence / entry.path.replace("/", os.sep)
        try:
            if not full.is_file():
                report.worktree_failed += 1
                report.fail(f"working tree: missing {entry.path}")
                continue
            data = full.read_bytes()
        except OSError as exc:
            report.worktree_failed += 1
            report.fail(f"working tree: unreadable {entry.path}: {exc}")
            continue
        actual = sha256_bytes(data)
        if actual == entry.digest:
            report.worktree_verified += 1
        else:
            report.worktree_failed += 1
            report.fail(
                f"working tree: {entry.path} hashes {actual}, manifest says {entry.digest}"
            )

    # --- committed blobs, raw bytes ------------------------------------------
    try:
        blobs = blob_bytes(repo, commit, [entry.path for entry in entries])
    except GitError as exc:
        report.fail(f"Git blobs: {exc}")
        blobs = {}
    for entry in entries:
        blob = blobs.get(entry.path)
        if blob is None:
            report.blob_failed += 1
            report.fail(f"committed blob: missing or unreadable {entry.path}")
            continue
        actual = sha256_bytes(blob)
        if actual == entry.digest:
            report.blob_verified += 1
        else:
            report.blob_failed += 1
            report.fail(
                f"committed blob: {entry.path} hashes {actual}, manifest says {entry.digest}"
            )

    # --- effective -text attribute -------------------------------------------
    repo_paths = [f"{EVIDENCE_DIR}/{entry.path}" for entry in entries]
    try:
        attributes = attribute_values(repo, repo_paths)
    except GitError as exc:
        report.fail(f"Git attributes: {exc}")
        attributes = {}
    for repo_path in repo_paths:
        value = attributes.get(repo_path)
        if value is None:
            report.attributes_failed += 1
            report.fail(f"attributes: Git returned no text attribute for {repo_path}")
        elif value != "unset":
            report.attributes_failed += 1
            report.fail(
                f"attributes: {repo_path} has text={value}, expected unset (-text)"
            )
        else:
            report.attributes_verified += 1

    # --- nothing unexpected on disk ------------------------------------------
    found: set[str] = set()
    for directory, _subdirs, files in os.walk(evidence, followlinks=False):
        for name in files:
            full = Path(directory) / name
            found.add(str(full.relative_to(evidence)).replace(os.sep, "/"))
    report.extra_on_disk = sorted(found - listed - {MANIFEST_NAME})
    for path in report.extra_on_disk:
        report.fail(f"unexpected file in {EVIDENCE_DIR}/: {path}")

    report.notes.append(f"resolved revision {commit}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="repository root (default: two levels above this script)",
    )
    parser.add_argument("--rev", default="HEAD", help="commit to read the tracked tree from")
    parser.add_argument(
        "--expect-manifest-sha256",
        default=None,
        help="fail unless the manifest file hashes to this value (pins the frozen manifest)",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON only")
    args = parser.parse_args(argv)

    report = check(Path(args.repo_root), args.rev, args.expect_manifest_sha256)

    payload = {
        "repo_root": str(Path(args.repo_root).resolve()),
        "rev": args.rev,
        "manifest": MANIFEST_REPO_PATH,
        "manifest_sha256": report.manifest_sha256,
        "entries": report.entries,
        "tracked_evidence_paths": report.tracked_evidence,
        "working_tree": {
            "verified": report.worktree_verified,
            "failed": report.worktree_failed,
        },
        "git_blob": {
            "verified": report.blob_verified,
            "failed": report.blob_failed,
        },
        "attributes": {
            "verified": report.attributes_verified,
            "failed": report.attributes_failed,
        },
        "inventory": {
            "missing_from_tree": report.missing_from_tree,
            "unlisted_in_tree": report.unlisted_in_tree,
            "extra_on_disk": report.extra_on_disk,
        },
        "failures": report.failures,
        "ok": report.ok,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"manifest      {MANIFEST_REPO_PATH}  {report.manifest_sha256}")
        print(f"entries       {report.entries}")
        print(f"tracked       {report.tracked_evidence} evidence paths (manifest excluded)")
        print(f"working tree  {report.worktree_verified}/{report.entries} verified")
        print(f"git blobs     {report.blob_verified}/{report.entries} verified")
        print(f"attributes    {report.attributes_verified}/{report.entries} text=unset")
        for note in report.notes:
            print(f"note          {note}")
        if report.ok:
            print("RESULT        PASS")
        else:
            print(f"RESULT        FAIL ({len(report.failures)} failure(s))")
            for failure in report.failures:
                print(f"  - {failure}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
