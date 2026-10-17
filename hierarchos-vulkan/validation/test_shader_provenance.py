import json
import tempfile
import unittest
from pathlib import Path

from check_shader_provenance import (
    Toolchain,
    embedded_artifacts,
    evaluate,
    parse_version,
    read_exceptions,
    read_manifest,
    record_versions,
)


def entry(**overrides):
    base = {
        "artifact": "demo.spv",
        "source": "demo.comp",
        "compiler": "glslangValidator",
        "flags": ["-V", "--target-env", "vulkan1.0"],
    }
    base.update(overrides)
    return base


def glslang(version="16.5.0", path="/fake/glslang"):
    return Toolchain(path, version, f"Glslang Version: 11:{version}")


class ParseVersionTests(unittest.TestCase):
    def test_glslang_banner(self):
        self.assertEqual(parse_version("Glslang Version: 11:16.5.0"), "16.5.0")

    def test_last_dotted_triple_wins(self):
        self.assertEqual(parse_version("shaderc 1.2.3 over glslang 16.6.0"), "16.6.0")

    def test_shaderc_reports_its_upstream_release(self):
        # The conda-forge package version (2025.3) is not what glslc prints; the
        # release it reports is the one a pin has to name.
        self.assertEqual(
            parse_version("shaderc v2023.8 unknown hash, 2025-08-31\n\nTarget: SPIR-V 1.0"),
            "2023.8",
        )

    def test_different_shaderc_packages_report_different_releases(self):
        self.assertEqual(
            parse_version("shaderc v2026.4 unknown hash, 2026-10-14\n\nTarget: SPIR-V 1.0"),
            "2026.4",
        )

    def test_spirv_tools_banner(self):
        self.assertEqual(parse_version("SPIRV-Tools 2025.5 2025.5, 0\nTargets:"), "2025.5")

    def test_only_the_release_line_is_read(self):
        # A later line names an embedded component, which is not the toolchain.
        banner = "shaderc v2023.8 unknown hash, 2025-08-31\n\nglslang 16.6.0\nTarget: SPIR-V 1.0"
        self.assertEqual(parse_version(banner), "2023.8")

    def test_banner_without_a_release(self):
        self.assertIsNone(parse_version("unknown build"))
        self.assertIsNone(parse_version(""))


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="provenance-test-")
        self.root = Path(self.scratch.name)
        (self.root / "shaders").mkdir()
        (self.root / "shaders" / "demo.comp").write_text("void main() {}\n")
        (self.root / "shaders" / "demo.spv").write_bytes(b"embedded")

    def tearDown(self):
        self.scratch.cleanup()

    def run_evaluate(self, entries, rebuild, **overrides):
        options = {
            "toolchains": {"glslangValidator": [glslang()]},
            "allow_missing": frozenset(),
            "required_versions": {},
            "root": self.root,
        }
        options.update(overrides)
        return evaluate(entries, rebuild, **options)

    def test_identical_artifact_passes(self):
        def rebuild(_entry, _compiler, out):
            out.write_bytes(b"embedded")
            return 0, ""

        rows, failures = self.run_evaluate([entry(compiler_version="16.5.0")], rebuild)
        self.assertEqual(rows[0]["status"], "identical")
        self.assertEqual(rows[0]["version"], "16.5.0")
        self.assertEqual(rows[0]["version_source"], "manifest")
        self.assertEqual(failures, [])

    def test_drifted_artifact_fails(self):
        def rebuild(_entry, _compiler, out):
            out.write_bytes(b"different")
            return 0, ""

        rows, failures = self.run_evaluate([entry(compiler_version="16.5.0")], rebuild)
        self.assertEqual(rows[0]["status"], "drifted")
        self.assertNotEqual(rows[0]["embedded_sha256"], rows[0]["rebuilt_sha256"])
        self.assertEqual(len(failures), 1)

    def test_compile_failure_is_reported_with_output(self):
        def rebuild(_entry, _compiler, _out):
            return 1, "error: 0:3: 'x' : undeclared identifier"

        rows, failures = self.run_evaluate([entry()], rebuild)
        self.assertEqual(rows[0]["status"], "compile_failed")
        self.assertIn("undeclared identifier", rows[0]["error"])
        self.assertEqual(len(failures), 1)

    def test_missing_compiler_fails_unless_allowed(self):
        def rebuild(_entry, _compiler, _out):  # pragma: no cover - must not run
            raise AssertionError("rebuild should not be called without a compiler")

        rows, failures = self.run_evaluate([entry()], rebuild, toolchains={})
        self.assertEqual(rows[0]["status"], "missing_compiler")
        self.assertEqual(len(failures), 1)

        rows, failures = self.run_evaluate(
            [entry()],
            rebuild,
            toolchains={},
            allow_missing=frozenset({"glslangValidator"}),
        )
        self.assertEqual(rows[0]["status"], "skipped")
        self.assertEqual(failures, [])

    def test_recorded_release_selects_the_matching_candidate(self):
        seen = []

        def rebuild(_entry, compiler, out):
            seen.append(compiler)
            out.write_bytes(b"embedded")
            return 0, ""

        rows, failures = self.run_evaluate(
            [entry(compiler_version="16.5.0")],
            rebuild,
            toolchains={
                "glslangValidator": [
                    glslang("16.6.0", "/fake/glslang-16.6.0"),
                    glslang("16.5.0", "/fake/glslang-16.5.0"),
                ]
            },
        )
        self.assertEqual(rows[0]["status"], "identical")
        self.assertEqual(seen, ["/fake/glslang-16.5.0"])
        self.assertEqual(rows[0]["compiler_path"], "/fake/glslang-16.5.0")
        self.assertNotIn("toolchain_ambiguous", rows[0])
        self.assertEqual(failures, [])

    def test_a_spanned_tree_is_checked_by_one_run(self):
        # Two artifacts built by two different releases, as a drifted tree looks
        # before it is rebuilt under one compiler.
        def rebuild(recipe, compiler, out):
            out.write_bytes(b"embedded" if recipe["artifact"] == "demo.spv" else b"other")
            return 0, ""

        old = entry(artifact="demo.spv", source="demo.comp", compiler_version="16.5.0")
        new = entry(artifact="other.spv", source="demo.comp", compiler_version="16.6.0")
        (self.root / "shaders" / "other.spv").write_bytes(b"other")
        rows, failures = self.run_evaluate(
            [old, new],
            rebuild,
            toolchains={
                "glslangValidator": [
                    glslang("16.6.0", "/fake/new"),
                    glslang("16.5.0", "/fake/old"),
                ]
            },
        )
        self.assertEqual([row["status"] for row in rows], ["identical", "identical"])
        self.assertEqual([row["compiler_path"] for row in rows], ["/fake/old", "/fake/new"])
        self.assertEqual(failures, [])

    def test_no_candidate_reports_the_recorded_release(self):
        def rebuild(_entry, _compiler, out):  # pragma: no cover - must not run
            raise AssertionError("a mismatched toolchain should not be used to compile")

        rows, failures = self.run_evaluate(
            [entry(compiler_version="16.5.0")],
            rebuild,
            toolchains={"glslangValidator": [glslang("16.6.0")]},
        )
        self.assertEqual(rows[0]["status"], "wrong_version")
        self.assertEqual(rows[0]["expected_version"], "16.5.0")
        self.assertEqual(rows[0]["observed_versions"], ["16.6.0"])
        self.assertEqual(len(failures), 1)

    def test_unrecorded_release_uses_the_first_candidate(self):
        def rebuild(_entry, _compiler, out):
            out.write_bytes(b"embedded")
            return 0, ""

        rows, failures = self.run_evaluate([entry()], rebuild)
        self.assertEqual(rows[0]["status"], "identical")
        self.assertEqual(rows[0]["version"], "16.5.0")
        self.assertIsNone(rows[0]["version_source"])
        self.assertEqual(failures, [])

        rows, _ = self.run_evaluate(
            [entry()],
            rebuild,
            toolchains={"glslangValidator": [glslang("16.5.0", "/first"), glslang("16.6.0", "/second")]},
        )
        self.assertEqual(rows[0]["compiler_path"], "/first")
        self.assertTrue(rows[0]["toolchain_ambiguous"])

    def test_flag_pin_applies_to_an_unrecorded_entry(self):
        def rebuild(_entry, _compiler, out):
            out.write_bytes(b"embedded")
            return 0, ""

        rows, failures = self.run_evaluate(
            [entry()], rebuild, required_versions={"glslangValidator": "16.5.0"}
        )
        self.assertEqual(rows[0]["status"], "identical")
        self.assertEqual(rows[0]["version_source"], "flag")
        self.assertEqual(failures, [])

    def test_a_flag_conflicting_with_a_recorded_release_fails(self):
        def rebuild(_entry, _compiler, out):  # pragma: no cover - must not run
            raise AssertionError("a conflicting pin should be decided before compiling")

        rows, failures = self.run_evaluate(
            [entry(compiler_version="16.5.0")],
            rebuild,
            required_versions={"glslangValidator": "16.6.0"},
        )
        self.assertEqual(rows[0]["status"], "pin_conflict")
        self.assertEqual(rows[0]["expected_version"], "16.5.0")
        self.assertEqual(rows[0]["conflicting_version"], "16.6.0")
        self.assertEqual(len(failures), 1)

    def test_absent_source_or_artifact_fails(self):
        def rebuild(_entry, _compiler, out):
            out.write_bytes(b"embedded")
            return 0, ""

        missing_source, failures = self.run_evaluate([entry(source="gone.comp")], rebuild)
        self.assertEqual(missing_source[0]["status"], "missing_source")
        self.assertEqual(len(failures), 1)

        (self.root / "shaders" / "demo.spv").unlink()
        missing_artifact, failures = self.run_evaluate([entry()], rebuild)
        self.assertEqual(missing_artifact[0]["status"], "missing_artifact")
        self.assertEqual(len(failures), 1)


class RecordVersionsTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="provenance-record-")
        self.path = Path(self.scratch.name) / "training_shader_manifest.json"

    def tearDown(self):
        self.scratch.cleanup()

    def write(self, entries, crlf=False):
        text = json.dumps({"schema": 1, "shaders": entries}, indent=2) + "\n"
        if crlf:
            text = text.replace("\n", "\r\n")
        self.path.write_bytes(text.encode("utf-8"))

    def test_records_only_artifacts_that_reproduced(self):
        self.write(
            [
                {"artifact": "a.spv", "source": "a.comp", "compiler": "glslc", "flags": ["-O"]},
                {"artifact": "b.spv", "source": "b.comp", "compiler": "glslc", "flags": []},
            ]
        )
        rows = [
            {"artifact": "a.spv", "status": "identical", "version": "2023.8"},
            {"artifact": "b.spv", "status": "drifted", "version": "2023.8"},
        ]
        self.assertEqual(record_versions(self.path, rows), (1, 0, 1))
        document = json.loads(self.path.read_text(encoding="utf-8"))
        first, second = document["shaders"]
        self.assertEqual(first["compiler_version"], "2023.8")
        self.assertNotIn("compiler_version", second)
        self.assertEqual(list(first), ["artifact", "source", "compiler", "compiler_version", "flags"])

    def test_an_already_recorded_release_is_left_alone(self):
        self.write(
            [{"artifact": "a.spv", "source": "a.comp", "compiler": "glslc", "flags": [], "compiler_version": "2023.8"}]
        )
        rows = [{"artifact": "a.spv", "status": "identical", "version": "2023.8"}]
        self.assertEqual(record_versions(self.path, rows), (0, 1, 0))

    def test_line_endings_are_preserved(self):
        self.write(
            [{"artifact": "a.spv", "source": "a.comp", "compiler": "glslc", "flags": []}], crlf=True
        )
        record_versions(self.path, [{"artifact": "a.spv", "status": "identical", "version": "2023.8"}])
        raw = self.path.read_bytes()
        self.assertIn(b"\r\n", raw)
        self.assertNotIn(b"\n", raw.replace(b"\r\n", b""))


class ManifestTests(unittest.TestCase):
    def test_committed_manifest_is_complete(self):
        manifest = Path(__file__).resolve().parents[1] / "validation" / "training_shader_manifest.json"
        entries, problems = read_manifest(manifest, Path(__file__).resolve().parents[1])
        self.assertEqual(problems, [])
        self.assertEqual(len(entries), len({item["artifact"] for item in entries}))
        self.assertTrue(any(item["compiler"] == "glslangValidator" for item in entries))
        # Every artifact records the release that reproduces it, which is what lets
        # the check select a toolchain per artifact.
        self.assertTrue(all(item.get("compiler_version") for item in entries))

    def test_nested_artifact_paths_are_addressable(self):
        # A frozen module that lives in a subdirectory is pinned by its path under
        # shaders/, e.g. the pre-tuning scalar fallbacks in shaders/portable_scalar/.
        with tempfile.TemporaryDirectory(prefix="provenance-nested-") as scratch:
            root = Path(scratch)
            (root / "shaders" / "portable_scalar").mkdir(parents=True)
            (root / "shaders" / "portable_scalar" / "module.spv").write_bytes(b"spv")
            (root / "shaders" / "portable_scalar" / "module.comp").write_text("#version 450\n")
            path = root / "manifest.json"
            path.write_text(json.dumps({"schema": 1, "shaders": [{
                "artifact": "portable_scalar/module.spv",
                "source": "portable_scalar/module.comp",
                "compiler": "glslc",
                "flags": [],
            }]}))
            entries, problems = read_manifest(path, root)
            self.assertEqual(problems, [])
            self.assertEqual(entries[0]["artifact"], "portable_scalar/module.spv")

    def test_every_embedded_artifact_is_pinned_or_explained(self):
        # The manifest is only worth something if nothing dispatched escapes it:
        # every `include_bytes!` module must be pinned, or listed in the exception
        # file with a reason, and an exception that no longer applies must fail.
        root = Path(__file__).resolve().parents[1]
        embedded = {path.removeprefix("shaders/") for path in embedded_artifacts(root)}
        manifest = root / "validation" / "training_shader_manifest.json"
        pinned = {item["artifact"] for item in json.loads(manifest.read_text(encoding="utf-8"))["shaders"]}
        exceptions = read_exceptions(root / "validation" / "shader_provenance_exceptions.json")
        missing = sorted(embedded - pinned - set(exceptions))
        self.assertEqual(missing, [], "embedded modules without provenance: " + ", ".join(missing))
        stale = sorted(set(exceptions) - embedded)
        self.assertEqual(stale, [], "exceptions that are no longer embedded: " + ", ".join(stale))
        self.assertEqual([name for name, reason in exceptions.items() if not reason.strip()], [])

    def test_unsupported_schema_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="provenance-manifest-") as scratch:
            root = Path(scratch)
            path = root / "manifest.json"
            path.write_text(json.dumps({"schema": 2, "shaders": [entry()]}))
            entries, problems = read_manifest(path, root)
            self.assertEqual(entries, [])
            self.assertTrue(problems)

    def test_entry_keys_are_checked(self):
        with tempfile.TemporaryDirectory(prefix="provenance-manifest-") as scratch:
            root = Path(scratch)
            (root / "shaders").mkdir()
            path = root / "manifest.json"
            path.write_text(json.dumps({"schema": 1, "shaders": [{"artifact": "demo.spv"}]}))
            _, problems = read_manifest(path, root)
            self.assertTrue(any("missing keys" in problem for problem in problems))

    def test_uncommitted_artifact_is_reported(self):
        with tempfile.TemporaryDirectory(prefix="provenance-manifest-") as scratch:
            root = Path(scratch)
            (root / "shaders").mkdir()
            path = root / "manifest.json"
            path.write_text(json.dumps({"schema": 1, "shaders": [entry()]}))
            _, problems = read_manifest(path, root)
            self.assertTrue(any("not committed" in problem for problem in problems))

    def test_recorded_release_must_be_a_non_empty_string(self):
        with tempfile.TemporaryDirectory(prefix="provenance-manifest-") as scratch:
            root = Path(scratch)
            (root / "shaders").mkdir()
            (root / "shaders" / "demo.spv").write_bytes(b"embedded")
            path = root / "manifest.json"
            for recorded in ("", "   ", 1650, None, ["16.5.0"]):
                path.write_text(
                    json.dumps({"schema": 1, "shaders": [entry(compiler_version=recorded)]})
                )
                _, problems = read_manifest(path, root)
                self.assertTrue(
                    any("compiler_version" in problem for problem in problems), recorded
                )


if __name__ == "__main__":
    unittest.main()
