import struct
import tempfile
import unittest
from pathlib import Path

from check_push_constant_layouts import (
    GlslMember,
    SpirvModule,
    define_table,
    evaluate,
    glsl_block,
    host_problems,
    parse_glsl_push_block,
    preprocess,
    shader_problems,
)

OP_TYPE_INT = 21
OP_TYPE_POINTER = 32
OP_TYPE_STRUCT = 30
OP_VARIABLE = 59
OP_MEMBER_DECORATE = 72
OP_MEMBER_NAME = 6


def _string(text):
    raw = text.encode("utf-8") + b"\0"
    raw += b"\0" * (-len(raw) % 4)
    return list(struct.unpack(f"<{len(raw) // 4}I", raw))


def _instruction(opcode, operands):
    return [((len(operands) + 1) << 16) | opcode, *operands]


def _offset(widths):
    offsets, cursor = [], 0
    for width in widths:
        cursor += -cursor % width
        offsets.append(cursor)
        cursor += width
    return offsets


def spirv_module(members, named=True, offsets=None):
    """A minimal module with one push-constant block.

    `members` is a list of `(name, width)` pairs; `offsets` overrides the layout so
    a test can describe a module whose members are packed differently from the
    source it claims to come from.
    """
    widths = [width for _, width in members]
    offsets = offsets if offsets is not None else _offset(widths)
    limit = max((offset + width for offset, width in zip(offsets, widths)), default=0)

    words = [0x07230203, 0x00010000, 0, limit + 8, 0]
    uint, struct_type, pointer, variable = 1, 2, 3, 4
    words += _instruction(OP_TYPE_INT, [uint, 32, 0])
    words += _instruction(OP_TYPE_STRUCT, [struct_type, *[uint] * len(members)])
    words += _instruction(OP_TYPE_POINTER, [pointer, 9, struct_type])
    words += _instruction(OP_VARIABLE, [pointer, variable, 9])
    for index, (offset, (name, _)) in enumerate(zip(offsets, members)):
        words += _instruction(OP_MEMBER_DECORATE, [struct_type, index, 35, offset])
        if named:
            words += _instruction(OP_MEMBER_NAME, [struct_type, index, *_string(name)])
    return struct.pack(f"<{len(words)}I", *words)


def glsl(*lines):
    body = "\n".join(lines)
    return (
        "#version 450\n"
        "layout(local_size_x = 1) in;\n"
        "layout(push_constant) uniform Push {\n"
        f"{body}\n"
        "} pc;\n"
        "void main() {}\n"
    )


class SpirvBlockTests(unittest.TestCase):
    def test_reads_members_and_offsets(self):
        module = SpirvModule(spirv_module([("rows", 4), ("beta_scale", 4)]))
        block = module.push_block()
        self.assertIsNotNone(block)
        self.assertEqual([(m.name, m.offset, m.size) for m in block.members], [
            ("rows", 0, 4),
            ("beta_scale", 4, 4),
        ])
        self.assertEqual(block.size, 8)

    def test_module_without_push_block_is_none(self):
        words = [0x07230203, 0x00010000, 0, 5, 0]
        module = SpirvModule(struct.pack(f"<{len(words)}I", *words))
        self.assertIsNone(module.push_block())


class GlslBlockTests(unittest.TestCase):
    def test_scalars_pack_at_four_bytes(self):
        parsed = parse_glsl_push_block(glsl("uint rows;", "float beta_scale;"))
        block, _ = glsl_block(parsed[1])
        self.assertEqual([(m.name, m.offset, m.size) for m in block.members], [
            ("rows", 0, 4),
            ("beta_scale", 4, 4),
        ])

    def test_vec3_aligns_to_sixteen_and_keeps_twelve_bytes(self):
        parsed = parse_glsl_push_block(glsl("uint rows;", "vec3 axis;", "float tail;"))
        block, _ = glsl_block(parsed[1])
        self.assertEqual([(m.offset, m.size) for m in block.members], [
            (0, 4),
            (16, 12),
            (28, 4),
        ])

    def test_arrays_use_a_tight_element_stride(self):
        parsed = parse_glsl_push_block(glsl("uint rows;", "uint shape[2];"))
        block, _ = glsl_block(parsed[1])
        self.assertEqual(block.size, 12)
        self.assertEqual([m.name for m in block.members], ["rows", "shape"])


class PreprocessorTests(unittest.TestCase):
    def test_undefined_macro_drops_the_member(self):
        source = glsl("uint rows;", "#if PACKED", "uint offset;", "#endif")
        parsed = parse_glsl_push_block(preprocess_source(source, {}))
        self.assertEqual([m.name for m in parsed[1]], ["rows"])

    def test_pinned_macro_keeps_the_member(self):
        source = glsl("uint rows;", "#if PACKED", "uint offset;", "#endif")
        parsed = parse_glsl_push_block(preprocess_source(source, {"PACKED": "1"}))
        self.assertEqual([m.name for m in parsed[1]], ["rows", "offset"])

    def test_elif_does_not_re_enable_a_taken_branch(self):
        source = glsl(
            "#if A",
            "uint first;",
            "#elif B",
            "uint second;",
            "#else",
            "uint third;",
            "#endif",
        )
        self.assertEqual([m.name for m in parse_glsl_push_block(preprocess_source(source, {"A": "1"}))[1]], ["first"])
        self.assertEqual([m.name for m in parse_glsl_push_block(preprocess_source(source, {"B": "1"}))[1]], ["second"])
        self.assertEqual([m.name for m in parse_glsl_push_block(preprocess_source(source, {}))[1]], ["third"])

    def test_define_flags_become_macros(self):
        self.assertEqual(define_table(["-DFUSE=1", "-DCOUNT=4", "-O"]), {"FUSE": "1", "COUNT": "4"})

    def test_comparisons_and_negation(self):
        self.assertTrue(evaluate("COUNT == 4", {"COUNT": "4"}))
        self.assertFalse(evaluate("!FUSE", {"FUSE": "1"}))
        self.assertTrue(evaluate("defined(FUSE) && !defined(OTHER)", {"FUSE": "1"}))

    def test_undefined_macro_is_zero(self):
        # This is how a variant pins a conditional member off: the flag is absent.
        self.assertFalse(evaluate("PACKED", {}))
        self.assertFalse(evaluate("!defined(X) && OTHER", {"X": "1"}))

    def test_unsupported_expression_is_reported_rather_than_guessed(self):
        self.assertIsNone(evaluate("defined(X) ? 1 : 0", {"X": "1"}))


def preprocess_source(source, macros):
    with tempfile.TemporaryDirectory(prefix="push-constant-") as scratch:
        path = Path(scratch) / "demo.comp"
        path.write_text(source, encoding="utf-8")
        return preprocess(path, macros)


class ShaderComparisonTests(unittest.TestCase):
    def _compare(self, module_members, source, named=True, offsets=None):
        with tempfile.TemporaryDirectory(prefix="push-constant-") as scratch:
            root = Path(scratch)
            spv = root / "demo.spv"
            spv.write_bytes(spirv_module(module_members, named=named, offsets=offsets))
            comp = root / "demo.comp"
            comp.write_text(glsl(*source), encoding="utf-8")
            return shader_problems("demo", spv, comp)

    def test_matching_block_passes(self):
        problems, compared = self._compare([("rows", 4), ("num_value_heads", 4)], ["uint rows;", "uint num_value_heads;"])
        self.assertTrue(compared)
        self.assertEqual(problems, [])

    def test_member_added_to_source_is_a_stale_artifact(self):
        # The exact drift that shipped: a module built before `beta_scale` existed.
        problems, compared = self._compare(
            [("rows", 4), ("num_value_heads", 4)],
            ["uint rows;", "uint num_value_heads;", "float beta_scale;"],
        )
        self.assertTrue(compared)
        self.assertEqual(len(problems), 1)
        self.assertIn("beta_scale", problems[0])

    def test_source_that_declares_no_block_but_module_has_one_fails(self):
        with tempfile.TemporaryDirectory(prefix="push-constant-") as scratch:
            root = Path(scratch)
            spv = root / "demo.spv"
            spv.write_bytes(spirv_module([("rows", 4)]))
            comp = root / "demo.comp"
            comp.write_text("#version 450\nvoid main() {}\n", encoding="utf-8")
            problems, _ = shader_problems("demo", spv, comp)
        self.assertEqual(len(problems), 1)

    def test_nothing_to_compare_is_not_a_failure(self):
        with tempfile.TemporaryDirectory(prefix="push-constant-") as scratch:
            root = Path(scratch)
            spv = root / "demo.spv"
            words = [0x07230203, 0x00010000, 0, 5, 0]
            spv.write_bytes(struct.pack(f"<{len(words)}I", *words))
            comp = root / "demo.comp"
            comp.write_text("#version 450\nvoid main() {}\n", encoding="utf-8")
            problems, compared = shader_problems("demo", spv, comp)
        self.assertEqual(problems, [])
        self.assertFalse(compared)

    def test_reordered_members_are_reported(self):
        problems, _ = self._compare(
            [("rows", 4), ("num_heads", 4)],
            ["uint num_heads;", "uint rows;"],
        )
        self.assertEqual(len(problems), 1)

    def test_offsets_are_compared_even_without_member_names(self):
        problems, _ = self._compare(
            [("rows", 4), ("heads", 4)],
            ["uint rows;", "vec4 heads;"],
            named=False,
        )
        self.assertEqual(len(problems), 1)


class HostComparisonTests(unittest.TestCase):
    def _block(self, *widths):
        module = SpirvModule(spirv_module([(f"m{index}", width) for index, width in enumerate(widths)]))
        return module.push_block()

    def test_struct_wider_than_block_is_allowed(self):
        # The documented shared-head case: the extra loss words are pushed but the
        # module reads only the leading geometry words.
        problems, compared = host_problems(
            "demo",
            "WidePush",
            [("rows", "u32"), ("dim", "u32"), ("loss_scale", "f32")],
            "demo.spv",
            self._block(4, 4),
        )
        self.assertTrue(compared)
        self.assertEqual(problems, [])

    def test_host_narrower_than_block_fails(self):
        problems, compared = host_problems(
            "demo",
            "NarrowPush",
            [("rows", "u32")],
            "demo.spv",
            self._block(4, 4),
        )
        self.assertTrue(compared)
        self.assertEqual(len(problems), 1)
        self.assertIn("fewer words", problems[0])

    def test_prefix_with_a_different_width_fails(self):
        problems, _ = host_problems(
            "demo",
            "WidePush",
            [("rows", "u64"), ("dim", "u32")],
            "demo.spv",
            self._block(4, 4),
        )
        self.assertEqual(len(problems), 1)

    def test_struct_field_names_do_not_have_to_match(self):
        # The host renames fields for its own clarity (`rows_or_batch`), and that
        # is fine as long as the widths line up in order.
        problems, _ = host_problems(
            "demo",
            "RenamedPush",
            [("rows_or_batch", "u32")],
            "demo.spv",
            self._block(4),
        )
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
