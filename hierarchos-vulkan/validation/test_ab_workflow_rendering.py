#!/usr/bin/env python3
"""Contract tests for `validation/render_ab_workflow.py`.

The A/B workflow is generated from the route guard's spec table, so these tests
pin the two directions that matter: every spec in the table gets its full sweep
and its guard step in every job it declares, and the committed workflow is the
rendered one. Adding a producer to the guard therefore cannot leave the workflow
without that producer's measurement or its guard.
"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
import check_measured_producer_routes as guard  # noqa: E402
import render_ab_workflow as render  # noqa: E402


class WorkflowRenderTest(unittest.TestCase):
    def test_every_spec_has_the_step_fields_the_renderer_needs(self) -> None:
        for spec in guard.ALL_SPECS:
            with self.subTest(spec=spec.name):
                self.assertTrue(spec.label)
                self.assertTrue(spec.driver.startswith("hierarchos-vulkan/validation/"))
                self.assertTrue(spec.guard_label)
                self.assertTrue(spec.test_module)
                self.assertIn(spec.step_kind, {"ab", "composition"})
                self.assertTrue(spec.jobs)
                self.assertLessEqual(set(spec.jobs), set(render.JOBS))

    def test_each_job_renders_the_sweep_and_guard_of_the_specs_it_runs(self) -> None:
        for job in render.JOBS:
            block = render.render_steps(job)
            for spec in guard.ALL_SPECS:
                with self.subTest(job=job, spec=spec.name):
                    if job not in spec.jobs:
                        self.assertNotIn(f"--producer {spec.name}", block)
                        continue
                    self.assertIn(spec.driver, block)
                    self.assertIn(f"--producer {spec.name}", block)
                    self.assertIn(f"--record {render.relative(spec.receipt)}", block)
                    for batch in getattr(spec, "sweep", ()):
                        self.assertIn(f"--batch {batch}", block)
                    self.assertIn(
                        f"{spec.guard_label}{render.JOB_SUFFIX[job]}", block
                    )

    def test_every_job_has_at_least_one_producer(self) -> None:
        for job in render.JOBS:
            with self.subTest(job=job):
                self.assertIn("--producer ", render.render_steps(job))

    def test_every_spec_contributes_its_artifact_and_test_module(self) -> None:
        modules = render.test_modules()
        for job in render.JOBS:
            paths = render.artifact_paths(job)
            for spec in guard.ALL_SPECS:
                if job not in spec.jobs:
                    continue
                with self.subTest(job=job, spec=spec.name):
                    self.assertIn(render.relative(spec.receipt), paths)
                    self.assertIn(spec.test_module, modules)
                    for extra in spec.extra_artifacts:
                        self.assertIn(render.relative(extra), paths)
                    for module in spec.extra_test_modules:
                        self.assertIn(module, modules)

    def test_the_committed_workflow_uses_the_generated_lists(self) -> None:
        committed = render.WORKFLOW.read_text(encoding="utf-8")
        for job in render.JOBS:
            for path in render.artifact_paths(job):
                self.assertIn(path, committed)
        for module in render.test_modules():
            self.assertIn(module, committed)

    def test_the_producer_job_compares_a_fresh_ladder_receipt(self) -> None:
        """The hand-written ladder step is part of the contract, not a nicety.

        `workspace-check` pins the committed receipt on the CPU; only the GPU job
        can re-run the ladder, so a fresh receipt that drops a step or shrinks
        coverage has to be rejected there before `--record` can adopt it.
        """
        committed = render.WORKFLOW.read_text(encoding="utf-8")
        self.assertIn(
            "verify_training_stack.py --receipt target/stack_receipt.json", committed
        )
        self.assertIn(
            "check_training_stack_receipt.py --compare target/stack_receipt.json",
            committed,
        )

    def test_committed_workflow_is_the_rendered_one(self) -> None:
        committed = render.WORKFLOW.read_text(encoding="utf-8")
        self.assertEqual(
            render.render_workflow(committed),
            committed,
            "run `python hierarchos-vulkan/validation/render_ab_workflow.py --write`",
        )

    def test_a_new_producer_cannot_miss_its_sweep_or_its_guard(self) -> None:
        """The regression this generator exists for."""
        template = guard.PRODUCERS_BY_NAME["low-rank-forward"]
        added = dataclasses.replace(
            template,
            name="new-producer",
            label="New producer",
            driver="hierarchos-vulkan/validation/bench_new_producer.py",
            guard_label="New producer route across the sweep",
            receipt=render.ROOT / "validation" / "new_producer_bench.json",
            sweep=(1, 4, 8, 16, 32, 64),
            test_module="test_new_producer_receipt.py",
        )
        for job in render.JOBS:
            with self.subTest(job=job):
                block = render.render_steps(job, specs=guard.PRODUCERS + (added,))
                self.assertIn("--batch 64", block)
                self.assertIn("--producer new-producer", block)
                self.assertIn("new_producer_bench.json", block)

    def test_an_incomplete_producer_refuses_to_render(self) -> None:
        broken = dataclasses.replace(guard.PRODUCERS[0], label="", jobs=())
        with self.assertRaises(SystemExit):
            render.render_steps("producer-ab", specs=(broken,))

    def test_rendering_requires_the_markers(self) -> None:
        with self.assertRaises(SystemExit):
            render.render_workflow("jobs: {}\n")

    def test_the_cli_check_accepts_the_committed_workflow(self) -> None:
        self.assertEqual(render.main(["--check"]), 0)


if __name__ == "__main__":
    unittest.main()
