#!/usr/bin/env python3
"""Audit native Vulkan model-type coverage against this Transformers checkout.

This intentionally parses source text instead of importing ``transformers`` so
the audit remains usable while the Python environment has optional dependency
or version skew. It is a registry audit, not a numerical parity certificate.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path


MAPPINGS = {
    "causal_lm": "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES",
    "masked_lm": "MODEL_FOR_MASKED_LM_MAPPING_NAMES",
    "seq2seq_lm": "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES",
}


def _mapping_model_types(modeling_auto: str, mapping_name: str, _seen: frozenset[str] = frozenset()) -> set[str]:
    if mapping_name in _seen:
        raise RuntimeError(f"cyclic mapping reference: {mapping_name}")
    for node in ast.parse(modeling_auto).body:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == mapping_name for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Name):
            break
        if node.value.func.id != "OrderedDict":
            break
        if not node.value.args:
            return set()
        entries = node.value.args[0]
        if not isinstance(entries, (ast.List, ast.Tuple)):
            break
        result = set()
        for entry in entries.elts:
            if isinstance(entry, ast.Starred):
                expanded = entry.value
                if (
                    isinstance(expanded, ast.Call)
                    and isinstance(expanded.func, ast.Name)
                    and expanded.func.id == "list"
                    and len(expanded.args) == 1
                ):
                    expanded = expanded.args[0]
                if (
                    isinstance(expanded, ast.Call)
                    and not expanded.args
                    and not expanded.keywords
                    and isinstance(expanded.func, ast.Attribute)
                    and expanded.func.attr == "items"
                    and isinstance(expanded.func.value, ast.Name)
                ):
                    result.update(
                        _mapping_model_types(
                            modeling_auto,
                            expanded.func.value.id,
                            _seen | {mapping_name},
                        )
                    )
                    continue
                raise RuntimeError(f"unsupported mapping expansion in {mapping_name}")
            if not isinstance(entry, (ast.List, ast.Tuple)) or len(entry.elts) != 2:
                raise RuntimeError(f"non-literal entry in {mapping_name}")
            key = ast.literal_eval(entry.elts[0])
            if not isinstance(key, str):
                raise RuntimeError(f"non-string model type in {mapping_name}")
            result.add(key)
        return result
    raise RuntimeError(f"could not locate literal {mapping_name} in modeling_auto.py")


def _vulkan_model_types(transformer_rs: str) -> tuple[set[str], set[str]]:
    model_type_match = re.search(
        r"pub fn model_type\(self\).*?match self \{(.*?)\n\s*\}\n\s*\}",
        transformer_rs,
        re.DOTALL,
    )
    if model_type_match is None:
        raise RuntimeError("could not locate VulkanTransformerArchitecture::model_type")
    canonical = set(re.findall(r'=>\s*"([^"]+)"', model_type_match.group(1)))

    alias_match = re.search(
        r"HF_MODEL_TYPE_ALIASES.*?=\s*&\[(.*?)\];",
        transformer_rs,
        re.DOTALL,
    )
    if alias_match is None:
        raise RuntimeError("could not locate HF_MODEL_TYPE_ALIASES")
    aliases = set(re.findall(r'"([^"]+)"', alias_match.group(1)))
    return canonical, aliases


def _all_config_model_types(models_root: Path) -> set[str]:
    model_types: set[str] = set()
    pattern = re.compile(r'^\s*model_type\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
    for path in models_root.rglob("configuration_*.py"):
        model_types.update(pattern.findall(path.read_text(encoding="utf-8")))
    return model_types


def build_report(repo_root: Path, transformers_root: Path | None = None) -> dict[str, object]:
    transformers_root = transformers_root or repo_root
    modeling_auto_path = transformers_root / "src/transformers/models/auto/modeling_auto.py"
    transformer_rs_path = repo_root / "hierarchos-vulkan/src/transformer.rs"
    models_root = transformers_root / "src/transformers/models"
    for path in (modeling_auto_path, transformer_rs_path, models_root):
        if not path.exists():
            raise FileNotFoundError(path)

    modeling_auto = modeling_auto_path.read_text(encoding="utf-8")
    transformer_rs = transformer_rs_path.read_text(encoding="utf-8")
    canonical, aliases = _vulkan_model_types(transformer_rs)
    supported = canonical | aliases
    all_config_types = _all_config_model_types(models_root)

    report: dict[str, object] = {
        "coverage_basis": "model_type registry overlap only; not task implementation or numerical parity",
        "vulkan": {
            "canonical_model_types": len(canonical),
            "aliases": len(aliases),
            "advertised_model_types": len(supported),
        },
        "transformers": {
            "config_model_types": len(all_config_types),
            "advertised_config_coverage": len(supported & all_config_types),
            "missing_config_model_types": sorted(all_config_types - supported),
        },
        "task_mappings": {},
        "additional_task_registry_overlap": {},
    }
    task_mappings = report["task_mappings"]
    assert isinstance(task_mappings, dict)
    for label, mapping_name in MAPPINGS.items():
        model_types = _mapping_model_types(modeling_auto, mapping_name)
        covered = model_types & supported
        task_mappings[label] = {
            "total": len(model_types),
            "covered": len(covered),
            "missing": len(model_types - supported),
            "coverage_percent": round(100.0 * len(covered) / len(model_types), 2) if model_types else 100.0,
            "missing_model_types": sorted(model_types - supported),
        }
    # Classification, vision, audio and other AutoModel task heads are part of
    # the parity target too. A text-backbone alias is not proof that its tower
    # or task head executes: report registry overlap without calling it support.
    additional = report["additional_task_registry_overlap"]
    assert isinstance(additional, dict)
    mapping_names = re.findall(r"^(MODEL_FOR_\w+_MAPPING_NAMES)\s*=\s*OrderedDict", modeling_auto, re.MULTILINE)
    for mapping_name in mapping_names:
        if mapping_name in MAPPINGS.values():
            continue
        model_types = _mapping_model_types(modeling_auto, mapping_name)
        label = mapping_name.removeprefix("MODEL_FOR_").removesuffix("_MAPPING_NAMES").lower()
        additional[label] = {
            "total": len(model_types),
            "registry_overlap": sorted(model_types & supported),
            "missing_model_types": sorted(model_types - supported),
            "task_implementation_verified": False,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--all-tasks", action="store_true", help="also print all other AutoModel task registries")
    parser.add_argument(
        "--transformers-root",
        type=Path,
        help="path to the Transformers source checkout (or set TRANSFORMERS_CHECKOUT)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    transformers_root = args.transformers_root
    if transformers_root is None:
        configured = os.environ.get("TRANSFORMERS_CHECKOUT")
        transformers_root = Path(configured) if configured else repo_root
    report = build_report(repo_root, transformers_root)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    vulkan = report["vulkan"]
    transformers = report["transformers"]
    assert isinstance(vulkan, dict) and isinstance(transformers, dict)
    print(report["coverage_basis"])
    print(
        "Vulkan registry: "
        f"{vulkan['canonical_model_types']} canonical + {vulkan['aliases']} aliases = "
        f"{vulkan['advertised_model_types']} advertised model types"
    )
    print(
        "Transformers config registry: "
        f"{transformers['advertised_config_coverage']}/{transformers['config_model_types']} "
        "model types advertised by Vulkan"
    )
    task_mappings = report["task_mappings"]
    assert isinstance(task_mappings, dict)
    for label, stats in task_mappings.items():
        assert isinstance(stats, dict)
        print(
            f"{label}: {stats['covered']}/{stats['total']} registry overlap "
            f"({stats['coverage_percent']:.2f}%), {stats['missing']} missing"
        )
        missing = stats["missing_model_types"]
        if missing:
            print("  missing: " + ", ".join(missing))
    if args.all_tasks:
        additional = report["additional_task_registry_overlap"]
        assert isinstance(additional, dict)
        for label, stats in additional.items():
            print(
                f"{label}: {len(stats['registry_overlap'])}/{stats['total']} registry overlap; task support unverified"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
