from __future__ import annotations

import json
import argparse
import subprocess
import sys
import hashlib
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "validation" / "verify_peft_lora_strict.py"
FIXTURES = ROOT / ".peft-oracle-fixtures"

# One explicit row per locally strict-qualified text family/config surface.
# Closely related fixture variants stay separate when they exercise materially
# different native graphs (hybrid/MoE/QSA/PLE/etc.).
def native_green_inventory():
    documentation = (ROOT / "COMPATIBILITY.md").read_text(encoding="utf-8")
    section = documentation.split("<!-- peft-native-green:start -->", 1)[1].split(
        "<!-- peft-native-green:end -->", 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4 or cells[-1] != "2e-7":
            continue
        fixtures = [value.strip() for value in cells[2].split(",")]
        if not fixtures or any(not value for value in fixtures):
            raise RuntimeError(f"base-green architecture has no PEFT follow-up fixture: {cells[0]}")
        rows.append({"architecture": cells[0], "model_types": cells[1], "fixtures": fixtures})
    if not rows:
        raise RuntimeError("native-green documentation inventory is empty")
    return rows


INVENTORY = native_green_inventory()
FAMILIES = [fixture for row in INVENTORY for fixture in row["fixtures"]]
if len(FAMILIES) != len(set(FAMILIES)):
    raise RuntimeError("duplicate fixture in native-green documentation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["lora", "switch", "saved", "all"], default="all")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--families", nargs="+", choices=FAMILIES,
                        help="run selected documented surfaces; omitted means all")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    families = args.families or FAMILIES
    # Keep the report's origin explicit: collect-only does not requalify a
    # binary, and a partial run must never overwrite the all-surface matrix.
    output = FIXTURES / ("selected-matrix-report.json" if args.families else "green-matrix-report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.collect_only:
        output.rename(output.with_name(output.stem + f".prior-{time.time_ns()}.json"))
    inputs = list((ROOT / "shaders").glob("*.spv"))
    inputs += list((ROOT / "shaders").glob("*.comp"))
    inputs += list((ROOT / "shaders").glob("*.glsl"))
    inputs += list((ROOT / "validation").glob("verify_peft_*.py"))
    # The requested local source clone is the mathematical oracle, not an
    # installed wheel. Fingerprint it alongside native executable inputs.
    transformers_root = Path(r"C:\Users\User\transformers")
    oracle_sources = sorted((transformers_root / "src/transformers").rglob("*.py"))
    if not oracle_sources:
        raise RuntimeError(f"local Transformers source is missing: {transformers_root}")
    inputs += list((ROOT / "src").rglob("*.rs"))
    inputs += [ROOT / "Cargo.toml", ROOT / "Cargo.lock", ROOT / "COMPATIBILITY.md"]
    inputs += [ROOT / "target/debug" / (name + (".exe" if sys.platform == "win32" else ""))
               for name in ["transformer_parity", "hierarchos-vulkan-transformer-logits", "peft_bank_parity"]]
    def fingerprint():
        native = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in inputs if path.is_file()}
        native.update({"local-transformers/" + path.relative_to(transformers_root).as_posix():
                       hashlib.sha256(path.read_bytes()).hexdigest() for path in oracle_sources})
        return native
    evidence = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "stage": args.stage, "collect_only": args.collect_only,
                "families": families, "entry_sha256": fingerprint()}
    evidence_path = output.with_name(output.stem + "-provenance.json")
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    def evaluate(family):
        print(f"[peft-matrix] {family}", flush=True)
        stages = {
            "lora": (ORACLE, "report.json"),
            "switch": (ROOT / "validation/verify_peft_multi_adapter.py", "multi-adapter/report.json"),
            "saved": (ROOT / "validation/verify_peft_saved_modules.py", "saved-modules-report.json"),
        }
        results = {}
        for stage, (script, relative) in stages.items():
            path = FIXTURES / f"{family}-strict" / relative
            if not args.collect_only and args.stage in (stage, "all"):
                # Delete only the old report, so a failed rerun cannot inherit
                # a green result. Preserve detailed process output per surface.
                path.unlink(missing_ok=True)
                log = FIXTURES / f"{family}-{stage}.log"
                with log.open("w", encoding="utf-8") as stream:
                    completed = subprocess.run([sys.executable, str(script), family],
                        cwd=ROOT, text=True, stdout=stream, stderr=subprocess.STDOUT)
                if completed.returncode:
                    results[stage] = {"pass": False, "returncode": completed.returncode, "log": str(log)}
                    continue
            if path.is_file():
                result = json.loads(path.read_text(encoding="utf-8"))
                results[stage] = {"pass": all(item["pass"] for item in result), "modules": result} if isinstance(result, list) else result
            else:
                results[stage] = {"pass": False, "status": "not run"}
        report_path = FIXTURES / f"{family}-strict" / "report.json"
        selected_pass = (
            all(value["pass"] for value in results.values())
            if args.stage == "all"
            else results[args.stage]["pass"]
        )
        if results["lora"]["pass"] and report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            row = {
                "family": family,
                "pass": selected_pass,
                "stages": results,
                "peft_max_abs": report["peft_max_abs"],
                "gradient_max_abs": report["gradient_max_abs"],
                "two_step_adamw_max_abs": report["two_step_adamw_max_abs"],
                "frozen_base_max_abs": report["frozen_base_max_abs"],
                "resume_hf_max_abs": report["resume_hf_max_abs"],
                "lifecycle_max_abs": max(report["lifecycle"].values(), default=0.0),
            }
        else:
            row = {
                "family": family,
                "pass": selected_pass,
                "stages": results,
            }
        print(json.dumps({"family": family, "stages": {name: value["pass"] for name, value in results.items()}}, sort_keys=True), flush=True)
        return row

    # Each surface owns its fixture directory. Stages within one surface stay
    # sequential; independent architectures can run concurrently.
    rows = []
    progress = output.with_name(output.stem + "-progress.json")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(evaluate, families):
            rows.append(row)
            progress.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    evidence["finished_utc"] = datetime.now(timezone.utc).isoformat()
    evidence["exit_sha256"] = fingerprint()
    evidence["inputs_unchanged"] = evidence["entry_sha256"] == evidence["exit_sha256"]
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    failures = [row["family"] for row in rows if not row["pass"]]
    print(json.dumps({"families": len(rows), "failures": failures}, indent=2))
    if failures or not evidence["inputs_unchanged"]:
        if not evidence["inputs_unchanged"]:
            print("Qualification invalid: source/shader/binary inputs changed during the run.", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
