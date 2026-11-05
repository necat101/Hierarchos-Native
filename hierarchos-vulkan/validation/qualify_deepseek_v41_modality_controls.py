"""Run the four V4.1 modality controls and reject empty or incomplete runs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

EXPECTED = {
    "deepseek_v41_modality_absent_gradients_match_text_path_bits",
    "deepseek_v41_modality_absent_optimizer_matches_train_sequence_bits",
    "deepseek_v41_modality_absent_optimizer_state_matches_train_sequence",
    "deepseek_v41_modality_hc_repeat_adjoint_matches_frozen_words",
}


def verify(log):
    results = re.findall(r"^test [^\r\n]*::(deepseek_v41_modality\w+) \.\.\. (?:ok[ \t]*$|[^\r\n]*\r?\n^ok[ \t]*$)", log, re.M)
    names = set(results)
    summary = re.search(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;", log)
    if names != EXPECTED or len(results) != len(EXPECTED) or summary is None:
        raise ValueError("Missing or duplicate modality control results")
    if tuple(map(int, summary.groups())) != (len(EXPECTED), 0, 0):
        raise ValueError("Incomplete modality control run")
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    result = subprocess.run([str(binary), "deepseek_v41_modality", "--test-threads=1", "--nocapture"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = result.stdout + result.stderr
    (args.output / "controls.log").write_text(log, encoding="utf-8")
    names = verify(log) if result.returncode == 0 else []
    passed = result.returncode == 0 and len(names) == len(EXPECTED)
    if hashlib.sha256(binary.read_bytes()).hexdigest() != binary_sha:
        raise ValueError("Control executable changed during the run")
    receipt = dict(schema=1, **{"pass": passed}, expected_cases=len(EXPECTED),
                   completed_cases=len(names), cases=names, returncode=result.returncode,
                   binary_sha256=binary_sha,
                   scope="Text-path/optimizer preservation and frozen HC adjoint; not full multimodal qualification")
    (args.output / "qualification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
