import json
from pathlib import Path

d = json.loads(
    (Path(__file__).resolve().parent / "raw" / "10_adapter_drift.json").read_text()
)
print("prefix_consistent:", d["trajectory_prefix_consistent"])
print("loss_at_steps:", json.dumps(d["loss_at_steps"]))
print("reproduction:", json.dumps(d["reproduction_of_phase_xiii_29"], indent=2))
print("dispatch:", json.dumps(d["dispatch"]))
print("worst tensor:", d["worst_tensor"], "monotone:", d["monotone_growth_worst_tensor"])
print()
header = "%-14s %5s %10s %10s %6s %9s %8s" % (
    "tensor",
    "steps",
    "max_abs",
    "rms",
    "n>.02",
    "p99",
    "rel_rms",
)
print(header)
print("-" * len(header))
for steps in ("6", "12", "29"):
    for name, row in sorted(d["drift_vs_steps"][steps].items()):
        short = (
            name.replace("base_model.model.model.layers.", "L")
            .replace(".self_attn.q_proj", "")
            .replace(".weight", "")
        )
        print(
            "%-14s %5s %10.6f %10.6f %6d %9.6f %8.5f"
            % (
                short,
                steps,
                row["max_abs"],
                row["rms"],
                row["elements_over_max_abs"],
                row["p99_abs"],
                row["relative_rms"],
            )
        )
