"""Additional fixed weight seeds and top-3 expert accumulation at the same hard logit gate."""
import argparse
import hashlib
import json
from pathlib import Path
import validate_deepseek_v41_model as oracle

FIXTURE=oracle.FIXTURE.parent/"deepseek_v41_model_additional.json"
SPECS=[(4141018,2),(4141019,3)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path",type=Path,required=True)
    parser.add_argument("--kernel-reference-path",type=Path,required=True)
    parser.add_argument("--write-fixture",action="store_true")
    parser.add_argument("--native-output",type=Path)
    args=parser.parse_args()
    f={"cases":[oracle.make_fixture(args.reference_path,args.kernel_reference_path,seed=seed,topk=topk) for seed,topk in SPECS]}
    if args.write_fixture: FIXTURE.write_text(json.dumps(f,separators=(",",":"),allow_nan=False))
    elif json.loads(FIXTURE.read_text())!=f: raise AssertionError("additional fixtures differ from the pinned reference")
    print(f"PASS independent additional model oracle: seeds/topk {SPECS}")
    if args.native_output:
        native=json.loads(args.native_output.read_text())
        if len(native["cases"])!=len(f["cases"]): raise AssertionError("additional model count mismatch")
        failures=[]
        for actual,expected in zip(native["cases"],f["cases"]):
            if actual["seed"]!=expected["seed"]: raise AssertionError("additional model seed mismatch")
            print(f"MODEL seed={expected['seed']}, topk={expected['config']['text_config']['num_experts_per_tok']}")
            try: oracle.compare_native(expected,actual,fixture_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest())
            except AssertionError as error: failures.append(str(error))
        if failures: raise AssertionError(failures)


if __name__=="__main__":main()
