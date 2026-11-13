"""Inventory every intermediate diagnostic without changing the frozen gate.

The maintained qualifiers explicitly distinguish required checks from strict
intermediate diagnostics. This audit preserves both sets and all failing raw
maxima; it never promotes an internal tensor to strict equivalence because its
visible logits passed.
"""
import argparse, hashlib, json, math, subprocess
from collections import defaultdict
from pathlib import Path

STAGES={'shared77':77,'seq2seq':6,'gemma3':4,'gemma4-legacy':1,'gemma4-rgb':1,'llava':4}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();groups=defaultdict(list);raw=[];bindings={};required_cases=0
    for stage,count in STAGES.items():
        path=a.replay/stage/'qualification.json';doc=json.loads(path.read_text());bindings[stage]=digest(path)
        if not doc.get('pass') or len(doc['results'])!=count:raise ValueError('maintained FP32 stage failed: '+stage)
        for case in doc['results']:
            if not case['pass'] or not case['checks'] or not all(row['pass'] for row in case['checks']):
                raise ValueError('required FP32 check failed')
            required_cases+=1
            for row in case['strict_intermediate_diagnostics']:
                if not math.isfinite(row['max_abs']) or row['max_abs']<0:raise ValueError('invalid diagnostic')
                expected=row['max_abs']<2e-7
                if row['pass']!=expected:raise ValueError('diagnostic PASS disagrees with unchanged strict limit')
                record=dict(stage=stage,case=case.get('case',case.get('name')),**row)
                raw.append(record);groups[(stage,row['metric'].split('/',1)[-1])].append(record)
    if required_cases!=93 or len(raw)!=1116:raise ValueError('original image/diagnostic coverage changed')
    summary=[]
    for (family,tensor),rows in sorted(groups.items()):
        failing=[r for r in rows if not r['pass']]
        summary.append(dict(family=family,tensor=tensor,checks=len(rows),strict_failures=len(failing),
            max_abs=max(r['max_abs'] for r in rows),failed_cases=sorted({r['case'] for r in failing})))
    report=dict(schema=1,maintained_gate_status='PASS',diagnostic_status='FAIL' if any(not r['pass'] for r in raw) else 'PASS',
        maintained_cases=required_cases,diagnostic_checks=len(raw),strict_diagnostic_failures=sum(not r['pass'] for r in raw),
        worst_diagnostic=max(raw,key=lambda r:r['max_abs']),groups=summary,raw_diagnostics=raw,
        replay_sha256=bindings,frozen_manifest_sha256=digest(a.frozen/'HANDOFF-MANIFEST.json'),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        contract=dict(required='Original required visible logits and parameter comparisons remain strict max_abs < 2e-7, rtol=0; original loss/update/reload policies are unchanged.',
            intermediates='Every original intermediate retains its strict 2e-7 diagnostic verdict. Failing internal tensors are not qualified as CPU-identical; they are outside the original observable-value acceptance contract.',
            precision='FP32 storage/arithmetic does not imply identical CPU/GPU reduction, transcendental or contraction order for all internal tensors. No universal internal strict-equivalence claim is admitted.'))
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print('MAINTAINED FP32: PASS 93/93; INTERMEDIATE STRICT DIAGNOSTICS:',report['diagnostic_status'],
          report['strict_diagnostic_failures'],'/',len(raw),flush=True)


if __name__=='__main__':main()
