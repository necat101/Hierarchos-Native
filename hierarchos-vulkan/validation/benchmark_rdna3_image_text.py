"""Paired ABBA timings of packed vs serial RDNA3 norms with identical arithmetic.

Includes the composed probe's forward/backward, two updates, export and reload.
Each pair must produce identical report tensors and exported FP32 parameters.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from qualify_multimodal_training import digest, read_f32_tensors, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--case',default='gpt2_wide/unmasked')
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=3)
    a=p.parse_args()
    if a.rounds<2:raise ValueError('at least two ABBA rounds required')
    a.output.mkdir(parents=True,exist_ok=False)
    folder=a.fixtures.resolve()/a.case
    binary=a.binary.resolve();binary_hash=digest(binary)
    records=[];reference=None;timings={'serial':[],'packed':[]}
    for round_ in range(a.rounds):
        for pos,mode in enumerate(['serial','packed','packed','serial']):
            dest=a.output/f'{round_}-{pos}-{mode}'
            env={**os.environ,'HIERARCHOS_ATEN_VECTOR_WIDTH':'16',
                'HIERARCHOS_VULKAN_RDNA3_PACKED_NORMS':'1' if mode=='packed' else '0'}
            started=time.perf_counter()
            process=subprocess.run([str(binary),'--model',str(folder/'model'),'--fixture',str(folder/'input.json'),
                '--output',str(dest.resolve())],env=env,capture_output=True,text=True)
            (a.output/(dest.name+'.log')).write_text(process.stdout+process.stderr)
            process.check_returncode()
            report=json.loads((dest/'report.json').read_text())
            elapsed=report.pop('elapsed_seconds')
            state={f'{step}/{kind}':read_f32_tensors(dest/f'step-{step}'/kind/'model.safetensors')
                for step in [1,2] for kind in ['text','image']}
            value=(report,state)
            if reference is None:reference=value
            if value!=reference:raise ValueError('packing changed a report tensor or exported parameter')
            timings[mode].append(elapsed)
            records.append(dict(round=round_,position=pos,mode=mode,probe_seconds=elapsed,
                process_seconds=time.perf_counter()-started))
            print(mode,elapsed,flush=True)
    if digest(binary)!=binary_hash:raise ValueError('binary changed during timing')
    medians={k:statistics.median(v) for k,v in timings.items()}
    write_json(a.output/'benchmark.json',dict(case=a.case,binary_sha256=binary_hash,
        fixtures_sha256=digest(a.fixtures/'manifest.json'),rounds=a.rounds,records=records,
        medians_seconds=medians,packed_speedup=medians['serial']/medians['packed'],
        report_and_export_identity=True,scope='probe includes forward/backward, two updates, export/reload; not production-size training'))


if __name__=='__main__':main()
