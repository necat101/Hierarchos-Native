"""Collect hash-bound, strict CPU/PEFT streaming qualification attempts.

Failed attempts remain evidence; only passing reports with real streamed
linears and measured post-load retirement or a source-first loader contribute.
"""
import argparse
import hashlib
import json
from pathlib import Path
from qualify_parameter_streaming import FAMILIES,ROOT

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();profiles={};attempts=[]
    for folder in a.runs:
        path=folder/'qualification.json';payload=json.loads(path.read_text())
        assert (payload['strict_atol'],payload['rtol'])==(2e-7,0)
        binaries=payload.get('binaries',{})
        assert len(binaries)==2,'qualification requires a frozen executable snapshot'
        for name,binding in binaries.items():assert sha(folder/'bin'/name)==binding
        attempts.append(dict(path=str(path.resolve()),sha256=sha(path),binaries=binaries))
        for family,row in payload['results'].items():
            report=row.get('report',{})
            if row['exit_code'] or not report.get('pass'):continue
            assert family in FAMILIES and report['parameter_streaming']
            assert row['streamed_linears']>0 and (row['retired_parameter_bytes']>0 or row.get('source_first_parameters'))
            assert report['comparison']=='strict_less_than' and report['rtol']==0 and report['peft_tolerance']==2e-7
            metrics={k:v for k,v in report.items() if k.endswith('_max_abs')}
            metrics.update({'lifecycle.'+k:v for k,v in report['lifecycle'].items()})
            assert metrics and all(v<2e-7 for v in metrics.values()),(family,metrics)
            fixture=Path(row['fixture_directory'])
            assert json.loads((fixture/'report.json').read_text())==report
            profiles[family]=dict(streamed_linears=row['streamed_linears'],retired_parameter_bytes=row['retired_parameter_bytes'],
                source_first_parameters=row.get('source_first_parameters',False),
                loading_peak_buffer_bytes=row.get('loading_peak_buffer_bytes'),
                max_abs=max(metrics.values()),metrics=metrics,report_sha256=sha(fixture/'report.json'),
                report_path=str((fixture/'report.json').resolve()),attempt=str(path.resolve()))
    missing=sorted(set(FAMILIES)-profiles.keys())
    sources={}
    for name in ['src/transformer.rs','src/transformer/streamed_load.rs','src/transformer/peft_state.rs','src/weight_offload.rs','src/bin/transformer_logits.rs','src/bin/transformer_parity.rs',
                 'validation/verify_peft_lora_strict.py','validation/kimi_k3_reference.py']:
        sources[name]=sha(ROOT/'hierarchos-vulkan'/name)
    result=dict(status='qualified' if not missing else 'partial',comparison='strict_less_than',atol=2e-7,rtol=0,
        profiles_qualified=len(profiles),missing_profiles=missing,max_abs=max((r['max_abs'] for r in profiles.values()),default=None),
        scope='already-green CPU/PEFT profiles; frozen projection, routed/shared expert and vocabulary execution streaming',
        source_first_profiles=sorted(k for k,v in profiles.items() if v['source_first_parameters']),
        limits=['source-first body loading covers ordinary Llama/Mistral/Qwen2/Mixtral packages; other families convert after loading','specialized parameter operators outside VulkanLinear stay resident',
                'full base-weight training offload and encoder-decoder streaming excluded','146-family coverage and Intel Gen9 performance not claimed'],
        attempts=attempts,sources=sources,profiles=profiles)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ['status','profiles_qualified','missing_profiles','max_abs']},indent=2))
    return int(bool(missing))

if __name__=='__main__':raise SystemExit(main())
