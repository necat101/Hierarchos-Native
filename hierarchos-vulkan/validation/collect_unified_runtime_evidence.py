"""Collect real test/GUI results and execute precision rejection controls.

This collector does not create passing test output. Required logs, binaries and
eight actual Start exports must exist. Rejection controls execute the real CLI
and trainer and check that no model/checkpoint was published.
"""
import argparse, json, os, re, subprocess
from pathlib import Path
from check_unified_release import ROOT,digest,require,verify_mode


def summary(path,binary,floor):
    text=path.read_text(errors='replace')
    rows=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',text)
    require(len(rows)==1,'missing/ambiguous successful test summary: '+str(path))
    passed,failed,ignored=map(int,rows[0]);require(passed>=floor and failed==0,'incomplete test execution')
    return dict(passed=passed,failed=failed,ignored=ignored,exit_code=0,log_sha256=digest(path),binary_sha256=digest(binary))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--gui-fixture',type=Path,help='Start-test input/output root when separate from pristine CPU inputs')
    p.add_argument('--precision-run',type=Path,required=True)
    p.add_argument('--libtest',type=Path,required=True)
    p.add_argument('--gui-test',type=Path,required=True)
    p.add_argument('--cli',type=Path,required=True)
    p.add_argument('--bins',type=Path,required=True)
    p.add_argument('--gui-prefix',default='phase22-gui')
    a=p.parse_args();out=a.work.resolve();out.mkdir(parents=True,exist_ok=False)
    fixture=a.fixture.resolve();binary=a.cli.resolve();bins=a.bins.resolve()
    tests={key:summary(ROOT/'target'/log,exe,floor) for key,log,exe,floor in [
        ('library','phase22-library-final.log',a.libtest,1159),
        ('cli','phase22-cli-unit.log',binary,50),('gui','phase22-gui-unit.log',a.gui_test,17),
        ('oversized_fallback','phase22-full-fallback-build-test-3.log',a.libtest,1),
        ('gui_hardware','phase22-gui-hardware-waited.log',a.gui_test,2)]}
    fallback=(ROOT/'target/phase22-full-fallback-build-test-3.log').read_text()
    require('reduction=128 forward=3 dW=3 dX=3' in fallback and 'cooperative delta=[0,0]' in fallback,'fallback test did not execute all FP32 phases')
    workflows=[]
    for mode in ('mixed','full-fp16'):
        for name in ('vision','full','full-finetune','lora'):
            path=(a.gui_fixture or fixture)/f'{a.gui_prefix}-{mode}-{name}'/'training_report.json'
            r=json.loads(path.read_text());require(r['optimizer_updates']==2 and r['steps']==2,'GUI update count mismatch')
            verify_mode(r['precision'],mode,name!='lora')
            workflows.append(dict(workflow=name,steps=r['steps'],optimizer_updates=r['optimizer_updates'],precision=r['precision'],
                raw_report_sha256=digest(path),raw_report=str(path)))
    controls=[]
    def reject(label,command,expected,output,mode):
        env={**os.environ,'HIERARCHOS_WMMA_MODE':mode,'HIERARCHOS_ATEN_VECTOR_WIDTH':'8'}
        run=subprocess.run(list(map(str,command)),env=env,cwd=ROOT,capture_output=True,text=True)
        text=run.stdout+run.stderr;(out/(label+'.log')).write_text(text)
        # Empty staging directories are not model/checkpoint publication.
        published=output.exists() and any(p.name in ('model.safetensors','adapter_model.safetensors','training_report.json',
            'hierarchos_adapter_optimizer.safetensors') for p in output.rglob('*'))
        require(run.returncode!=0 and expected.lower() in text.lower() and not published,'precision control unexpectedly accepted: '+label)
        controls.append(dict(control=label,command=list(map(str,command)),exit_code=run.returncode,
            expected_message=expected,output_created=published,log_sha256=digest(out/(label+'.log'))))
    base=[binary,'transformer-train','--model-path',fixture/'model/text','--train',fixture/'dataset.jsonl',
        '--batch-size','2','--seq-len','24','--epochs','1']
    reject('invalid-mode',base+['--wmma-mode','turbo','--out-dir',out/'invalid-mode'],'wmma',out/'invalid-mode','off')
    wide=ROOT/'target/unified-precision-fixture/model/text'
    reject('unqualified-full-profile',[binary,'transformer-train','--model-path',wide,'--train',fixture/'dataset.jsonl',
        '--batch-size','2','--seq-len','24','--epochs','1','--wmma-mode','full-fp16','--out-dir',out/'wrong-profile'],
        '64-wide',out/'wrong-profile','off')
    resume=a.precision_run.resolve()/'lora-mixed-state'
    reject('wrong-checkpoint-precision',[binary,'transformer-finetune','--model-path',fixture/'model/text',
        '--train',fixture/'dataset.jsonl','--out-dir',out/'wrong-resume','--batch-size','2','--seq-len','24','--epochs','1',
        '--wmma-mode','full-fp16','--lora-rank','8','--lora-alpha','16','--lora-dropout','0',
        '--lora-target-module','q_proj','--lora-target-module','v_proj','--peft-resume',resume],
        'precision',out/'wrong-resume','off')
    reject('partial-window-checkpoint',[bins/'hierarchos-vulkan-transformer-train.exe','--model',fixture/'model/text',
        '--dataset',fixture/'dataset.jsonl','--output',out/'partial-window','--batch-size','2','--seq-len','24',
        '--epochs','1','--max-steps','1','--gradient-accumulation-steps','2','--lora-rank','8','--lora-alpha','16',
        '--lora-target','q_proj','--lora-target','v_proj','--save-training-state',out/'partial-window/state'],
        'inside a gradient accumulation window',out/'partial-window','full-fp16')
    (out/'tests.json').write_text(json.dumps(tests,indent=2)+'\n')
    (out/'gui-cli.json').write_text(json.dumps(dict(status='PASS',cli_sha256=digest(binary),gui_test_sha256=digest(a.gui_test),
        workflows=workflows,unsupported_controls=controls,environment={'HIERARCHOS_ATEN_VECTOR_WIDTH':'8'}),indent=2)+'\n')
    print('RUNTIME EVIDENCE: eight GUI workflows, actual oversized fallback, complete test suites and rejection controls PASS',flush=True)


if __name__=='__main__':main()
