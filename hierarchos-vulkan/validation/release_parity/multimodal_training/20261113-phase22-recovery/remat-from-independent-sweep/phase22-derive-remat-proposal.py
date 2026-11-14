import hashlib,json,shutil,sys
from pathlib import Path
root=Path.cwd();v=root/'hierarchos-vulkan/validation';sys.path.insert(0,str(v))
import bench_rwkv_tbptt_packed_chains as driver
import check_measured_producer_routes as guard
source=v/'release_parity/multimodal_training/20261112-unified-release/phase19z/packed-cell';status=json.loads((source/'status.json').read_text())
out=root/'target/phase22-remat-proposal';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
table=v/'tbptt_packed_rematerialization_decisions.json';candidate=out/table.name;shutil.copyfile(table,candidate);shutil.copyfile(table,out/'before.json')
expected=status['source_files']['hierarchos-vulkan/validation/bench_rwkv_tbptt_packed_chains.py'];assert sha(v/'bench_rwkv_tbptt_packed_chains.py')==expected
assert status['decision_record_sha256'][table.resolve().as_posix()]==sha(table)
meta=[]
for name,storage,file in [('tbptt-packed-rematerialization','fp32','tbptt_packed_remat_bench.json'),('tbptt-packed-rematerialization-fp16','fp16-storage','tbptt_packed_remat_fp16_bench.json')]:
 s=guard.ALL_SPECS_BY_NAME[name];q=json.loads((source/file).read_text());runs=q['batches'];spec=driver.CHAINS['remat']
 assert status['results'][name]['bench']==0 and status['results'][name]['guard']!=0
 assert q['samples']==1200 and q['warmup']==2 and q['steps']==8 and guard.receipt_policy(q)==storage
 guard.check_sweep(runs,s.sweep)
 for r in runs:
  guard.check_shipped_route_timings(s,r)
  assert r['worst_max_abs']==0.0 and {t['label'] for t in r['tensors']}==driver.TENSORS
  assert all(t['max_abs']==0.0 and t['worst_excess']<=0 for t in r['tensors'])
 shutil.copyfile(source/file,out/file)
 driver.ab.record_decisions(candidate,runs,kind=spec['kind'],probe_key='tbptt',notes=driver.RECORD_NOTES['remat'],measured_by=driver.MEASURED_BY,margin=.02,probe_label=spec['probe'],parameter_storage=None if storage=='fp32' else storage,candidates=lambda r:{arm:r[arm+'_ms'] for arm in spec['arms']},withhold_when=driver.withheld_backward_reason)
 driver.ab.refuse_shared_verdicts(candidate,device=runs[0]['device'],width=runs[0]['width'],default_route=spec['structural_default'],refusals=spec['policy_refusals'])
 meta.append(dict(producer=name,source_receipt_sha256=sha(source/file),storage=storage,batches=[r['batch'] for r in runs],selected_source='shipped-route-v1'))
(out/'provenance.json').write_bytes((json.dumps(dict(kind='calibration-proposal-only',qualification_status='NOT RUN',measured_source_revision=status['source_revision'],driver_sha256=expected,helper_sha256=sha(v/'ab_harness.py'),before_sha256=sha(table),candidate_sha256=sha(candidate),measurements=meta),indent=2)+'\n').encode())
(out/'README.md').write_bytes(b'Calibration proposal only. Both complete five-batch rematerialization measurements passed exact numerical checks but failed the current performance verdict at batch 1. The unchanged rows are inputs to the maintained record_decisions/refuse_shared_verdicts functions. No timing or tolerance is edited. The active runtime table remains unchanged. After the full failed sweep is archived, applying/rebuilding this candidate requires a new independent thirteen-pair replay.\n')
assert sha(table)==status['decision_record_sha256'][table.resolve().as_posix()]
print('Prepared candidate from both actual full five-batch measurements; runtime table untouched')
for e in json.loads(candidate.read_text())['decisions']:
 if 'pci=1002:15bf' in e['device'] and e['batch']==1:print(e)
