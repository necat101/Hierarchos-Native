import os,sys,json,subprocess,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];v=root/'hierarchos-vulkan/validation';sys.path.insert(0,str(v))
from qualify_packed_current import merge_batch_receipts,digest
from check_measured_producer_routes import ALL_SPECS
by_name={s.name:s for s in ALL_SPECS}
out=root/'target/phase22-promoted-remat-focused';out.mkdir(exist_ok=False)
binary=root/'hierarchos-vulkan/target/release/hierarchos-vulkan-rwkv-tbptt-packed-chains-bench.exe';table=v/'tbptt_packed_rematerialization_decisions.json'
lock={str(p):digest(p) for p in (binary,table,v/'bench_rwkv_tbptt_packed_chains.py',v/'ab_harness.py')}
commands=[]
for name in ('tbptt-packed-rematerialization','tbptt-packed-rematerialization-fp16'):
 s=by_name[name];receipts=[]
 for batch in s.sweep:
  dest=out/(name+'-'+str(batch)+'.json');command=[sys.executable,str(root/s.driver),*s.bench_args,'--batch',str(batch),'--record',str(dest)]
  commands.append(command)
  with (out/(name+'-'+str(batch)+'.log')).open('w',encoding='utf-8') as f:r=subprocess.run(command,cwd=root,stdout=f,stderr=subprocess.STDOUT)
  print(name,batch,'exit',r.returncode,flush=True)
  if r.returncode:raise SystemExit(r.returncode)
  receipts.append(json.loads(dest.read_text()))
 dest=out/s.receipt.name;dest.write_text(json.dumps(merge_batch_receipts(receipts),indent=2)+'\n',encoding='utf-8')
 command=[sys.executable,str(v/'check_measured_producer_routes.py'),'--producer',name,'--receipt',str(dest)]
 commands.append(command)
 with (out/(name+'-guard.log')).open('w',encoding='utf-8') as f:r=subprocess.run(command,cwd=root,stdout=f,stderr=subprocess.STDOUT)
 print(name,'guard',r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
assert all(digest(p)==h for p,h in lock.items())
(out/'status.json').write_text(json.dumps(dict(status='PASS_FOCUSED_ONLY',qualification_status='NOT RUN',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),immutable_sha256=lock,commands=commands),indent=2)+'\n',encoding='utf-8')
