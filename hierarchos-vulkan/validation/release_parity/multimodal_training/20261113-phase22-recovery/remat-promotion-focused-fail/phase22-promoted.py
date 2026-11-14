import hashlib,json,os,subprocess,time,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];v=root/'hierarchos-vulkan/validation';bins=root/'hierarchos-vulkan/target/release'
py=root/'.phase19w-venv/Scripts/python.exe'
env={**os.environ,'CARGO_TARGET_DIR':str(root/'hierarchos-vulkan/target'),'CARGO_BUILD_JOBS':'2','HIERARCHOS_WMMA_MODE':'off','PYTHONPATH':'C:/Users/User/transformers/src','PYTHONUNBUFFERED':'1'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
revision=lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip()
rows=[]
def run(label,command,extra=None):
 print('Starting',label,flush=True);t=time.time();log=root/'target'/('phase22-promoted-'+label+'.log')
 assert not log.exists(),str(log)
 with log.open('w',encoding='utf-8') as f:r=subprocess.run(list(map(str,command)),cwd=root,env={**env,**(extra or {})},stdout=f,stderr=subprocess.STDOUT)
 rows.append(dict(label=label,command=list(map(str,command)),exit_code=r.returncode,seconds=time.time()-t,source_revision=revision()))
 (root/'target/phase22-promoted-status.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
 print('Finished',label,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
 return log
run('chain-build',['cargo','build','--locked','--release','--manifest-path','hierarchos-vulkan/Cargo.toml','--bin','hierarchos-vulkan-rwkv-tbptt-packed-chains-bench'])
run('remat-focused',[py,root/'target/phase22-remat-focused.py'])
run('classifier-tests',['cargo','test','--locked','--release','--manifest-path','hierarchos-vulkan/Cargo.toml','--bin','hierarchos-vulkan-rwkv-tbptt-packed-chains-bench','--','--test-threads=1'])
run('packed-current',[py,v/'qualify_packed_current.py','--output',v/'release_parity/multimodal_training/20261112-unified-release/phase19z/packed-cell'])
run('libtest-build',['cargo','test','--locked','--release','--manifest-path','hierarchos-vulkan/Cargo.toml','--lib','--no-run'])
audit=json.loads((root/'target/phase22-long-before-audit.json').read_text());post={}
for name,row in audit['binaries'].items():
 actual=sha(root/row['path']);changed=actual!=row['sha256']
 if not row['embedded_decision_tables'] and changed:raise RuntimeError('Unexpected change to unaffected binary '+name)
 post[name]=dict(**row,current_sha256=actual,changed=changed)
(root/'target/phase22-promoted-binary-audit.json').write_text(json.dumps(post,indent=2)+'\n',encoding='utf-8')
libtest=bins/'deps/hierarchos_vulkan-8c8d2140efff5506.exe'
run('library',[libtest,'--test-threads=1'])
run('full-fallback',[libtest,'oversized_full_fp16_executes_fp32_forward_backward_adamw_and_checkpoint','--nocapture','--test-threads=1'])
run('attention',[libtest,'transformer::tests::two_pass_attention_backward_matches_scalar_gradients','--exact','--nocapture','--test-threads=1'],{'HIERARCHOS_ATTENTION_BACKWARD_TWO_PASS':'1'})
oracle=root/'target/unified-oracle/Scripts/python.exe'
qualified={**env,'HIERARCHOS_ATEN_VECTOR_WIDTH':'8','PYTHONPATH':str(root/'target/transformers-oracle-469/src'),'HIERARCHOS_TRANSFORMERS_CHECKOUT':str(root/'target/transformers-oracle-469'),'ATEN_CPU_CAPABILITY':'avx2','MKL_ENABLE_INSTRUCTIONS':'AVX2','ONEDNN_MAX_CPU_ISA':'AVX2'}
frozen=root.parent/'Hierarchos-Native/hierarchos-vulkan/target/rdna3-port/gen9-frozen'
run('fp32-replay',[oracle,v/'qualify_rdna3_full_multimodal.py','--frozen',frozen,'--output',root/'target/phase22-promoted-fp32-replay','--bin-dir',bins,'--test-binary',libtest],qualified)
run('v41',[oracle,v/'qualify_v41_wmma.py','--fixture',root/'target/unified-v41-bounded-fixture.json','--stress-fixture',root/'target/unified-v41-wmma-fixture.json','--binary',bins/'deepseek_v41_vision_precision_probe.exe','--libtest',libtest,'--output',root/'target/phase22-promoted-v41'],qualified)
run('intermediates',[oracle,v/'audit_fp32_intermediates.py','--replay',root/'target/phase22-promoted-fp32-replay','--frozen',frozen,'--output',root/'target/phase22-promoted-fp32-intermediates.json'],qualified)
