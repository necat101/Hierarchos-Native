"""Execute all thirteen maintained packed-cell benchmark/guard pairs serially.

Preserves the spec's complete batch sweeps and timing configuration. Fresh
receipts are written in a new directory; existing decision records are read
only. A failing numerical or route decision remains a failure, never a PASS
based on the benchmark process alone. No self-hosted CI result is inferred.
"""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
from check_measured_producer_routes import ALL_SPECS

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise RuntimeError('commit a clean candidate before packed-cell measurements')
    paths=subprocess.check_output(['git','-C',str(ROOT),'ls-files','hierarchos-vulkan/src','hierarchos-vulkan/shaders',
        'hierarchos-vulkan/validation/check_measured_producer_routes.py'],text=True).splitlines()
    paths+=sorted({spec.driver for spec in ALL_SPECS})+[Path(__file__).relative_to(ROOT).as_posix()]
    paths+=['hierarchos-vulkan/Cargo.toml','hierarchos-vulkan/Cargo.lock','hierarchos-inference/Cargo.toml','hierarchos-inference/Cargo.lock']
    sources={name:digest(ROOT/name) for name in paths}
    # The benchmark drivers must not update their route tables during replay.
    records={p.as_posix():digest(p) for p in (ROOT/'hierarchos-vulkan/validation').rglob('*decisions*.json')}
    env={**os.environ,'CARGO_TARGET_DIR':str(ROOT/'hierarchos-vulkan/target'),'CARGO_BUILD_JOBS':'2','HIERARCHOS_WMMA_MODE':'off'}
    cli=ROOT/'hierarchos-vulkan/target/release/hierarchos-native-cli.exe'
    capability_run=subprocess.run([str(cli),'precision-info'],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
    capability=json.loads(capability_run.stdout)
    if capability.get('vendor_id')!=0x1002 or capability.get('device_id')!=0x15bf:
        raise RuntimeError('current packed qualification requires the recorded physical RDNA3 adapter')
    (out/'device-capabilities.json').write_text(json.dumps(capability,indent=2)+'\n')
    binaries={p.name:digest(p) for p in (ROOT/'hierarchos-vulkan/target/release').glob('*.exe')}
    status=dict(schema=1,kind='phase19z-packed-cell-amd-local',status='NOT RUN',source_revision=revision,
        device=capability['device_key'],adapter='amd-rdna3-lds64',binary_sha256=binaries,
        ci=dict(status='NOT RUN',note='Local physical GPU replay; self-hosted workflow is separate'),
        source_files=sources,decision_record_sha256=records,passes=[],failures=[],results={})
    for spec in ALL_SPECS:
        receipt=out/spec.receipt.name
        command=[sys.executable,str(ROOT/spec.driver)]
        if spec.step_kind!='composition':command += [v for batch in spec.sweep for v in ('--batch',str(batch))]
        command+=list(spec.bench_args)+['--record',str(receipt)]
        print('Starting packed route',spec.name,flush=True)
        with (out/(spec.name+'-bench.log')).open('w',encoding='utf-8') as log:
            bench=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        guard_command=[sys.executable,str(ROOT/'hierarchos-vulkan/validation/check_measured_producer_routes.py'),
                       '--producer',spec.name,'--receipt',str(receipt)]
        with (out/(spec.name+'-guard.log')).open('w',encoding='utf-8') as log:
            guard=subprocess.run(guard_command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        result=dict(bench=bench.returncode,guard=guard.returncode,command=command,guard_command=guard_command,
                    receipt_sha256=digest(receipt) if receipt.exists() else None)
        status['results'][spec.name]=result
        status['passes' if bench.returncode==guard.returncode==0 else 'failures'].append(spec.name)
        status['status']='FAIL' if status['failures'] else 'IN_PROGRESS'
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        print(spec.name,'PASS' if bench.returncode==guard.returncode==0 else 'FAIL',flush=True)
    try:
        if any(digest(ROOT/name)!=expected for name,expected in sources.items()):raise RuntimeError('bound source changed')
        if any(digest(path)!=expected for path,expected in records.items()):raise RuntimeError('historical decision record was modified')
        if any(digest(ROOT/'hierarchos-vulkan/target/release'/name)!=expected for name,expected in binaries.items()):
            raise RuntimeError('release executable changed during packed measurements')
        composition=ROOT/'hierarchos-vulkan/target/debug/hierarchos-vulkan-rwkv-packed-cell-step.exe'
        if composition.exists():status['additional_binary_sha256']={composition.relative_to(ROOT).as_posix():digest(composition)}
    except RuntimeError as error:
        status['status']='FAIL';status['fatal_error']=str(error)
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        raise
    status['status']='FAIL' if status['failures'] else 'PASS'
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    if status['failures']:raise SystemExit('Packed routes failed: '+', '.join(status['failures']))
    print('CURRENT PACKED CELL: PASS 13/13',flush=True)


if __name__=='__main__':main()
