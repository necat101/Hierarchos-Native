"""Execute all thirteen maintained packed-cell benchmark/guard pairs serially.

Preserves the spec's complete batch sweeps and timing configuration. Fresh
receipts are written in a new directory; existing decision records are read
only. A failing numerical or route decision remains a failure, never a PASS
based on the benchmark process alone. No self-hosted CI result is inferred.
"""
import argparse, copy, hashlib, json, os, statistics, subprocess, sys, tempfile
from pathlib import Path
from check_measured_producer_routes import ALL_SPECS

ROOT=Path(__file__).resolve().parents[2]
PACKED_RELEASE_BINARIES=(
    'hierarchos-vulkan-rwkv-key-state-forward-bench',
    'hierarchos-vulkan-rwkv-low-rank-fp16-forward-bench',
    'hierarchos-vulkan-rwkv-packed-cell-channel-mix-bench',
    'hierarchos-vulkan-rwkv-packed-cell-low-rank-forward-bench',
    'hierarchos-vulkan-rwkv-packed-cell-time-mix-bench',
    'hierarchos-vulkan-rwkv-tbptt-packed-chains-bench',
    'hierarchos-vulkan-rwkv-tbptt-time-mix-bench',
    'rwkv_tbptt_packed_forward_parity',
    'hierarchos-vulkan-rwkv-packed-cell-step',
)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def merge_batch_receipts(receipts):
    """One sweep receipt from one fresh-process measurement per batch.

    Measuring every batch in a long-lived process lets the APU's clock/thermal
    state at batch N depend on every earlier batch.  That changed route ratios
    beyond the unchanged noise band between recording and replay even at 1200
    samples.  Each batch is therefore measured in its own process, and this
    merges the per-batch receipts without changing any row or ratio.
    """
    merged = copy.deepcopy(receipts[0])
    batches = sorted(
        (row for receipt in receipts for row in receipt["batches"]),
        key=lambda row: row["batch"],
    )
    merged["batches"] = batches
    ratios = [
        row["composed_over_fused"]
        for row in batches
        if isinstance(row.get("composed_over_fused"), (int, float))
    ]
    if ratios:
        merged["median_composed_over_fused"] = statistics.median(ratios)
    withheld = {}
    for receipt in receipts:
        for entry in receipt.get("withheld_routes", []):
            withheld[(entry["batch"], entry["route"])] = entry
    if "withheld_routes" in merged or withheld:
        merged["withheld_routes"] = [withheld[key] for key in sorted(withheld)]
    unmeasured = sorted({batch for receipt in receipts for batch in receipt.get("unmeasured", [])})
    if "unmeasured" in merged or unmeasured:
        merged["unmeasured"] = unmeasured
    merged["batch_process_isolation"] = True
    return merged


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
    # Cargo may relink a consumer after a library metadata change even when it
    # does not embed the changed table. Build the full packed set before taking
    # the immutable binary snapshot, rather than rebuilding it during timing.
    build_command=['cargo','build','--locked','--release','--manifest-path',
        str(ROOT/'hierarchos-vulkan/Cargo.toml'),
        *[word for name in PACKED_RELEASE_BINARIES for word in ('--bin',name)]]
    with (out/'release-build.log').open('w',encoding='utf-8') as log:
        subprocess.run(build_command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    cli=ROOT/'hierarchos-vulkan/target/release/hierarchos-native-cli.exe'
    capability_run=subprocess.run([str(cli),'precision-info'],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
    capability=json.loads(capability_run.stdout)
    if capability.get('vendor_id')!=0x1002 or capability.get('device_id')!=0x15bf:
        raise RuntimeError('current packed qualification requires the recorded physical RDNA3 adapter')
    (out/'device-capabilities.json').write_text(json.dumps(capability,indent=2)+'\n')
    binaries={p.name:digest(p) for p in (ROOT/'hierarchos-vulkan/target/release').glob('*.exe')}
    status=dict(schema=1,kind='phase19z-packed-cell-amd-local',status='NOT RUN',source_revision=revision,
        release_build_command=build_command,release_build_exit_code=0,
        device=capability['device_key'],adapter='amd-rdna3-lds64',binary_sha256=binaries,
        ci=dict(status='NOT RUN',note='Local physical GPU replay; self-hosted workflow is separate'),
        source_files=sources,decision_record_sha256=records,passes=[],failures=[],results={})
    for spec in ALL_SPECS:
        receipt=out/spec.receipt.name
        command=[sys.executable,str(ROOT/spec.driver)]
        command+=list(spec.bench_args)
        print('Starting packed route',spec.name,flush=True)
        with (out/(spec.name+'-bench.log')).open('w',encoding='utf-8') as log:
            if spec.step_kind=='composition':
                bench=subprocess.run(command+['--record',str(receipt)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                commands=[command+['--record',str(receipt)]]
            else:
                receipts=[]
                commands=[]
                bench=None
                with tempfile.TemporaryDirectory(prefix='packed-batch-'+spec.name+'-') as scratch:
                    for batch in spec.sweep:
                        batch_receipt=Path(scratch)/f'batch-{batch}.json'
                        batch_command=command+['--batch',str(batch),'--record',str(batch_receipt)]
                        commands.append(batch_command)
                        log.write('$ '+' '.join(map(str,batch_command))+'\n');log.flush()
                        bench=subprocess.run(batch_command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                        log.flush()
                        if bench.returncode!=0 or not batch_receipt.exists():break
                        receipts.append(json.loads(batch_receipt.read_text(encoding='utf-8')))
                    if bench is not None and bench.returncode==0 and len(receipts)==len(spec.sweep):
                        receipt.write_text(json.dumps(merge_batch_receipts(receipts),indent=2)+'\n',encoding='utf-8')
        guard_command=[sys.executable,str(ROOT/'hierarchos-vulkan/validation/check_measured_producer_routes.py'),
                       '--producer',spec.name,'--receipt',str(receipt)]
        with (out/(spec.name+'-guard.log')).open('w',encoding='utf-8') as log:
            guard=subprocess.run(guard_command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        result=dict(bench=bench.returncode,guard=guard.returncode,commands=commands,guard_command=guard_command,
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
