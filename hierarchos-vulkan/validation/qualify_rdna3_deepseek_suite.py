"""Run every green DeepSeek test with bounded driver-cache lifetime on the APU.

All 218 active tests execute exactly once. The original 26 opt-in tests are
enumerated separately and must match the released Gen9 ignore list exactly.
"""
import argparse,hashlib,json,os,re,subprocess
from pathlib import Path

CRATE=Path(__file__).resolve().parents[1]
BASELINE=CRATE/'validation/release_parity/multimodal_training/20261105-gen9-v41-upload-batching/deepseek-suite-qualification.json'
# These tests live in the DeepSeek test module because they exercise its
# package allocator, but they are CI infrastructure rather than numerical cases.
# Keep them out of the 218 active/26 ignored numerical inventory.
NON_NUMERICAL_TESTS={
    'deepseek_v41::model::tests::temporary_test_roots_do_not_reuse_stale_directories',
    'deepseek_v41::model::tests::temporary_test_roots_are_unique_under_concurrent_reservation',
    'deepseek_v41::model::tests::temporary_test_roots_support_repeated_sequential_reservation',
    'deepseek_v41::model::tests::training::native_vision::native_state::native_checkpoint_cleanup_is_scoped_to_test_owned_paths',
}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_bytes((json.dumps(v,indent=2)+'\n').encode())
def require(c,m):
    if not c:raise ValueError(m)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shard-size',type=int,default=8);p.add_argument('--host-width',type=int,choices=[8,16],default=16)
    a=p.parse_args();a.binary=a.binary.resolve(strict=True);a.output=a.output.resolve()
    require(1<=a.shard_size<=16,'use bounded shards of 1..16 tests')
    a.output.mkdir(parents=True,exist_ok=False)
    (CRATE.parent/'target/deepseek-v41-reference').mkdir(parents=True,exist_ok=True)
    env={**os.environ,'HIERARCHOS_ATEN_VECTOR_WIDTH':str(a.host_width)}
    for key in ['HIERARCHOS_VULKAN_PROFILE_KERNELS','HIERARCHOS_VULKAN_SUBMISSION_REPORT']:env.pop(key,None)
    def run(args):return subprocess.run([str(a.binary),*args],cwd=CRATE,env=env,
        capture_output=True,text=True,encoding='utf-8',errors='replace')
    full=run(['deepseek_v41','--list']);ignored=run(['deepseek_v41','--ignored','--list'])
    (a.output/'all-tests.log').write_text(full.stdout+full.stderr)
    (a.output/'ignored-tests.log').write_text(ignored.stdout+ignored.stderr)
    names=lambda text:set(re.findall(r'^(\S+): test$',text,re.M))
    all_names=names(full.stdout)-NON_NUMERICAL_TESTS;ignored_names=names(ignored.stdout)
    require(full.returncode==ignored.returncode==0 and len(all_names)==244,'full suite enumeration changed')
    require(ignored_names==set(json.loads(BASELINE.read_text())['ignored']),'opt-in test list changed')
    active=sorted(all_names-ignored_names);require(len(active)==218,'active test list changed')
    sha=digest(a.binary);rows=[]
    def persist():
        completed=sum(len(r['tests']) for r in rows if r['pass'])
        write(a.output/'qualification.json',dict(schema=1,target='rdna3',host_vector_width=a.host_width,
            **{'pass':completed==218 and all(r['pass'] for r in rows)},complete=sum(len(r['tests']) for r in rows)==218,
            expected_tests=244,expected_passed=218,expected_ignored=26,verified_passed=completed,
            ignored=sorted(ignored_names),active=active,shards=rows,binary_sha256=sha,
            runner_sha256=digest(Path(__file__)),baseline_sha256=digest(BASELINE),
            process_policy='fresh process per bounded shard; serial GPU execution; no test omitted'))
    persist()
    for index,start in enumerate(range(0,len(active),a.shard_size)):
        chunk=active[start:start+a.shard_size]
        result=run([*chunk,'--exact','--test-threads=1','--show-output'])
        log=result.stdout+result.stderr;(a.output/f'shard-{index:02}.log').write_text(log,encoding='utf-8')
        expected=f'test result: ok. {len(chunk)} passed; 0 failed; 0 ignored;'
        completed=set(re.findall(r'^test (\S+) \.\.\. ok$',log,re.M))
        passed=result.returncode==0 and expected in log and completed==set(chunk)
        require(digest(a.binary)==sha,'test binary changed during replay')
        rows.append(dict(index=index,tests=chunk,returncode=result.returncode,**{'pass':passed}))
        persist();print('shard',index,'PASS' if passed else 'FAIL',f'{len(chunk)} tests',flush=True)
    require(all(r['pass'] for r in rows),'DeepSeek test failures retained')

if __name__=='__main__':main()
