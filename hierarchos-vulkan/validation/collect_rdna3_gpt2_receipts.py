"""Collect completed physical runs; bind receipts, source and frozen oracles."""
import argparse,hashlib,json,re,shutil,subprocess,zipfile
from pathlib import Path
from check_rdna3_gpt2_receipts import COUNTS,RECEIPTS,ROOT,validate_qualification

CRATE=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_bytes((json.dumps(value,indent=2)+'\n').encode())
def read(path):return json.loads(path.read_text())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,default=Path('target/rdna3-port'))
    p.add_argument('--prefix',default='gpt2-selected')
    p.add_argument('--cpu',type=Path,default=Path('target/rdna3-port/gpt2-shapes-final-cpu'))
    p.add_argument('--output',type=Path,default=RECEIPTS)
    a=p.parse_args();r=a.run_root;prefix=a.prefix
    for name,count in COUNTS.items():validate_qualification(read(r/(prefix+'-validation')/name/'qualification.json'),count)
    for name in ['native','bridge']:
        if not read(r/(prefix+'-v41-'+name)/'qualification.json')['pass']:raise ValueError('V4.1 failed')
    if not read(r/'gpt2-provenance-selected.json')['pass']:raise ValueError('shader provenance failed')
    cpu=read(a.cpu/'manifest.json')
    if not cpu['complete'] or cpu['oracle']['text_attention']!='sdpa':raise ValueError('incomplete CPU oracle')
    benchmark=read(r/(prefix+'-benchmark')/'benchmark.json')
    if benchmark['source_sha256']!=digest(CRATE/'shaders/image_text_gpt2_avx512_amd_rdna3.comp'):
        raise ValueError('benchmark source changed')
    if benchmark['candidate_sha256']!=digest(CRATE/'shaders/image_text_gpt2_packed8_avx512_amd_rdna3.spv'):
        raise ValueError('benchmark candidate differs from production artifact')
    a.output.mkdir(parents=True,exist_ok=False)
    for name in COUNTS:
        shutil.copyfile(r/(prefix+'-validation')/name/'qualification.json',a.output/(name+'-qualification.json'))
    for name in ['native','bridge']:
        shutil.copyfile(r/(prefix+'-v41-'+name)/'qualification.json',a.output/('v41-'+name+'-qualification.json'))
    mapping={'cpu-manifest.json':a.cpu/'manifest.json',
        'primitives.log':r/(prefix+'-final-primitives.log'),
        'shader-provenance.json':r/'gpt2-provenance-selected.json',
        'spirv-validation.json':r/'gpt2-spirv-selected.json',
        'isolation.json':r/'gpt2-isolation-final.json',
        'benchmark.json':r/(prefix+'-benchmark')/'benchmark.json'}
    for name,source in mapping.items():shutil.copyfile(source,a.output/name)
    sweep=[]
    for lanes in [2,4,8,16,32]:
        sweep.append(read(r/f'gpt2-benchmark-lanes-{lanes}'/'benchmark.json'))
    sweep.append(read(r/'gpt2-final-benchmark/benchmark.json'))
    write(a.output/'forward-workgroup-sweep.json',sweep)
    archive=r/'gpt2-shapes-avx512-oracles.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as file:
        for path in sorted(a.cpu.rglob('*')):
            if path.is_file():file.write(path,path.relative_to(a.cpu).as_posix())
    shader_manifest=read(CRATE/'validation/training_shader_manifest.json')
    recipes={e['artifact']:e for e in shader_manifest['shaders']}
    sources=set(CRATE/name for name in ['Cargo.toml','Cargo.lock','src/cpu_adamw.rs','src/vendor.rs','src/transformer.rs',
        'src/transformer/image_text.rs','src/transformer/vision.rs','src/transformer/rdna3_text_attention_tests.rs',
        'src/transformer/rdna3_gpt2_bench.rs','validation/training_shader_manifest.json'])
    def include_closure(path):
        if path in sources:return
        sources.add(path)
        for name in re.findall(r'#include\s+"([^"]+)"',path.read_text()):include_closure(path.parent/name)
    for row in read(a.output/'shader-provenance.json')['shaders']:
        artifact=row['artifact'];sources.add(CRATE/'shaders'/artifact)
        include_closure(CRATE/'shaders'/recipes[artifact]['source'])
    sources.add(CRATE/'shaders/rdna3_zen4_rsqrt_seed.bin')
    sources.update((CRATE/'validation').glob('*rdna3*gpt2*.py'))
    sources.update((CRATE/'validation').glob('*rdna3*rsqrt*.rs'))
    sources.update((CRATE/'validation').glob('*rdna3*tanh*.py'))
    sources.update((CRATE/'validation').glob('*rdna3*log*.py'))
    sources.update((CRATE/'validation').glob('freeze_rdna3_*.py'))
    sources.update((CRATE/'validation/fixtures').glob('rdna3_*.json'))
    sources.add(CRATE/'validation/diagnose_shared_text.py')
    sources.add(CRATE/'validation/diagnose_gpt2_wide_stages.py')
    sources.update(CRATE/'validation'/name for name in ['qualify_multimodal_training.py',
        'qualify_shared_image_pipeline.py','qualify_seq2seq_image_pipeline.py','qualify_gemma3_pipeline.py',
        'qualify_gemma4_pipeline.py','qualify_llava_pipeline.py','qualify_deepseek_v41_native_pipeline.py',
        'qualify_deepseek_v41_shared_image.py'])
    summary=dict(schema=1,**{'pass':True},target='rdna3',absolute_limit=2e-7,rtol=0,comparison='strict_less_than',
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),worktree=True,
        expanded_cases=16,retained_shared_cases=85,native_gemma_llava_cases=10,native_v41_tests=30,
        profile_count=44,primitive_tests=9,shader_artifacts=43,
        cpu_contract='Windows Torch 2.6.0+cpu, SDPA, Z1 Extreme CPUID 00a70f41',
        adapter=dict(vendor='1002',device='15bf',driver='00800184'),
        oracle_archive=dict(path=str(archive.resolve()),sha256=digest(archive),bytes=archive.stat().st_size),
        limits=['eight measured GPT2 geometries; not a Cartesian shape certificate',
            'isolated kernel timings; production-size training throughput unqualified',
            'native V4.1 retains its frozen-language LoRA contract','no physical Gen9 execution'])
    write(a.output/'summary.json',summary)
    write(a.output/'receipt-hashes.json',dict(schema=1,
        receipts={p.name:digest(p) for p in sorted(a.output.iterdir()) if p.is_file()},
        sources={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(sources)}))
    print('Collected',a.output,'oracle archive',archive,archive.stat().st_size,'bytes')
if __name__=='__main__':main()
