"""Seal one current-source release after every independent component verifies.

The input map names actual generated component JSON files, binary paths and raw
logs. Their bytes are copied without rebinding historical source hashes. The
maintained Phase XIX checkers run before the new release receipt is published.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
from check_unified_release import ROOT, REQUIRED, DEFAULT, digest, require, validate_payload, verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,default=DEFAULT)
    a=p.parse_args();inputs=json.loads(a.inputs.read_text());out=a.output.resolve();base=out.parent
    require(not out.exists(),'refusing to overwrite a release receipt')
    require(set(inputs['components'])==REQUIRED,'input component inventory incomplete')
    require(not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no'],text=True).strip(),
            'commit a clean candidate and component evidence before release sealing')
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    for name in ('phase19x','phase19y','phase19z'):
        script={'phase19x':'check_phase19x_current.py','phase19y':'check_phase19y_full_attention.py','phase19z':'check_phase19z_current.py'}[name]
        subprocess.run([sys.executable,str(ROOT/'hierarchos-vulkan/validation'/script),'--receipt',
                        str(Path(inputs['components'][name]).resolve())],cwd=ROOT,check=True)
    from phase19w_matrix import validate
    validate(json.loads(Path(inputs['components']['phase19w']).read_text()))
    destination=base/'evidence/unified'
    require(not destination.exists(),'refusing to overwrite unified evidence')
    destination.mkdir(parents=True)
    components={};bindings={};evidence={}
    for name,value in inputs['components'].items():
        source=Path(value).resolve();target=destination/(name+'.json');target.write_bytes(source.read_bytes())
        relative=target.relative_to(base).as_posix();sha=digest(target)
        bindings[name]=dict(path=relative,sha256=sha,original_path=os.path.relpath(source,base).replace('\\','/'))
        components[name]=json.loads(target.read_text());evidence[relative]=sha
    for name,value in inputs.get('raw_evidence',{}).items():
        require('..' not in Path(name).parts and not Path(name).is_absolute(),'unsafe raw evidence path')
        target=destination/'raw'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(value).read_bytes())
        evidence[target.relative_to(base).as_posix()]=digest(target)
    paths=subprocess.check_output(['git','-C',str(ROOT),'ls-files',
        'hierarchos-vulkan/src','hierarchos-vulkan/shaders','hierarchos-inference/src',
        'hierarchos-native-cli/src','hierarchos-gui/src',
        ':(glob)hierarchos-vulkan/validation/*.py',':(glob)hierarchos-vulkan/validation/*.json',
        ':(glob)hierarchos-*/Cargo.toml',':(glob)hierarchos-*/Cargo.lock','.github/workflows'],text=True).splitlines()
    sources={name:digest(ROOT/name) for name in paths}
    binaries={name:dict(path=Path(value).resolve().relative_to(ROOT).as_posix(),sha256=digest(value))
              for name,value in inputs['binaries'].items()}
    policy=dict(default='off',mixed_operands='fp16',mixed_accumulation='fp32',
        full_fp16_max_reduction=64,full_fp16_is_universal_graph_precision=False,
        full_fp16_admission='qualified local Llama64/shared CLIP64 profile and AMD 1002:15bf driver 00800184',
        master_weights='fp32',optimizer_state='fp32',long_full_reductions='explicit FP32 fallback')
    capabilities=[]
    for family in ('transformer_full','transformer_lora','shared_clip_projector','bart_image','t5_image','switch_image',
                   'llava','gemma3','gemma4','deepseek_v41'):
        full=family in ('transformer_full','transformer_lora','shared_clip_projector')
        scaling=family in ('transformer_full','transformer_lora','shared_clip_projector','llava')
        capabilities.append(dict(family=family,
            fp32=dict(status='fully_qualified',scope='original maintained frozen observable-value inventory; native V4.1 bounded fixture',
                      evidence=['text166',*IMAGE_COUNTS_FOR(family)]),
            mixed=dict(status='supported_with_explicit_fp32_suboperations',
                scope='executed bounded full/LoRA, wide multimodal or native-vision fixture; not arbitrary model/range qualification',
                loss_scaling='GPU global scaling/unscaling/skip' if scaling else 'tested unscaled FP32 storage; FP16 GEMM operand quantization remains; no automatic-scaling claim',
                evidence=['native_v41'] if family=='deepseek_v41' else ['precision'] if full else ['wide']),
            full_fp16=dict(status='fully_qualified' if full else 'unsupported',
                scope='qualified Llama64/shared CLIP64 package, adapter/driver and reductions <=64; FP32 suboperations retained' if full else 'not admitted',
                evidence=['precision','tests','gui_cli'] if full else [])))
    document=dict(schema=1,status='PASS_SCOPED_RELEASE',source_revision=revision,source_clean=True,
        device=inputs['device'],components=bindings,source_files=sources,binaries=binaries,evidence_files=evidence,
        precision_policy=policy,precision_capabilities=capabilities,
        limitations=[
            'The unchanged 2e-7 intermediate diagnostic failures remain disclosed; original strict 93-image and 166-text observable-value gates are separate.',
            'Additional 64/128-wide FP32 and native V4.1 stress diagnostics are not promoted to universal strict CPU equivalence.',
            'Full FP16 is scoped matrix accumulation with FP32 masters/state and explicit FP32 fallback, not unrestricted FP16 graph training.',
            'Automatic scaling is claimed only for executed GPT2/Llama/ordinary-LoRA and shared CLIP/LLaVA joint transactions.',
            'Full-model optimizer resume, image accumulation, dropout accumulation and partial-window resume remain unsupported.',
            'Local thirteen-route qualification does not assert a queued or unavailable self-hosted GitHub workflow passed.',
            'GitHub-hosted CI conclusions must be checked at the published release commit; sealing this receipt does not predict them.'])
    validate_payload(document,components)
    require(all(digest(ROOT/name)==sha for name,sha in sources.items()),'source changed while sealing')
    matrix=base/'precision-capabilities.json';require(not matrix.exists(),'capability matrix already exists')
    matrix.write_text(json.dumps(dict(schema=1,source_revision=revision,device=inputs['device'],policy=policy,families=capabilities),indent=2)+'\n')
    document['evidence_files'][matrix.relative_to(base).as_posix()]=digest(matrix)
    with out.open('x') as f:json.dump(document,f,indent=2,allow_nan=False)
    verify(out)
    print('SEALED UNIFIED RELEASE:',revision,flush=True)


def IMAGE_COUNTS_FOR(family):
    return {'shared_clip_projector':['shared77'],'bart_image':['seq2seq'],'t5_image':['seq2seq'],'switch_image':['seq2seq'],
        'llava':['llava'],'gemma3':['gemma3'],'gemma4':['gemma4-legacy','gemma4-rgb'],'deepseek_v41':['native_v41']}.get(family,[])


if __name__=='__main__':main()
