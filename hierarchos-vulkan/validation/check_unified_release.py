"""Verify current FP32, Mixed, scoped FP16 and Phase XIX evidence together.

No PASS field is sufficient: numeric rows, optimizer counts, accumulator routes,
coverage and component hashes are checked again. Existing Phase XIX checkers
remain mandatory. Additional strict intermediate failures stay explicit.
"""
import argparse, hashlib, json, math, re, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'hierarchos-vulkan/validation/release_parity/multimodal_training/20261112-unified-release/release-qualification.json'
IMAGE_COUNTS={'shared77':77,'seq2seq':6,'gemma3':4,'gemma4-legacy':1,'gemma4-rgb':1,'llava':4}
REQUIRED={'text166','fp32_replay','precision','wide','native_v41','wmma_mixed','wmma_full_original','wmma_full_dot2',
          'shader727','shader_wmma','tests','gui_cli','packed','intermediates','phase19w','phase19x','phase19y','phase19z',*IMAGE_COUNTS}


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def strict(value,limit):return isinstance(value,(int,float)) and math.isfinite(value) and 0<=value<limit


def mode_counts(precision):
    return precision.get('accumulation_dispatches',
        [precision.get('mixed_fp32_accumulation',0),precision.get('full_fp16_accumulation',0)])


def phases(precision):
    return precision.get('phases',[precision.get('forward',0),precision.get('weight_grad',0),precision.get('input_grad',0)])


def verify_mode(precision,mode,weight_grad=True):
    require(precision['requested']==mode,'wrong requested precision')
    counts=mode_counts(precision);p=phases(precision)
    if mode=='off':require(counts==[0,0],'FP32 dispatched cooperative low precision')
    else:
        idx=0 if mode=='mixed' else 1
        require(counts[idx]>0 and counts[1-idx]==0,'missing or wrong cooperative accumulator route')
        require(p[0]>0 and p[2]>0 and (p[1]>0)==weight_grad,'missing/wrong cooperative training phase')
    require(precision['master_weights']=='fp32' and precision['optimizer_state']=='fp32','master/optimizer precision changed')


def verify_primitive(q,mode,failed):
    require(q['mode']==mode and q['gates']==37 and len(q['results'])==37,'primitive coverage changed')
    failures=[]
    for row in q['results']:
        ok=(row['nonfinite']==0 and math.isfinite(row['max_abs']) and row['max_abs']<=row['max_abs_limit']
            and (row['rms_limit'] is None or row['rms']<=row['rms_limit']))
        require(row['passed']==ok,'primitive PASS disagrees with numeric observation')
        if not ok:failures.append(row)
    require(len(failures)==q['failed']==failed,'primitive failure accounting changed')
    return failures


def validate_payload(document,components):
    require(document.get('schema')==1 and document.get('status')=='PASS_SCOPED_RELEASE','invalid unified release status')
    require(set(components)==REQUIRED,'missing unified qualification components')
    require(document['precision_policy']['full_fp16_max_reduction']==64,'Full FP16 numerical range was widened')
    require(document['precision_policy']['full_fp16_is_universal_graph_precision'] is False,'Full FP16 graph semantics overstated')
    binaries=document['binaries']
    def bound(name,sha):require(name in binaries and binaries[name]['sha256']==sha,'executable binding disagrees with run: '+name)
    bound('hierarchos-vulkan-transformer-logits',components['text166']['binary_sha256'])
    for stage,stem in [('shared77','image_text_pipeline_probe'),('seq2seq','image_text_pipeline_probe'),
        ('llava','llava_pipeline_probe'),('gemma3','gemma3_vision_probe'),('gemma4-legacy','gemma4_vision_probe'),('gemma4-rgb','gemma4_vision_probe')]:
        bound(stem,components[stage]['binary_sha256'])
    for stem,sha in components['precision']['binary_sha256'].items():bound(stem,sha)
    for family,sha in components['wide']['binary_sha256'].items():
        bound({'seq2seq':'image_text_pipeline_probe','llava':'llava_pipeline_probe','gemma3':'gemma3_vision_probe','gemma4':'gemma4_vision_probe'}[family],sha)
    bound('deepseek_v41_vision_precision_probe',components['native_v41']['binary_sha256']['probe'])
    bound('libtest',components['native_v41']['binary_sha256']['libtest'])
    text=components['text166'];rows=text['records']
    require(text['pass'] and len(rows)==166 and len({(r['family'],r['case'],r['mode']) for r in rows})==166,'text inventory incomplete')
    require(text['absolute_limit']==2e-7 and text['rtol']==0 and text['comparison']=='strict_less_than','text policy changed')
    for r in rows:require(r['pass'] and strict(r['max_abs'],2e-7) and r['failing_values']==0,'strict text comparison failed')
    require(components['fp32_replay']['pass'] and components['fp32_replay']['complete'],'FP32 full replay is incomplete')
    for stage,count in IMAGE_COUNTS.items():
        q=components[stage]
        require(q['pass'] and q['rtol']==0 and q['comparison']=='strict_less_than','image policy/result changed')
        require(len(q['results'])==count,'original image inventory incomplete')
        for r in q['results']:
            require(r['pass'] and r['checks'] and all(x['pass'] for x in r['checks']),'required FP32 image gate failed')
            for c in r['checks']:
                if 'logits' in c['metric'] and 'max_abs' in c:require(strict(c['max_abs'],2e-7),'image logits exceed strict limit')
    precision=components['precision'];require(precision['status']=='PASS_SCOPED' and precision['steps']>=20,'precision convergence qualification incomplete')
    reports=precision['reports']
    for kind in ('full','lora'):
        for mode in ('off','mixed','full-fp16'):
            key=kind+'-'+mode;r=reports[key]
            require(r['optimizer_steps']==20 and r['attempted_optimizer_windows']==20,'incomplete optimizer trajectory')
            verify_mode(r['precision'],mode,kind=='full')
            loss=precision['losses'][key]
            require(len(loss)==20 and all(math.isfinite(x) for x in loss) and loss[-1]<loss[0],'convergence trajectory failed')
            if mode!='off':
                require(r['loss_scaler']['skipped_steps']==0 and r['loss_scaler']['pending_microbatches']==0,'unexpected normal overflow/partial window')
        for mode in ('mixed','full-fp16'):
            key='accumulation-'+kind+'-'+mode;r=reports[key]
            require(r['optimizer_steps']==20 and r['attempted_optimizer_windows']==40,'incomplete microbatch trajectory')
            verify_mode(r['precision'],mode,kind=='full')
            require(r['loss_scaler']['accumulation_steps']==2 and r['loss_scaler']['pending_microbatches']==0,'incorrect accumulation state')
    for key,row in precision['checks'].items():
        if 'exact-resume' in key:
            require(row and all(v['max_abs']==0 for v in row.values()),'exact resume diverged')
        if '/state' in key and key.startswith('vision-overflow-'):
            require(row['skipped_steps']==2 and row['scale']==2.**78,'joint overflow skip/backoff failed')
        metrics=[row] if isinstance(row,dict) and 'max_abs' in row else list(row.values()) if isinstance(row,dict) else []
        for value in metrics:
            if not isinstance(value,dict) or 'max_abs' not in value:continue
            maximum=value['max_abs'];require(isinstance(maximum,(int,float)) and math.isfinite(maximum) and maximum>=0,'nonfinite precision measurement')
            if value.get('role','').startswith('additional CPU diagnostic'):
                require(value['within_strict_limit']==(maximum<2e-7),'strict diagnostic failure was relabeled')
                continue
            exact=any(word in key for word in ('exact-resume','-unchanged','/reload-'))
            limit=.02 if 'mixed' in key else .05
            require(maximum==0 if exact else maximum<limit,'precision numerical check failed: '+key)
            if not exact:require(strict(value['rms'],.003 if 'mixed' in key else .01),'precision RMS check failed: '+key)
    wide=components['wide'];require(wide['status']=='PASS_SCOPED' and len(wide['results'])==9,'wide matrix incomplete or failing')
    require({r['family'] for r in wide['results']}=={'llava','gemma3','gemma4','seq2seq'},'wide architecture coverage incomplete')
    require({(r['family'],r['case']) for r in wide['results']}=={
        ('llava','clip_last_layer_padded'),('gemma3','siglip_full_grid'),('gemma4','gemma4_rgb8'),
        *{('seq2seq',family+'/'+case) for family in ('t5','bart','switch_transformers') for case in ('unmasked','padded')}},'wide case inventory changed')
    for r in wide['results']:
        require(r['status']=='PASS_SCOPED','wide family failed')
        verify_mode(r['precision']['off'],'off');verify_mode(r['precision']['mixed'],'mixed')
        if r['family']=='gemma4':require(r['precision']['mixed']['policy_fallbacks']>0,'Gemma4 sensitive FP32 policy missing')
        for key,value in r['checks'].items():
            if key.startswith('mixed/') and isinstance(value,dict) and 'max_abs' in value:
                limit=2e-7 if '/reload/' in key else .02
                require(strict(value['max_abs'],limit),'wide Mixed maximum failed')
                if '/reload/' not in key:require(strict(value['rms'],.003),'wide Mixed RMS failed')
    native=components['native_v41']
    require(native['status']=='PASS_BOUNDED_NATIVE_VISION' and native['reports']['bounded']['independent_FP32_qualified'],'native bounded fixture failed')
    verify_mode(native['reports']['bounded']['precision'],'mixed')
    verify_primitive(components['wmma_mixed'],'mixed',0)
    verify_primitive(components['wmma_full_original'],'full-fp16',9)
    failures=verify_primitive(components['wmma_full_dot2'],'full-fp16',3)
    require(components['wmma_full_dot2']['full_oracle']=='rdna3-dot2','wrong matched FP16 oracle')
    require(all(r['label'].startswith('128x128x96') and 'vs FP32' in r['label'] for r in failures),'uncharacterized Full FP16 failure')
    applicable=[r for r in components['wmma_full_dot2']['results'] if int(re.match(r'\d+x\d+x(\d+)',r['label']).group(1))<=64]
    require(len(applicable)==28 and all(r['passed'] for r in applicable),'short-reduction Full FP16 policy failed')
    for key,count in [('shader727',727),('shader_wmma',4)]:
        q=components[key];require(q['pass'] and q['status_counts']=={'identical':count} and len(q['shaders'])==count,'shader reproduction failed')
    tests=components['tests']
    bound('libtest',tests['library']['binary_sha256'])
    for name,floor in [('library',1159),('cli',50),('gui',17)]:
        r=tests[name];require(r['passed']>=floor and r['failed']==0 and r['exit_code']==0,'test suite failed or incomplete: '+name)
    require(tests['oversized_fallback']['passed']==1 and tests['oversized_fallback']['failed']==0,'real oversized fallback was not executed')
    gui=components['gui_cli'];require(gui['status']=='PASS' and len(gui['workflows'])==8,'GUI Start coverage incomplete')
    bound('hierarchos-native-cli',gui['cli_sha256']);bound('gui_test',gui['gui_test_sha256'])
    for r in gui['workflows']:
        require(r['optimizer_updates']==2 and r['steps']==2,'GUI optimizer updates incomplete')
        verify_mode(r['precision'],r['precision']['requested'],r['workflow']!='lora')
    require(gui['unsupported_controls'] and all(r['exit_code']!=0 and not r['output_created'] for r in gui['unsupported_controls']),'unsupported precision silently accepted')
    from check_phase19z_current import verify_packed_status
    verify_packed_status(components['packed'])
    intermediate=components['intermediates']
    require(intermediate['maintained_cases']==93 and intermediate['diagnostic_checks']==1116,'FP32 diagnostic inventory changed')
    require(intermediate['strict_diagnostic_failures']==sum(not r['pass'] for r in intermediate['raw_diagnostics']),'FP32 diagnostic failures hidden')
    require(document['limitations'],'scoped release limitations missing')


def verify(path=DEFAULT,*,check_subreceipts=True):
    path=Path(path);document=json.loads(path.read_text());base=path.parent
    require(set(document['components'])==REQUIRED,'component binding inventory incomplete')
    components={}
    for name,row in document['components'].items():
        relative=Path(row['path']);require(not relative.is_absolute() and '..' not in relative.parts,'unsafe evidence path')
        p=base/relative;require(p.is_file() and digest(p)==row['sha256'],'component evidence changed: '+name)
        components[name]=json.loads(p.read_text())
    validate_payload(document,components)
    required_sources=set(subprocess.check_output(['git','-C',str(ROOT),'ls-files','hierarchos-vulkan/src','hierarchos-vulkan/shaders',
        'hierarchos-inference/src','hierarchos-native-cli/src','hierarchos-gui/src'],text=True).splitlines())
    required_sources.update({'hierarchos-vulkan/validation/check_unified_release.py',
        'hierarchos-vulkan/validation/qualify_unified_release.py','hierarchos-vulkan/validation/test_unified_release.py',
        '.github/workflows/workspace-check.yml'})
    require(required_sources.issubset(document['source_files']),'unified runtime/source inventory incomplete')
    for relative,sha in document['source_files'].items():require(digest(ROOT/relative)==sha,'current source changed: '+relative)
    for relative,sha in document['evidence_files'].items():require(digest(base/relative)==sha,'release evidence changed: '+relative)
    for name,row in document['binaries'].items():
        p=ROOT/row['path']
        if p.exists():require(digest(p)==row['sha256'],'current executable changed: '+name)
    require(subprocess.run(['git','merge-base','--is-ancestor',document['source_revision'],'HEAD'],cwd=ROOT,capture_output=True).returncode==0,'release source revision is not an ancestor')
    if check_subreceipts:
        for name in ('phase19x','phase19y','phase19z'):
            p=base/document['components'][name]['original_path']
            script={'phase19x':'check_phase19x_current.py','phase19y':'check_phase19y_full_attention.py','phase19z':'check_phase19z_current.py'}[name]
            run=subprocess.run([sys.executable,str(ROOT/'hierarchos-vulkan/validation'/script),'--receipt',str(p)],cwd=ROOT,capture_output=True,text=True)
            require(run.returncode==0,name+' failed: '+(run.stdout+run.stderr)[-600:])
        from phase19w_matrix import validate
        validate(components['phase19w'])
    return document,components


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',type=Path,default=DEFAULT)
    a=p.parse_args();verify(a.receipt);print('UNIFIED RELEASE: PASS_SCOPED_RELEASE',flush=True)


if __name__=='__main__':main()
