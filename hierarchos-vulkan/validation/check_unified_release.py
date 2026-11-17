"""Verify current FP32, Mixed, scoped FP16 and Phase XIX evidence together.

No PASS field is sufficient: numeric rows, optimizer counts, accumulator routes,
coverage and component hashes are checked again. Existing Phase XIX checkers
remain mandatory. Additional strict intermediate failures stay explicit.
"""
import argparse, hashlib, json, math, re, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'hierarchos-vulkan/validation/release_parity/multimodal_training/20261116-phase38-current-source/release-qualification.json'
IMAGE_COUNTS={'shared77':77,'seq2seq':6,'gemma3':4,'gemma4-legacy':1,'gemma4-rgb':1,'llava':4}
REQUIRED={'text166','fp32_replay','precision','wide','native_v41','wmma_mixed','wmma_full_original','wmma_full_dot2',
          'shader727','shader_wmma','tests','gui_cli','packed','intermediates','phase19w','phase19x','phase19y','phase19z',*IMAGE_COUNTS}
QUALIFICATION_TARGET={'cpu':'AVX512','gpu':'RDNA3'}
HOST_QUALIFICATIONS='hierarchos-vulkan/validation/fixtures/fixture_host_qualifications.json'
# Native V4.1 is a numerical qualification of the route it measured. These are
# the policy constants the route receipt is checked against; they are the
# production values, never a test-only override.
NATIVE_POLICY={'wmma_min_profitable_macs':16_000_000,'full_fp16_max_reduction':64}
NATIVE_CLAIM='native V4.1 numerical behaviour on the measured execution route'
NATIVE_NOT_CLAIMED=['current-source cooperative Mixed WMMA dispatch','native vision acceleration']
NATIVE_SEPARATE_EVIDENCE=['precision','wide','wmma_mixed','wmma_full_original','wmma_full_dot2']
# The wide-multimodal component qualifies the numerical behaviour of its nine
# fixtures on the route their own counters measured. At the released CPU-scale
# geometries the unchanged profitability policy dispatches nothing cooperatively,
# so this component is not independent current-source WMMA dispatch evidence; the
# precision and primitive components remain the mandatory proof of that route.
WIDE_POLICY={'wmma_min_profitable_macs':16_000_000,'full_fp16_max_reduction':64}
WIDE_CLAIM=('wide multimodal numerical behaviour of all nine model-family fixtures '
            'on the measured execution route')
WIDE_NOT_CLAIMED=['current-source cooperative Mixed WMMA accumulator dispatch']
WIDE_SEPARATE_EVIDENCE=['precision','wmma_mixed','wmma_full_original','wmma_full_dot2']
WIDE_FAMILIES=('llava','gemma3','gemma4','bart_image','t5_image','switch_image')
WIDE_CASES={('llava','clip_last_layer_padded'),('gemma3','siglip_full_grid'),('gemma4','gemma4_rgb8'),
            *{('seq2seq',family+'/'+case) for family in ('t5','bart','switch_transformers') for case in ('unmasked','padded')}}
WIDE_NUMERICAL_POLICY={'mixed_max_abs':.02,'mixed_rms':.003,'original_fp32_max_abs':2e-7,'rtol':0}


def require(ok,message):
    if not ok:raise ValueError(message)


def target_fixture_provenance():
    import fixture_toolchain as ft
    expected=ft.pin(ft.FIXTURES/'deepseek_v41_mixed_block128.json')
    historical=json.loads((ROOT/'HANDOFF-MANIFEST.json').read_text())
    report=ft.audit(ft.FIXTURES,expected,target_cpu=QUALIFICATION_TARGET['cpu'],historical=historical)
    require(not report['problems'],'target fixture provenance failed: '+str(report['problems']))
    return report


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
    require(sum(p)==sum(counts),'cooperative phase inventory disagrees with the dispatch counters')
    if mode=='off':require(counts==[0,0],'FP32 dispatched cooperative low precision')
    else:
        idx=0 if mode=='mixed' else 1
        require(counts[idx]>0 and counts[1-idx]==0,'missing or wrong cooperative accumulator route')
        require(p[0]>0 and p[2]>0,'missing cooperative forward or input-gradient phase')
        fallbacks=precision.get('full_fp16_reduction_fallbacks',0)
        if weight_grad:
            if p[1]>0:
                # The documented Full FP16 cap keeps reductions above 64 on their
                # FP32 kernel; a cooperative weight gradient and a recorded
                # reduction fallback cannot both describe the same route.
                require(mode!='full-fp16' or fallbacks==0,
                        'Full FP16 recorded a reduction fallback beside a cooperative weight gradient')
            elif mode=='full-fp16':
                # A full-parameter Full FP16 run may take the reduction cap on its
                # weight-gradient projection, but only as a recorded fallback.
                require(fallbacks>0,'missing Full FP16 weight-gradient phase without a recorded reduction fallback')
            else:
                require(False,'missing cooperative weight-gradient phase')
        else:
            require(p[1]==0,'unexpected cooperative weight-gradient phase')
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


def verify_replay(q,binaries):
    stages=[*IMAGE_COUNTS,'text166','deepseek215','native-v41']
    # Execution order is recorded separately; both inventories must be complete.
    require(q['pass'] and q['complete'],'FP32 full replay is incomplete')
    require(len(q['stages'])==9 and set(q['stages'])==set(stages),'FP32 replay stage inventory changed')
    require([r['stage'] for r in q['results']]==q['stages']
            and all(r['pass'] and r['returncode']==0 for r in q['results']),
            'FP32 replay execution is incomplete or failing')
    names=('libtest','image_text_pipeline_probe','gemma3_vision_probe','gemma4_vision_probe',
           'llava_pipeline_probe','hierarchos-vulkan-transformer-logits')
    require(len(q['binary_sha256'])==6 and set(q['binary_sha256'].values())==
            {binaries[name]['sha256'] for name in names},'FP32 replay executable inventory changed')


def verify_native_checks(native):
    require(set(native['checks'])=={'bounded','stress'},'native bounded/stress inventory incomplete')
    for label,checks in native['checks'].items():
        independent=[v for k,v in checks.items() if k.endswith('independent-FP32')]
        require(len(independent)==65,'native independent comparison inventory changed')
        for key,value in checks.items():
            require(strict(value['max_abs'],float('inf')) and strict(value['rms'],float('inf')),
                    'nonfinite native observation')
            if key.endswith('independent-FP32'):
                ok=strict(value['max_abs'],2e-7)
                require(value['strict_limit']==2e-7 and value['within_strict_limit']==ok,
                        'native strict diagnostic flag disagrees with numeric observation')
                if label=='bounded':require(ok,'native bounded independent FP32 comparison failed')
            elif key.endswith('mixed-vs-FP32'):
                require(strict(value['max_abs'],.02) and strict(value['rms'],.003),
                        'native Mixed numerical check failed')
            elif key.endswith('/reload'):
                require(value['max_abs']==0,'native export/reload diverged')
        require(native['reports'][label]['independent_FP32_qualified']==
                all(strict(v['max_abs'],2e-7) for v in independent),
                'native aggregate qualification disagrees with numeric observations')


def native_scope_summary(native):
    """The manifest-facing projection of the measured native V4.1 scope."""
    scope=native['qualification_scope']
    return {key:scope[key] for key in ('kind','measured_route','cooperative_mixed_executed',
                                       'not_claimed','separate_wmma_evidence')}


def verify_native_route(native):
    """Validate the measured route, not a required route label.

    A cooperative run must prove its dispatches and its phases. An FP32 run must
    prove that nothing dispatched and that its comparisons stayed bit-identical.
    Either way the recorded route has to agree with the executed counters and
    with the fixture geometry against the production profitability policy, so a
    receipt cannot relabel a fallback as WMMA execution or the reverse.
    """
    scope=native.get('qualification_scope')
    require(isinstance(scope,dict),'native V4.1 qualification scope missing')
    require(scope.get('claim')==NATIVE_CLAIM,'native V4.1 claim overstated or changed')
    require(scope.get('separate_wmma_evidence')==NATIVE_SEPARATE_EVIDENCE,
            'native scope does not bind the separate current-source WMMA evidence')
    kinds=set();measured=set()
    for label in ('bounded','stress'):
        precision=native['reports'][label]['precision'];route=native['reports'][label].get('route')
        require(isinstance(route,dict),f'native {label} route metadata missing')
        require(precision['requested']=='mixed','native bounded/stress precision request changed')
        require(precision['master_weights']=='fp32' and precision['optimizer_state']=='fp32',
                'native master/optimizer precision changed')
        require(precision['automatic_loss_scaling'] is False,'native V4.1 automatic loss scaling was claimed')
        counts=mode_counts(precision);phases_list=phases(precision)
        require(route['accumulation_dispatches']==counts and route['phases']==phases_list,
                'native route counters disagree with the executed precision report')
        require(counts[1]==0,'native V4.1 executed Full FP16')
        cooperative=counts[0]>0
        require(route['cooperative_mixed_executed'] is cooperative,
                'native route dispatch flag disagrees with the executed counters')
        require(route['measured_route']==('cooperative_mixed' if cooperative else 'fp32_no_cooperative_dispatch'),
                'native measured route label disagrees with the executed counters')
        require(route['scope']==('mixed_cooperative_numerical' if cooperative else 'numerical_fp32_route'),
                'native qualification scope disagrees with the executed route')
        require(route['policy']==NATIVE_POLICY,'native profitability/numerical policy changed')
        if cooperative:require(all(value>0 for value in phases_list),
                'native cooperative route lacks forward, weight- or input-gradient accumulation')
        else:require(phases_list==[0,0,0],'native FP32 route recorded cooperative phases')
        geometry=route['geometry'];n=geometry['n_vit_tokens'];dim=geometry['vision_dim']
        inter=geometry['vision_inter_dim'];patch=geometry['patch_reduction']
        require(all(value>0 for value in (n,dim,inter,patch)),'native fixture geometry is unusable')
        expected=dict(patch_embed=n*dim*patch,wqkv=n*3*dim*dim,w1=n*inter*dim,w2=n*dim*inter,w3=n*inter*dim)
        require(geometry['projections']==expected,'native projection geometry changed')
        require(geometry['max_projection_macs']==max(expected.values())
                and geometry['widest']==max(expected,key=expected.get),
                'native widest projection disagrees with the recorded geometry')
        eligible=geometry['max_projection_macs']>=NATIVE_POLICY['wmma_min_profitable_macs']
        require(route['admission']==('admitted' if eligible else 'declined'),
                'native admission flag disagrees with the profitability policy')
        require(route['declined_by']==(None if cooperative else
                ('wmma_shape_profitable' if not eligible else 'numerical_policy')),
                'native decline reason disagrees with the measured geometry')
        require(not cooperative or eligible,'cooperative dispatch claimed below the profitability floor')
        label_route='cooperative_parity' if cooperative else 'fallback_parity'
        for key,row in native['checks'][label].items():
            if key.endswith('mixed-vs-FP32'):require(row.get('route')==label_route,
                    'native comparison route label changed')
        if not cooperative:
            require(all(row['max_abs']==0 for key,row in native['checks'][label].items()
                        if key.endswith('mixed-vs-FP32')),
                    'a declined cooperative request did not stay on the FP32 kernels')
        control=native['reports'][label].get('control_route') or {}
        require(control.get('accumulation_dispatches')==[0,0] and control.get('phases')==[0,0,0],
                'native FP32 control dispatched cooperative accumulation')
        kinds.add(route['scope']);measured.add(tuple(counts))
    require(len(kinds)==1,'bounded and stress fixtures measured different execution routes')
    require(len(measured)==1,'bounded and stress dispatch inventories differ')
    kind=kinds.pop();current=list(measured.pop())
    require(scope['kind']==kind,'native qualification kind disagrees with the measured route')
    cooperative=kind=='mixed_cooperative_numerical'
    require(scope['cooperative_mixed_executed']==cooperative,'native scope dispatch flag disagrees with the route')
    require(scope['measured_route']==('cooperative_mixed' if cooperative else 'fp32_no_cooperative_dispatch'),
            'native scope route label changed')
    require(scope['not_claimed']==([] if cooperative else NATIVE_NOT_CLAIMED),
            'native scope understates what the measured route did not qualify')
    historical=scope.get('historical_pre_gate_receipts')
    require(isinstance(historical,list) and historical,'native historical dispatch disclosure missing')
    for row in historical:
        require(row['accumulation_dispatches'][1]==0 and row['accumulation_dispatches'][0]>0,
                'historical disclosure is not a cooperative Mixed receipt')
        require(row['release_source_revision']!=native['source_revision'],
                'historical cooperative receipt is presented as current source')
        require(isinstance(row['native_receipt_sha256'],str) and len(row['native_receipt_sha256'])==64,
                'historical disclosure lacks a receipt digest')
        require(strict(row['bounded_max_abs'],float('inf')),'historical disclosure lacks its numerical result')
    require(any(row['accumulation_dispatches']!=current for row in historical),
            'historical disclosure does not describe a different execution route')


def wide_scope_summary(wide):
    """The manifest-facing projection of the measured wide-multimodal scope."""
    scope=wide['qualification_scope']
    return {key:scope[key] for key in ('kind','measured_route','cooperative_mixed_executed',
                                       'not_claimed','separate_wmma_evidence')}


def verify_wide_case(case,fixtures):
    """Validate one wide case against its own counters and bound provenance.

    The released wide fixtures are CPU-scale, so the unchanged profitability
    policy declines every cooperative projection and the `mixed` request runs the
    FP32 kernels. The route the counters measured is verified instead of
    demanding an accumulator that never executed; a case that did dispatch
    cooperatively must prove it, and a case that did not must say so and keep
    every numerical gate the component already enforced.
    """
    require(set(case['precision'])=={'off','mixed'},'wide case must record both precision arms')
    off=case['precision']['off'];mixed=case['precision']['mixed']
    verify_mode(off,'off')
    route=case.get('route')
    require(isinstance(route,dict),'wide case measured route metadata missing')
    counts=mode_counts(mixed);phase_counts=phases(mixed)
    require(route.get('configured_precision')=='mixed','wide case configured precision changed')
    require(route.get('accumulation_dispatches')==counts,
            'wide route counters disagree with the executed precision report')
    require(route.get('phases')==phase_counts,
            'wide route phases disagree with the executed precision report')
    require(sum(phase_counts)==sum(counts),
            'wide cooperative phase inventory disagrees with the dispatch counters')
    cooperative=counts[0]>0
    require(route.get('cooperative_mixed_wmma_dispatched') is cooperative,
            'wide route dispatch flag disagrees with the executed counters')
    require(route.get('full_fp16_wmma_dispatched') is (counts[1]>0),
            'wide route Full FP16 flag disagrees with the executed counters')
    require(route.get('observed_route')==('mixed_cooperative_route' if cooperative
            else 'mixed_configured_non_cooperative_route'),
            'wide observed route label disagrees with the executed counters')
    require(route.get('scope')==('wide_cooperative_mixed_numerical' if cooperative
            else 'wide_numerical_non_cooperative_route'),
            'wide case scope disagrees with the executed route')
    telemetry=route.get('fallback_telemetry')
    require(isinstance(telemetry,dict),'wide route fallback telemetry missing')
    for key,value in telemetry.items():
        require(isinstance(value,int) and value>=0,'wide fallback counter is not a count: '+key)
        require(mixed.get(key)==value,
                'wide fallback telemetry disagrees with the executed report: '+key)
    provenance=case.get('provenance')
    require(isinstance(provenance,dict),'wide case provenance missing')
    for label in ('fixtures','oracle'):
        rows=provenance.get(label)
        require(isinstance(rows,dict) and rows,'wide '+label+' provenance missing')
        for path,sha in rows.items():
            require(isinstance(sha,str) and len(sha)==64,'wide '+label+' digest invalid: '+path)
            require(fixtures.get(path)==sha,
                    'wide '+label+' is not bound by the family fixture map: '+path)
    if cooperative:
        verify_mode(mixed,'mixed')
    else:
        require(counts==[0,0] and phase_counts==[0,0,0],
                'wide non-cooperative route recorded cooperative work')
        require(mixed['requested']=='mixed','wide Mixed request changed')
        require(mixed['master_weights']=='fp32' and mixed['optimizer_state']=='fp32',
                'wide Mixed master/optimizer precision changed')


def verify_wide_route(wide):
    """Validate the wide scope against the route its nine cases measured.

    Zero cooperative dispatches are accepted only when the component says so and
    stops claiming a cooperative qualification. The component must still bind the
    independent current-source WMMA evidence, so narrowing this row cannot weaken
    the release's Mixed WMMA claim.
    """
    scope=wide.get('qualification_scope')
    require(isinstance(scope,dict),'wide multimodal qualification scope missing')
    require(scope.get('claim')==WIDE_CLAIM,'wide multimodal claim overstated or changed')
    require(scope.get('separate_wmma_evidence')==WIDE_SEPARATE_EVIDENCE,
            'wide scope does not bind the separate current-source WMMA evidence')
    require(scope.get('policy')==WIDE_POLICY,'wide WMMA profitability/numerical policy changed')
    require(all(name in REQUIRED for name in WIDE_SEPARATE_EVIDENCE),
            'separate current-source WMMA evidence is not a required release component')
    routes={case['route']['observed_route'] for case in wide['results']}
    require(len(routes)==1,'wide cases measured different execution routes')
    cooperative=all(case['route']['cooperative_mixed_wmma_dispatched'] for case in wide['results'])
    require(scope.get('cooperative_mixed_executed') is cooperative,
            'wide scope dispatch flag disagrees with the measured route')
    require(scope.get('measured_route')==routes.pop(),
            'wide scope route label disagrees with the measured route')
    require(scope.get('kind')==wide['results'][0]['route']['scope'],
            'wide scope kind disagrees with the measured route')
    require(scope.get('not_claimed')==([] if cooperative else WIDE_NOT_CLAIMED),
            'wide scope understates what the measured route did not qualify')
    require(isinstance(wide.get('source_revision'),str) and len(wide['source_revision'])==40,
            'wide component source revision missing')
    require(subprocess.run(['git','merge-base','--is-ancestor',wide['source_revision'],'HEAD'],
            cwd=ROOT,capture_output=True).returncode==0,
            'wide component source revision is not in this repository history')
    history=scope.get('historical_pre_gate_receipts')
    require(isinstance(history,list) and len(history)==1,'wide historical pre-gate disclosure missing')
    row=history[0]
    require(row.get('release_source_revision')!=wide['source_revision'],
            'historical cooperative receipt is presented as current source')
    require(isinstance(row.get('component_sha256'),str) and len(row['component_sha256'])==64,
            'wide historical disclosure lacks its component digest')
    require(isinstance(row.get('release_source_revision'),str)
            and len(row['release_source_revision'])==40,
            'wide historical disclosure lacks its release revision')
    require(subprocess.run(['git','merge-base','--is-ancestor',row['release_source_revision'],'HEAD'],
            cwd=ROOT,capture_output=True).returncode==0,
            'wide historical disclosure revision is not in this repository history')
    recorded={(path['family'],path['case']):path for path in row.get('routes',[])}
    require(set(recorded)==WIDE_CASES and len(recorded)==9,
            'wide historical disclosure does not cover the qualified case inventory')
    for path in recorded.values():
        require(path['accumulation_dispatches'][1]==0 and path['accumulation_dispatches'][0]>0,
                'wide historical disclosure is not a cooperative Mixed receipt')
    require(any(recorded[(case['family'],case['case'])]['accumulation_dispatches']
                !=case['route']['accumulation_dispatches'] for case in wide['results']),
            'wide historical disclosure does not describe a different execution route')


def verify_producer(component, label):
    """Re-read the qualification producer from the revision it claims.

    Ancestry alone cannot distinguish the runtime build revision from the
    revision of the python producer that generated a receipt: a producer can be
    committed after the binaries it measures. The recorded producer digest is
    therefore verified against the git blob at the declared revision, and the
    runtime fact is required to stay in its own block.
    """
    producer=component.get('producer')
    require(isinstance(producer,dict),label+' qualification producer provenance missing')
    path=producer.get('path');revision=producer.get('revision');sha=producer.get('sha256')
    require(isinstance(path,str) and path.endswith('.py') and not Path(path).is_absolute()
            and '..' not in Path(path).parts,'invalid qualification producer path: '+label)
    require(isinstance(revision,str) and len(revision)==40,'invalid qualification producer revision: '+label)
    require(isinstance(sha,str) and len(sha)==64,'invalid qualification producer digest: '+label)
    sources=component.get('source_files') or component.get('source_sha256')
    require(isinstance(sources,dict),label+' component source map missing')
    require(sources.get(path)==sha,
            label+' qualification producer digest disagrees with the component source map')
    require(subprocess.run(['git','merge-base','--is-ancestor',revision,'HEAD'],
            cwd=ROOT,capture_output=True).returncode==0,
            label+' qualification producer revision is not in this repository history')
    blob=subprocess.run(['git','show',revision+':'+path],cwd=ROOT,capture_output=True)
    require(blob.returncode==0,label+' qualification producer is absent at its declared revision')
    require(hashlib.sha256(blob.stdout).hexdigest()==sha,
            label+' qualification producer digest does not match its declared revision')
    runtime=component.get('runtime')
    require(isinstance(runtime,dict),label+' runtime provenance block missing')
    require(runtime.get('source_revision')==component['source_revision'],
            label+' runtime source revision disagrees with the component source revision')
    require(runtime.get('binary_sha256')==component['binary_sha256'],
            label+' runtime binary digests disagree with the component executable map')


def verify_overflow_route(checks):
    """The joint skip/backoff must be executed, never explained away.

    A 2^80 joint scale overflows only where the shared transaction really
    quantizes its scaled gradients to FP16 GEMM operands, so the strict 2^78/2^76
    history is required on the arm that dispatched. An arm that the released
    reduction cap forces onto FP32 must prove that fallback instead of claiming a
    skip, and a below-threshold observation can never stand in for the test.
    """
    require(not any('FP32-scaler-reassigned' in key for key in checks),
            'a below-threshold result is presented as the joint overflow qualification')
    diagnostic=checks.get('vision-overflow/below-threshold-diagnostic')
    require(isinstance(diagnostic,dict),'below-threshold joint overflow diagnostic missing')
    require(diagnostic.get('cooperative_dispatches')==0 and diagnostic.get('skipped_steps')==0,
            'below-threshold joint overflow diagnostic claims dispatch or a skip')
    for mode in ('mixed','full-fp16'):
        label='vision-overflow-'+mode
        state=checks.get(label+'/state');restored=checks.get(label+'/restored-state')
        if state is not None or restored is not None:
            require(isinstance(state,dict) and isinstance(restored,dict),
                    'joint overflow state/restore inventory incomplete: '+label)
            require(state['skipped_steps']==2 and state['scale']==2.**78,
                    'joint overflow skip/backoff failed: '+label)
            require(restored['skipped_steps']==4 and restored['scale']==2.**76,
                    'restored global scaler history failed: '+label)
            for component in ('image','text'):
                row=checks.get(label+'/'+component+'-unchanged')
                require(isinstance(row,dict) and row
                        and all(value['max_abs']==0 for value in row.values()),
                        'joint overflow changed masters: '+label+'/'+component)
            continue
        require(mode=='full-fp16','joint overflow recovery was not executed: '+label)
        row=checks.get(label+'/reduction-cap-fp32')
        require(isinstance(row,dict),'joint overflow route is neither executed nor policy-explained: '+label)
        require(row.get('geometry')=='reduction-cap',
                'joint overflow explanation is not the released reduction cap: '+label)
        require(row.get('cooperative_dispatches')==0 and row.get('skipped_steps')==0,
                'reduction-cap joint arm claims dispatch or a skip: '+label)
        require(row.get('full_fp16_reduction_fallbacks',0)>0,
                'reduction-cap joint arm did not record its FP32 fallback: '+label)
        require(isinstance(row.get('reason'),str) and row['reason'],
                'joint overflow reduction-cap disclosure lacks its reason: '+label)


def verify_intermediates(q):
    rows=q['raw_diagnostics']
    require(q['maintained_cases']==93 and q['diagnostic_checks']==len(rows)==1116,
            'FP32 diagnostic inventory changed')
    for row in rows:
        require(strict(row['max_abs'],float('inf')) and row['pass']==strict(row['max_abs'],2e-7),
                'FP32 diagnostic flag disagrees with numeric observation')
    require(q['strict_diagnostic_failures']==sum(not strict(r['max_abs'],2e-7) for r in rows),
            'FP32 diagnostic failures hidden')


def validate_payload(document,components):
    require(document.get('schema')==1 and document.get('status')=='PASS_SCOPED_RELEASE','invalid unified release status')
    require(document.get('qualification_target')==QUALIFICATION_TARGET,'wrong release qualification target')
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
    verify_replay(components['fp32_replay'],binaries)
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
        if isinstance(row,dict) and 'redirect' in row:
            # A comparison may move to another fixture, but only to one that ran.
            require(isinstance(row.get('reason'),str) and row['reason'],
                    'reassigned comparison lacks its reason: '+key)
            require(row['redirect'] in precision['checks'],
                    'reassigned comparison target is missing: '+key)
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
    verify_overflow_route(precision['checks'])
    verify_producer(precision,'precision')
    for mode in ('mixed','full-fp16'):
        require('wide-'+mode+'/masters-vs-FP32' in precision['checks'],
                'long-row master comparison is missing at the wide fixture: '+mode)
    wide=components['wide'];require(wide['status']=='PASS_SCOPED' and len(wide['results'])==9,'wide matrix incomplete or failing')
    require({r['family'] for r in wide['results']}=={'llava','gemma3','gemma4','seq2seq'},'wide architecture coverage incomplete')
    require({(r['family'],r['case']) for r in wide['results']}==WIDE_CASES,'wide case inventory changed')
    require(set(wide.get('fixture_sha256',{}))=={'llava','gemma3','gemma4','seq2seq'},'wide fixture provenance incomplete')
    require(set(wide.get('binary_sha256',{}))=={'llava','gemma3','gemma4','seq2seq'},'wide executable inventory changed')
    require(set(wide.get('oracle_contract',{}))=={'llava','gemma3','gemma4','seq2seq'},'wide oracle provenance incomplete')
    for family,row in wide['oracle_contract'].items():
        require(row.get('cpu')=='AVX2' and row.get('threads')==1,'wide CPU oracle pinning changed: '+family)
        require(row.get('manifest_sha256')==wide['fixture_sha256'][family].get('manifest.json'),
                'wide oracle manifest disagrees with the fixture map: '+family)
        require(isinstance(row.get('oracle_sha256'),str) and len(row['oracle_sha256'])==64,
                'wide oracle provenance digest missing: '+family)
    require(len(wide.get('commands',[]))==18,'wide execution inventory is not nine cases times two arms')
    require({(row['mode'],row['exit_code']) for row in wide['commands']}=={('off',0),('mixed',0)},
            'wide execution commands failed or changed mode')
    require(wide.get('policy')==WIDE_NUMERICAL_POLICY,'wide numerical policy changed')
    require(isinstance(wide.get('source_files'),dict) and wide['source_files'],
            'wide component source provenance missing')
    for relative,sha in wide['source_files'].items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                'unsafe wide source path: '+relative)
        require(digest(ROOT/relative)==sha,'wide component source changed: '+relative)
    for r in wide['results']:
        require(r['status']=='PASS_SCOPED','wide family failed')
        verify_wide_case(r,wide['fixture_sha256'][r['family']])
        if r['family']=='gemma4':require(r['precision']['mixed']['policy_fallbacks']>0,'Gemma4 sensitive FP32 policy missing')
        for key,value in r['checks'].items():
            if key.startswith('mixed/') and isinstance(value,dict) and 'max_abs' in value:
                limit=2e-7 if '/reload/' in key else .02
                require(strict(value['max_abs'],limit),'wide Mixed maximum failed')
                if '/reload/' in key:require(value['max_abs']==0,'wide export/reload diverged')
                else:require(strict(value['rms'],.003),'wide Mixed RMS failed')
            if key.startswith('off/') and isinstance(value,dict) and 'within_strict_limit' in value:
                require(value['within_strict_limit']==strict(value['max_abs'],2e-7),
                        'wide strict diagnostic flag disagrees with numeric observation')
    verify_wide_route(wide)
    verify_producer(wide,'wide')
    native=components['native_v41']
    require(native['status']=='PASS_BOUNDED_NATIVE_VISION' and native['reports']['bounded']['independent_FP32_qualified'],'native bounded fixture failed')
    verify_native_checks(native)
    verify_native_route(native)
    require(document.get('native_v41_scope')==native_scope_summary(native),
            'release manifest native V4.1 scope disagrees with the measured component')
    require(document.get('wide_multimodal_scope')==wide_scope_summary(wide),
            'release manifest wide multimodal scope disagrees with the measured component')
    for capability in document['precision_capabilities']:
        if capability['family'] not in WIDE_FAMILIES:continue
        mixed_capability=capability['mixed']
        require(mixed_capability.get('execution_route')==wide['qualification_scope']['measured_route'],
                'manifest mixed route disagrees with the wide component: '+capability['family'])
        require(mixed_capability.get('independent_cooperative_qualification')
                is wide['qualification_scope']['cooperative_mixed_executed'],
                'manifest cooperative qualification disagrees with the wide component: '+capability['family'])
    verify_primitive(components['wmma_mixed'],'mixed',0)
    verify_primitive(components['wmma_full_original'],'full-fp16',9)
    failures=verify_primitive(components['wmma_full_dot2'],'full-fp16',3)
    for key in ('wmma_mixed','wmma_full_original','wmma_full_dot2'):
        bound('wmma_qualify',components[key]['producer']['binary_sha256'])
    require(components['wmma_full_dot2']['full_oracle']=='rdna3-dot2','wrong matched FP16 oracle')
    require(all(r['label'].startswith('128x128x96') and 'vs FP32' in r['label'] for r in failures),'uncharacterized Full FP16 failure')
    applicable=[r for r in components['wmma_full_dot2']['results'] if int(re.match(r'\d+x\d+x(\d+)',r['label']).group(1))<=64]
    require(len(applicable)==28 and all(r['passed'] for r in applicable),'short-reduction Full FP16 policy failed')
    # The historical component key is retained; the manifest now includes the
    # isolated RDNA3 key-state variant in addition to the original 727 recipes.
    for key,count in [('shader727',728),('shader_wmma',4)]:
        q=components[key];require(q['pass'] and q['status_counts']=={'identical':count} and len(q['shaders'])==count,'shader reproduction failed')
    artifacts=[r['artifact'] for r in components['shader727']['shaders']]
    require(len(set(artifacts))==728 and {'rwkv_key_state_forward_fused_wide.spv',
            'rwkv_key_state_forward_fused_wide_amd_rdna3.spv'}.issubset(artifacts),
            'legacy and isolated RDNA3 shader variants must both be reproduced')
    tests=components['tests']
    bound('libtest',tests['library']['binary_sha256'])
    bound('cli_test',tests['cli']['binary_sha256'])
    bound('gui_test',tests['gui']['binary_sha256'])
    bound('libtest',tests['oversized_fallback']['binary_sha256'])
    bound('gui_test',tests['gui_hardware']['binary_sha256'])
    for name,floor in [('library',1159),('cli',50),('gui',17)]:
        r=tests[name];require(r['passed']>=floor and r['failed']==0 and r['exit_code']==0,'test suite failed or incomplete: '+name)
    require(tests['oversized_fallback']['passed']==1 and tests['oversized_fallback']['failed']==0,'real oversized fallback was not executed')
    model_fallback=tests['oversized_fallback']['model_export']
    bound('image_text_pipeline_probe',model_fallback['binary_sha256'])
    require(model_fallback['status']=='PASS_INTERNAL_FALLBACK_ONLY' and model_fallback['optimizer_steps']==[1,2],
            'internal oversized fallback/export execution incomplete')
    require(model_fallback['precision']['full_fp16_reduction_fallbacks']>0
            and model_fallback['precision']['mixed_fp32_accumulation']==0,'oversized model did not execute Full-policy FP32 fallback')
    for key,row in model_fallback['checks'].items():
        for value in ([row] if 'max_abs' in row else row.values()):
            require(strict(value['max_abs'],.05) and strict(value['rms'],.01),'oversized fallback numerical check failed')
            if '/reload/' in key:require(value['max_abs']==0,'oversized fallback export/reload diverged')
    gui=components['gui_cli'];require(gui['status']=='PASS' and len(gui['workflows'])==8,'GUI Start coverage incomplete')
    require({(r['workflow'],r['precision']['requested']) for r in gui['workflows']}==
            {(name,mode) for name in ('vision','full','full-finetune','lora') for mode in ('mixed','full-fp16')},
            'GUI workflow/mode inventory changed')
    bound('hierarchos-native-cli',gui['cli_sha256']);bound('gui_test',gui['gui_test_sha256'])
    for r in gui['workflows']:
        require(r['optimizer_updates']==2 and r['steps']==2,'GUI optimizer updates incomplete')
        verify_mode(r['precision'],r['precision']['requested'],r['workflow']!='lora')
    require(len(gui['unsupported_controls'])==4 and {r['control'] for r in gui['unsupported_controls']}==
            {'invalid-mode','unqualified-full-profile','wrong-checkpoint-precision','partial-window-checkpoint'},
            'precision rejection controls incomplete')
    require(all(r['exit_code']!=0 and not r['output_created'] for r in gui['unsupported_controls']),'unsupported precision silently accepted')
    from check_phase19z_current import verify_packed_status
    verify_packed_status(components['packed'])
    intermediate=components['intermediates']
    verify_intermediates(intermediate)
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
        '.github/workflows/workspace-check.yml','RDNA3_RELEASE_CONTRACT.md','.gitattributes',HOST_QUALIFICATIONS})
    require(required_sources.issubset(document['source_files']),'unified runtime/source inventory incomplete')
    for relative,sha in document['source_files'].items():require(digest(ROOT/relative)==sha,'current source changed: '+relative)
    require(document.get('fixture_provenance')==target_fixture_provenance(),'target fixture provenance report changed')
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
