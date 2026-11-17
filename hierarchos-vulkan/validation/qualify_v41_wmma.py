"""Qualify native V4.1 vision numerically on the route it actually measured.

The bounded independent fixture must pass the original strict FP32 policy.
Every mixed comparison uses the unchanged mixed precision max/RMS policy.

The mixed request is a *request*, not a route: the receipt therefore records
which execution path ran. When `wmma_shape_profitable` declines every native
vision projection at the fixture geometry, the mixed request executes the FP32
kernels and records zero cooperative accumulations. That is a numerical
qualification of the FP32 route and is never presented as cooperative WMMA
dispatch evidence; the earlier pre-gate cooperative receipts are disclosed from
their own sealed roots instead of being restated as current measurement.
"""
import argparse,json,os,subprocess
from pathlib import Path
from qualify_precision_training import digest,metric,require,write,ROOT

JOINT='deepseek_v41::model::tests::training::native_vision::deepseek_v41_native_multimodal_joint_two_adamw_steps_match_frozen_cpu'
POLICY=dict(wmma_min_profitable_macs=16_000_000,full_fp16_max_reduction=64)
CLAIM='native V4.1 numerical behaviour on the measured execution route'
NOT_CLAIMED=['current-source cooperative Mixed WMMA dispatch','native vision acceleration']
SEPARATE_WMMA=['precision','wide','wmma_mixed','wmma_full_original','wmma_full_dot2']
HISTORICAL=['20261112-unified-release','20261115-phase25-current-source']
RELEASE_ROOTS=ROOT/'hierarchos-vulkan/validation/release_parity/multimodal_training'


def geometry(fixture):
    """Widest native projection MAC count implied by the fixture geometry."""
    config,image=fixture['config'],fixture['image']
    tokens=image['n_vit_h']*image['n_vit_w'];dim=config['vision_dim'];inter=config['vision_inter_dim']
    patch=config['vision_patch_size']**2*3
    require(tokens>0 and dim>0 and inter>0 and patch>0,'unusable native vision geometry')
    projections=dict(patch_embed=tokens*dim*patch,wqkv=tokens*3*dim*dim,w1=tokens*inter*dim,
        w2=tokens*dim*inter,w3=tokens*inter*dim)
    widest=max(projections,key=projections.get)
    return dict(n_vit_tokens=tokens,vision_dim=dim,vision_inter_dim=inter,patch_reduction=patch,
        projections=projections,widest=widest,max_projection_macs=projections[widest])


def route(label,precision,fixture_geometry):
    """Record the route the executed counters prove, and why it was taken."""
    counts=list(precision['accumulation_dispatches']);phases=list(precision['phases'])
    require(precision['requested']=='mixed','wrong native precision request')
    require(precision['master_weights']=='fp32' and precision['optimizer_state']=='fp32',
        'master/optimizer precision changed')
    require(precision['automatic_loss_scaling'] is False,'native V4.1 claims automatic loss scaling')
    require(counts[1]==0,'native V4.1 silently executed Full FP16')
    require((counts[0]>0)==any(phases),'native route counters disagree with the recorded phases')
    cooperative=counts[0]>0
    eligible=fixture_geometry['max_projection_macs']>=POLICY['wmma_min_profitable_macs']
    if cooperative:
        require(all(value>0 for value in phases),
            'cooperative route lacks forward, weight- or input-gradient accumulation')
        require(eligible,'cooperative dispatch claimed below the profitability floor')
    else:
        require(counts==[0,0] and phases==[0,0,0],'native V4.1 kept a nonzero accumulation on the FP32 route')
    return dict(requested=precision['requested'],
        source='measured probe pre/post accumulation counters',
        scope='mixed_cooperative_numerical' if cooperative else 'numerical_fp32_route',
        measured_route='cooperative_mixed' if cooperative else 'fp32_no_cooperative_dispatch',
        cooperative_mixed_executed=cooperative,
        accumulation_dispatches=counts,phases=phases,policy=dict(POLICY),geometry=fixture_geometry,
        admission='admitted' if eligible else 'declined',
        declined_by=None if cooperative else ('wmma_shape_profitable' if not eligible else 'numerical_policy'))


def historical_receipts():
    """Read the earlier cooperative V4.1 receipts from their own sealed roots."""
    rows=[]
    for name in HISTORICAL:
        release=RELEASE_ROOTS/name/'release-qualification.json'
        require(release.is_file(),f'historical sealed root missing: {name}')
        document=json.loads(release.read_text());binding=document['components']['native_v41']
        receipt=release.parent/binding['path']
        require(receipt.is_file() and digest(receipt)==binding['sha256'],
            f'historical native receipt changed: {name}')
        value=json.loads(receipt.read_text());precision=value['reports']['bounded']['precision']
        rows.append(dict(root=name,release_source_revision=document['source_revision'],
            native_receipt_sha256=binding['sha256'],
            accumulation_dispatches=precision['accumulation_dispatches'],phases=precision['phases'],
            bounded_max_abs=max(row['max_abs'] for key,row in value['checks']['bounded'].items()
                if key.endswith('independent-FP32')),
            qualification='cooperative Mixed dispatch (pre-gate implementation, historical only)'))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--stress-fixture',type=Path)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--libtest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);commands=[];checks={};reports={}
    source_names=subprocess.check_output(['git','-C',str(ROOT),'ls-files','--cached','--others','--exclude-standard',
        'hierarchos-vulkan/src','hierarchos-vulkan/shaders','hierarchos-vulkan/Cargo.toml','hierarchos-vulkan/Cargo.lock'],text=True).splitlines()
    source_names += [str(Path(__file__).relative_to(ROOT)).replace('\\','/'),
        'hierarchos-vulkan/validation/prepare_v41_wmma_fixture.py','hierarchos-vulkan/validation/qualify_precision_training.py']
    sources={name:digest(ROOT/name) for name in set(source_names)}
    binaries={name:digest(path) for name,path in [('probe',a.binary),('libtest',a.libtest)]}
    for label,fixture in [('bounded',a.fixture),('stress',a.stress_fixture)]:
        if fixture is None:continue
        cpu=json.loads(fixture.read_text());report={};rows={};fixture_geometry=geometry(cpu)
        width={'AVX2':'8','AVX512':'16'}.get(cpu['oracle']['cpu'])
        require(width is not None,'unrecognized immutable CPU ISA')
        routes={}
        for mode in ('off','mixed'):
            output=a.output/f'{label}-{mode}.json'
            command=[str(a.binary.resolve()),str(fixture.resolve()),str(output.resolve())]
            result=subprocess.run(command,env={**os.environ,'HIERARCHOS_WMMA_MODE':mode,'HIERARCHOS_ATEN_VECTOR_WIDTH':width},capture_output=True,text=True)
            (a.output/f'{label}-{mode}.log').write_text(result.stdout+result.stderr)
            commands.append(dict(command=command,mode=mode,oracle_vector_width=width,returncode=result.returncode))
            require(result.returncode==0,f'{label}-{mode} failed')
            report[mode]=json.loads(output.read_text())
            precision=report[mode]['precision'];require(precision['requested']==mode,'wrong native precision')
            require(precision['cpu_arithmetic_vector_width']==int(width),'native arithmetic does not match the recorded oracle ISA')
            if mode=='off':
                require(precision['accumulation_dispatches']==[0,0] and precision['phases']==[0,0,0],
                    'native FP32 control dispatched cooperative accumulation')
                routes[mode]=dict(requested='off',scope='fp32_control',measured_route='fp32_control',
                    cooperative_mixed_executed=False,accumulation_dispatches=[0,0],phases=[0,0,0],
                    policy=dict(POLICY),geometry=fixture_geometry)
            else:
                routes[mode]=route(label,precision,fixture_geometry)
            reload=metric(report[mode]['steps'][2]['projected'],report[mode]['reload'],2e-7)
            require(reload['max_abs']==0,'native vision reload mismatch');rows[mode+'/reload']=reload
        cooperative=routes['mixed']['cooperative_mixed_executed']
        parity_label='cooperative_parity' if cooperative else 'fallback_parity'
        def compare(name,native,cpu_values,mixed):
            strict=metric(native,cpu_values,2e-7 if label=='bounded' else float('inf'))
            strict.update(strict_limit=2e-7,within_strict_limit=strict['max_abs']<2e-7)
            rows[name+'/independent-FP32']=strict
            mixed_row=metric(native,mixed,.02,.003);mixed_row.update(route=parity_label)
            rows[name+'/mixed-vs-FP32']=mixed_row
        for i in range(3):
            x,y=report['off']['steps'][i],report['mixed']['steps'][i];c=cpu['steps'][i]
            compare(f'{i}/projected',x['projected'],c['projected']['values'],y['projected'])
            if i<2:
                compare(f'{i}/patch-gradient',x['patch_gradient'],c['patch_gradient']['values'],y['patch_gradient'])
                for key,ckey in [('gradients','gradients'),('parameters','updated_parameters')]:
                    require(x[key].keys()==y[key].keys()==c[ckey].keys(),'native tensor inventory mismatch')
                    for name,tensor in x[key].items():
                        compare(f'{i}/{key}/{name}',tensor['values'],c[ckey][name]['values'],y[key][name]['values'])
        parity=[value['max_abs'] for key,value in rows.items() if key.endswith('mixed-vs-FP32')]
        independent=[value for key,value in rows.items() if key.endswith('independent-FP32')]
        require(len(parity)==65 and len(independent)==65,f'native comparison inventory changed in {label}')
        if not cooperative:
            require(all(value==0 for value in parity),
                'a declined cooperative request produced non-identical parity, so it did not stay on the FP32 kernels')
        routes['mixed']['fallback_parity_exact']=all(value==0 for value in parity)
        checks[label]=rows
        reports[label]=dict(fixture_sha256=digest(fixture),precision=report['mixed']['precision'],
            route=routes['mixed'],control_route=routes['off'],
            independent_FP32_qualified=all(row['within_strict_limit'] for row in independent),
            scope='bounded native ViT/aligner' if label=='bounded' else 'larger-magnitude diagnostic, not promoted')
    for mode in ('off','mixed'):
        command=[str(a.libtest.resolve()),JOINT,'--exact','--nocapture','--test-threads=1']
        result=subprocess.run(command,env={**os.environ,'HIERARCHOS_WMMA_MODE':mode},capture_output=True,text=True)
        log=result.stdout+result.stderr;(a.output/f'joint-{mode}.log').write_text(log)
        commands.append(dict(command=command,mode=mode,returncode=result.returncode))
        require(result.returncode==0 and '1 passed; 0 failed' in log,'native joint frozen FP32 test failed')
    require(all(digest(ROOT/name)==sha for name,sha in sources.items()),'source changed during execution')
    require(digest(a.binary)==binaries['probe'] and digest(a.libtest)==binaries['libtest'],'binary changed during execution')
    kinds={value['route']['scope'] for value in reports.values()}
    require(len(kinds)==1,'bounded and stress fixtures measured different execution routes')
    kind=kinds.pop();cooperative=kind=='mixed_cooperative_numerical'
    historical=historical_receipts()
    current=reports['bounded']['route']['accumulation_dispatches']
    require(any(row['accumulation_dispatches']!=current for row in historical),
        'historical cooperative receipts are not disclosed as a distinct earlier route')
    qualification_scope=dict(kind=kind,claim=CLAIM,
        measured_route=reports['bounded']['route']['measured_route'],
        cooperative_mixed_executed=cooperative,
        statement=('Native V4.1 is qualified numerically on the execution route it measured. '
            'The mixed request executed cooperative accumulation.' if cooperative else
            'Native V4.1 is qualified numerically on the execution route it measured: the mixed '
            'request executed the FP32 kernels because wmma_shape_profitable declines every native '
            'vision projection at this fixture geometry. Cooperative Mixed WMMA dispatch is not claimed.'),
        not_claimed=[] if cooperative else list(NOT_CLAIMED),
        separate_wmma_evidence=list(SEPARATE_WMMA),historical_pre_gate_receipts=historical)
    write(a.output/'qualification.json',dict(schema=1,status='PASS_BOUNDED_NATIVE_VISION',
        source_revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        source_sha256=sources,binary_sha256=binaries,commands=commands,reports=reports,checks=checks,
        qualification_scope=qualification_scope,
        limitations=['Native language LoRA and short vision operations remain FP32',
            'No native V4.1 automatic loss scaling or Full FP16 admission',
            'The wide joint image/language configuration has not yet been independently qualified',
            'Cooperative Mixed WMMA dispatch is not qualified by this fixture; the current-source '
            'implementation is qualified by the separate precision, wide-multimodal and primitive evidence']))
    print('NATIVE V4.1 NUMERICAL:',reports['bounded']['route']['measured_route'],
        reports['bounded']['route']['accumulation_dispatches'],'PASS_BOUNDED_NATIVE_VISION',flush=True)


if __name__=='__main__':main()
