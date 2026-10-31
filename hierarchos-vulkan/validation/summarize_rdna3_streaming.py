"""Attach measured RDNA3 streaming/PEFT evidence without broadening its scope."""
import datetime,hashlib,json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'target/rdna3-streaming'
DURABLE=ROOT/'hierarchos-vulkan/validation/release_parity/deepseek_v41'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def binding(path):return dict(path=str(path.relative_to(ROOT)),sha256=sha(path))

def log_text(path):
    raw=path.read_bytes()
    return raw.decode('utf-16' if raw.startswith(b'\xff\xfe') else 'utf-8-sig')

def main():
    tests={}
    for name in ['v41-core-regression','v41-training-regression','v41-uninterrupted-trajectory',
                 'v41-40-layer-inference','generic-native-matrix-final','v41-application-residency',
                 'vulkan-regression','cli-regression','gui-regression']:
        path=RUN/(name+'.log');text=log_text(path)
        result=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored',text)
        assert len(result)==1 and int(result[0][1])==0 and int(result[0][0])>0,name
        tests[name]=dict(**binding(path),passed=int(result[0][0]),failed=0,ignored=int(result[0][2]))
    assert '"identical": 548' in log_text(RUN/'shader-provenance.log')
    assert 'PASS:' in log_text(RUN/'push-layout.log')
    stream_path=ROOT/'hierarchos-vulkan/validation/release_parity/parameter_streaming/qualification.json'
    stream=json.loads(stream_path.read_text());assert stream['status']=='qualified' and stream['profiles_qualified']==32
    trajectory_path=RUN/'v41-uninterrupted-trajectory.json'
    trajectory=json.loads(trajectory_path.read_text())
    assert trajectory['pass'] and not trajectory['pinned_step_inputs'] and not trajectory['failures']
    assert (trajectory['atol'],trajectory['rtol'],trajectory['comparison'])==(2e-7,0,'strict_less_than')
    values=sum(r['values'] for r in trajectory['records']);worst=max(r['max_abs'] for r in trajectory['records'])
    assert values==1536 and worst<2e-7
    base_path=ROOT/'hierarchos-vulkan/validation/fixtures/deepseek_v41_production_ratio_model.json'
    base=json.loads(base_path.read_text())['cases'][0]
    geometry={k:base['config']['text_config'][k] for k in ['num_hidden_layers','hidden_size','n_routed_experts','num_experts_per_tok','hc_mult']}
    assert tuple(geometry.values())==(6,32,384,6,4)
    oracle_path=base_path.with_name('deepseek_v41_production_ratio_training_v2.json')
    oracle=json.loads(oracle_path.read_text());assert oracle['base_fixture_sha256']==sha(base_path)
    trajectory.update(geometry=geometry,methods=['LoRA','rsLoRA'],optimizer_steps_per_method=3,
        values=values,max_abs=worst,base_fixture=binding(base_path),cpu_training_fixture=binding(oracle_path),
        cpu_reference={k:oracle[k] for k in ['torch_version','torch_git_version','source_sha256','contract']},
        log=binding(RUN/'v41-uninterrupted-trajectory.log'))
    measured=DURABLE/'uninterrupted-training.json';measured.write_text(json.dumps(trajectory,indent=2)+'\n')
    fresh=ROOT/'target/deepseek-v41-reference/scalable-release-native-logits.json'
    inference=json.loads(fresh.read_text())['cases'][0]
    assert inference['full_logits']==3216 and inference['max_abs_drift']<2e-7
    cli_path=RUN/'cli-lifecycle-v2/qualification.json';cli=json.loads(cli_path.read_text())
    assert cli['pass_'] and cli['cli_sha256']==sha(ROOT/'hierarchos-native-cli/target/debug/hierarchos-native-cli.exe')
    assert len(cli['records'])==6
    for method in ['lora','rslora']:
        rows=[r for r in cli['records'] if r.get('method')==method]
        assert len(rows)==2 and {r['stream_parameters'] for r in rows}=={False,True}
        assert len({r['adapter_sha256'] for r in rows})==1
    source_names=['hierarchos-vulkan/src/transformer.rs','hierarchos-vulkan/src/transformer/peft_state.rs',
        'hierarchos-vulkan/src/deepseek_v41/model.rs','hierarchos-vulkan/src/deepseek_v41/model_training_tests.rs',
        'hierarchos-native-cli/src/lib.rs','hierarchos-native-cli/src/deepseek_v41_training.rs','hierarchos-gui/src/main.rs']
    payload=dict(schema=1,status='qualified_for_recorded_scope',recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        comparison='strict_less_than',absolute_logit_limit=2e-7,rtol=0,cuda_required=False,production_weights_executed=False,
        device=inference['device'],
        v41_uninterrupted_training=dict(**binding(measured),values=values,max_abs=worst,pinned_step_inputs=False,geometry=geometry),
        v41_released_schedule_inference=dict(**binding(fresh),layers=40,experts=384,topk=6,logits=3216,max_abs=inference['max_abs_drift']),
        other_architecture_streaming=dict(**binding(stream_path),profiles=32,max_abs=stream['max_abs']),
        application_policy=dict(default='resident',cli_flag='--stream-parameters',gui_default=False,
            gui_label='Stream frozen parameters and MoE experts from disk',
            supported_workflows=['Transformer inference','frozen-base LoRA/rsLoRA training where qualified'],
            library_compatibility='historical V4.1 package constructors retain their tiered policy; application routes pass residency explicitly'),
        cli_lifecycle=dict(**binding(cli_path),resumed_uninterrupted_and_resident_streamed_adapters_byte_identical=True,
            v41_methods=['LoRA','rsLoRA'],generic_training_inference=['llama','mixtral']),
        checks=tests,shader_provenance=dict(**binding(RUN/'shader-provenance.log'),identical_artifacts=548),
        push_layout=binding(RUN/'push-layout.log'),depth_metadata_correction=binding(RUN/'training-depth-metadata-correction.json'),
        limits=stream['limits']+['scaled fixtures qualify the recorded shapes/features; full production weights were not run',
            'V4.1 unsupported adapter classes remain fail-closed under the existing capability contract'],
        source_bindings={name:sha(ROOT/name) for name in source_names})
    if (RUN/'scratch-cleanup.json').exists():payload['scratch_cleanup']=binding(RUN/'scratch-cleanup.json')
    output=DURABLE/'rdna3-streaming-qualification.json';output.write_text(json.dumps(payload,indent=2)+'\n')
    parent=DURABLE/'qualification.json';existing=json.loads(parent.read_text())
    existing['rdna3_streaming_followthrough']=binding(output)
    existing['training_qualification']['uninterrupted_trajectory']=binding(measured)
    parent.write_text(json.dumps(existing,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k in ['status','v41_uninterrupted_training','v41_released_schedule_inference','other_architecture_streaming']},indent=2))

if __name__=='__main__':main()
