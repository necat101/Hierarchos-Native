"""Measure source-first loading on synthetic dense/MoE checkpoints.

This gate compares the new loader with the resident loader, including storage
formats and sharding. Independent CPU/PEFT parity is a separate, unchanged gate
in qualify_parameter_streaming.py. No production checkpoint is downloaded.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 2e-7
TILE = 8 * 1024 * 1024


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_model(path, family, width, experts, intermediate=None):
    """Fixed-seed F32 measurement package, with a packed expert tensor > tile."""
    path.mkdir()
    inter = intermediate or 2 * width
    heads = 8 if width >= 1024 else 4
    cfg = dict(model_type=family, vocab_size=64, hidden_size=width,
               intermediate_size=inter, num_hidden_layers=1,
               num_attention_heads=heads, num_key_value_heads=heads // 2,
               max_position_embeddings=8, hidden_act='silu',
               rms_norm_eps=1e-6, rope_theta=10000.0,
               attention_dropout=0.0, tie_word_embeddings=False)
    if experts:
        cfg.update(num_local_experts=experts, num_experts_per_tok=2,
                   router_aux_loss_coef=0.001, sliding_window=4)
    (path / 'config.json').write_text(json.dumps(cfg))
    rng = torch.Generator().manual_seed(20261030)
    def rand(*shape):
        return torch.randn(shape, generator=rng) * 0.02
    state = {'model.embed_tokens.weight': rand(64, width),
             'lm_head.weight': rand(64, width),
             'model.norm.weight': torch.ones(width)}
    p = 'model.layers.0'
    state[p + '.input_layernorm.weight'] = torch.ones(width)
    state[p + '.post_attention_layernorm.weight'] = torch.ones(width)
    for name, out in [('q_proj', width), ('k_proj', width // 2),
                      ('v_proj', width // 2), ('o_proj', width)]:
        state[p + '.self_attn.' + name + '.weight'] = rand(out, width)
    if experts:
        state[p + '.mlp.gate.weight'] = rand(experts, width)
        state[p + '.mlp.experts.gate_up_proj'] = rand(experts, 2 * inter, width)
        state[p + '.mlp.experts.down_proj'] = rand(experts, width, inter)
    else:
        for name, shape in [('gate_proj', (inter, width)),
                            ('up_proj', (inter, width)), ('down_proj', (width, inter))]:
            state[p + '.mlp.' + name + '.weight'] = rand(*shape)
    save_file(state, path / 'model.safetensors')
    return sum(v.numel() for v in state.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = args.output or ROOT / 'target/initial-streaming' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    binary = out / 'transformer-logits.exe'
    shutil.copy2(ROOT / 'hierarchos-vulkan/target/debug/hierarchos-vulkan-transformer-logits.exe', binary)
    fixture = out / 'fixture.json'
    fixture.write_text(json.dumps(dict(batch_size=1, seq_len=8,
        input_ids=[1, 5, 9, 3, 7, 11, 13, 15], attention_mask=[1.0] * 8)))
    rows = []
    rejections = []

    def compare(package, label, elements=None, reference=None):
        reports = []
        for streamed in [False, True]:
            report = out / (label + ('-streamed' if streamed else '-resident') + '.json')
            model = package if streamed or reference is None else reference
            command = [str(binary), '--model', str(model), '--fixture', str(fixture), '--output', str(report)]
            if streamed:
                command.append('--stream-parameters')
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            (report.with_suffix('.log')).write_text(result.stdout + result.stderr)
            assert result.returncode == 0, (label, streamed, result.stderr[-3000:])
            reports.append(json.loads(report.read_text()))
        resident, streamed = reports
        assert streamed['source_first_parameters'] and streamed['streamed_linears'] > 0
        assert streamed['retired_parameter_bytes'] == 0
        drift = max(abs(a - b) for a, b in zip(resident['logits'], streamed['logits'], strict=True))
        assert drift < LIMIT, (label, drift)
        assert resident['logits'] == streamed['logits'], label
        io = streamed['loading_linear_package_io']
        assert len(io) == 1, 'body shards must share one bounded cache'
        assert io and all(r['max_f32_request_bytes'] <= TILE and r['max_payload_read_bytes'] <= TILE for _, r in io)
        assert all(r['cache_peak_bytes'] <= 4 * TILE for _, r in io)
        assert streamed['loading_peak_buffer_bytes'] < resident['loading_peak_buffer_bytes']
        row = dict(label=label, package=str(package), logical_elements=elements,
            resident_loading_peak=resident['loading_peak_buffer_bytes'],
            streamed_loading_peak=streamed['loading_peak_buffer_bytes'],
            streamed_largest_buffer=streamed['largest_loading_buffer_bytes'],
            streamed_linears=streamed['streamed_linears'], loading_linear_package_io=io,
            logit_count=len(streamed['logits']), max_abs=drift,
            resident_report_sha256=sha(out / (label + '-resident.json')),
            streamed_report_sha256=sha(out / (label + '-streamed.json')))
        row['source_files'] = {str(p.relative_to(package)): sha(p) for p in sorted(package.iterdir()) if p.is_file()}
        rows.append(row)
        print(f"PASS {label}: resident peak={row['resident_loading_peak']}, source-first={row['streamed_loading_peak']}, drift={drift}", flush=True)

    for family, width, experts in [('llama', 256, 0), ('llama', 512, 0),
                                    ('mixtral', 512, 2), ('mixtral', 512, 8)]:
        label = f'{family}-h{width}-e{experts}'
        package = out / label
        elements = write_model(package, family, width, experts)
        compare(package, label, elements)
    # At fixed width, increasing expert count grows the checkpoint but only
    # the small routing tape may grow during construction, never expert weights.
    small, large = rows[2:4]
    assert large['logical_elements'] > 3 * small['logical_elements']
    assert large['streamed_loading_peak'] - small['streamed_loading_peak'] < 256 * 1024
    assert large['streamed_largest_buffer'] < TILE
    # One individual dense projection also exceeds the read tile, independently
    # of the packed-expert boundary exercised above.
    package = out / 'llama-h1024-i3072'
    elements = write_model(package, 'llama', 1024, 0, intermediate=3072)
    compare(package, package.name, elements)
    assert max(r['max_f32_request_bytes'] for _, r in rows[-1]['loading_linear_package_io']) == TILE

    previous = json.loads((ROOT / 'hierarchos-vulkan/validation/release_parity/parameter_streaming/qualification.json').read_text())
    for family in ['llama', 'mistral', 'mixtral', 'qwen2_5_gqa', 'qwen2_5_sliding_tied']:
        source = Path(previous['profiles']['llama' if family == 'mistral' else family]['report_path']).parent / 'base'
        config = json.loads((source / 'config.json').read_text())
        if family == 'mistral':
            config.update(model_type='mistral', sliding_window=4)
        original = load_file(source / 'model.safetensors')
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            label = family + '-' + str(dtype).split('.')[-1] + '-sharded'
            package = out / label
            package.mkdir()
            (package / 'config.json').write_text(json.dumps(config))
            # The resident Mixtral auxiliary loader requires its canonical
            # prefix. Dense families also exercise wrapped backbone aliases.
            state = {k.replace('model.', 'model.language_model.', 1) if k.startswith('model.') and family != 'mixtral' else k:
                     v.to(dtype).contiguous() for k, v in original.items()}
            names = sorted(state)
            weight_map = {}
            for index in range(2):
                shard = f'model-0000{index + 1}-of-00002.safetensors'
                subset = {k: state[k] for k in names[index::2]}
                save_file(subset, package / shard)
                weight_map.update({k: shard for k in subset})
            (package / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}))
            compare(package, label)
    # Both ordinary per-expert naming conventions, checked against the same
    # unmodified packed resident checkpoint rather than reinitializing weights.
    source = Path(previous['profiles']['mixtral']['report_path']).parent / 'base'
    config = json.loads((source / 'config.json').read_text())
    original = load_file(source / 'model.safetensors')
    for legacy in [False, True]:
        label = 'mixtral-' + ('legacy' if legacy else 'ordinary') + '-experts'
        package = out / label
        package.mkdir()
        (package / 'config.json').write_text(json.dumps(config))
        state = {k: v for k, v in original.items() if '.experts.' not in k}
        for layer in range(config['num_hidden_layers']):
            p = f'model.layers.{layer}'
            packed = original[p + '.mlp.experts.gate_up_proj']
            down = original[p + '.mlp.experts.down_proj']
            for e in range(packed.shape[0]):
                gate, up = packed[e].chunk(2, dim=0)
                prefix = p + (f'.block_sparse_moe.experts.{e}' if legacy else f'.mlp.experts.{e}')
                names = ('w1', 'w3', 'w2') if legacy else ('gate_proj', 'up_proj', 'down_proj')
                for name, value in zip(names, [gate, up, down[e]], strict=True):
                    state[prefix + '.' + name + '.weight'] = value.clone().contiguous()
        save_file(state, package / 'model.safetensors')
        compare(package, label, reference=source)
    package = out / 'mixtral-wrapped'
    package.mkdir()
    (package / 'config.json').write_text(json.dumps(config))
    save_file({k.replace('model.', 'model.language_model.', 1) if k.startswith('model.') else k:
               v for k, v in original.items()}, package / 'model.safetensors')
    compare(package, 'mixtral-wrapped', reference=source)
    # Validate the last expert during construction, independent of routing.
    for failure in ['nonfinite-last-expert', 'wrong-projection-shape', 'mismatched-tied-alias']:
        source = out / 'mixtral-h512-e8' if failure == 'nonfinite-last-expert' else Path(previous['profiles']['llama']['report_path']).parent / 'base'
        config = json.loads((source / 'config.json').read_text())
        state = load_file(source / 'model.safetensors')
        if failure == 'nonfinite-last-expert':
            key = next(k for k in state if k.endswith('.mlp.experts.down_proj'))
            state[key].view(-1)[-1] = float('nan')
        elif failure == 'wrong-projection-shape':
            key = next(k for k in state if k.endswith('.self_attn.q_proj.weight'))
            state[key] = state[key][:-1].contiguous()
        else:
            config['tie_word_embeddings'] = True
            state['lm_head.weight'] = state['model.embed_tokens.weight'].clone()
            state['lm_head.weight'][0, 0] += 1.0
        package = out / failure
        package.mkdir()
        (package / 'config.json').write_text(json.dumps(config))
        save_file(state, package / 'model.safetensors')
        result = subprocess.run([str(binary), '--model', str(package), '--fixture', str(fixture),
            '--stream-parameters'], cwd=ROOT, text=True, capture_output=True)
        (out / (failure + '.log')).write_text(result.stdout + result.stderr)
        assert result.returncode != 0, failure
        expected = {'nonfinite-last-expert': 'non-finite', 'wrong-projection-shape': 'expected',
                    'mismatched-tied-alias': 'differs from embedding'}[failure]
        assert expected in result.stderr, (failure, result.stderr)
        rejections.append(dict(case=failure, stderr=result.stderr))
    receipt = dict(status='passed', comparison='strict_less_than', atol=LIMIT, rtol=0,
        scope='source-first loader identity and construction memory; CPU/PEFT oracle is separate',
        binary_sha256=sha(binary), comparisons=sum(r['logit_count'] for r in rows),
        max_abs=max(r['max_abs'] for r in rows), records=rows, rejections=rejections,
        sources={name: sha(ROOT / 'hierarchos-vulkan' / name) for name in [
            'src/transformer.rs', 'src/transformer/streamed_load.rs', 'src/weight_offload.rs',
            'src/bin/transformer_logits.rs', 'validation/qualify_source_first_streaming.py']},
        torch_version=torch.__version__, fixture_sha256=sha(fixture),
        limits=['synthetic checkpoints; no full production weights',
                'activations, norms, biases and adapter state remain resident',
                'only ordinary llama/mistral/qwen2/mixtral package loaders are source-first'])
    (out / 'qualification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(out / 'qualification.json', flush=True)


if __name__ == '__main__':
    main()
