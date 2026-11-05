"""Freeze joint encoder/decoder image training on unmodified CPU models.

This is new evidence for the three encoder/decoder inventory profiles. It never
replaces the frozen causal-model matrix, and reuses its strict replay policy.
"""
from pathlib import Path
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys

import qualify_shared_image_pipeline as shared
from qualify_multimodal_training import digest, write_json

VALIDATION = Path(__file__).resolve().parent
PROFILES = {'bart', 't5', 'switch_transformers'}


def prepare(args):
    import torch
    import numpy as np
    import PIL
    from PIL import Image
    from safetensors.torch import save_file
    import verify_hf_logits as reference
    from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor

    torch.set_num_threads(1)
    torch.manual_seed(0x53455132)
    selected = set(args.profiles or PROFILES)
    if not selected or selected - PROFILES:
        raise ValueError('expected BART, T5 or Switch profiles')
    required = [p['profile'] for p in json.loads(shared.INVENTORY.read_text())['profiles']]
    models = [(name, model) for name, model, _ in reference.tiny_models(training_reference=True) if name in selected]
    if {name for name, _ in models} != selected:
        raise ValueError('missing upstream encoder/decoder model')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        'schema': 1, 'complete': False, 'scope': 'shared_clip_full_parameter_encoder_decoder_image_training',
        'absolute_limit': shared.LIMIT, 'rtol': 0, 'comparison': 'strict_less_than',
        'cases': [], 'required_green_profiles': required, 'requested_profiles': sorted(selected),
        'oracle': {'cpu': torch.backends.cpu.get_cpu_capability(), 'torch': torch.__version__,
                   'torch_revision': torch.version.git_version, 'numpy': np.__version__,
                   'pillow': PIL.__version__, 'threads': 1,
                   'transformers_revision': subprocess.check_output(
                       ['git', '-C', str(reference.TRANSFORMERS_SOURCE.parent), 'rev-parse', 'HEAD'], text=True).strip()},
    }
    for name, base in models:
        if not base.config.is_encoder_decoder:
            raise ValueError('joint oracle requires a real encoder/decoder model')
        for padded in [False, True]:
            case = name + '/' + ('padded' if padded else 'unmasked')
            folder = out / case
            package = folder / 'model'
            package.mkdir(parents=True)
            text = copy.deepcopy(base).float().train()
            for module in text.modules():
                for field in ['jitter_noise', 'moe_jitter_eps']:
                    if hasattr(module, field):
                        setattr(module, field, 0.)
                if isinstance(module, torch.nn.Dropout) and module.p:
                    raise ValueError('dropout requires a shared RNG oracle')
            if hasattr(text.config, 'router_jitter_noise'):
                text.config.router_jitter_noise = 0.
            reference.save_reference_checkpoint(name, text, package / 'text')
            encoder = text.get_encoder()
            embedding = encoder.get_input_embeddings()
            vocab, width = embedding.weight.shape
            vision_config = CLIPVisionConfig(
                hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                image_size=8, patch_size=4, num_channels=3, hidden_act='quick_gelu', attention_dropout=0.)
            torch.manual_seed(0x434C4950 + sum(name.encode()))
            tower = CLIPVisionModel(vision_config).float().train()
            tower.set_attn_implementation('eager')
            projector = torch.nn.Sequential(torch.nn.Linear(16, width), torch.nn.GELU(), torch.nn.Linear(width, width))
            with torch.no_grad():
                for parameter in projector.parameters():
                    parameter.mul_(.15)
            image_parameters = {'vision_tower.vision_model.' + k: v for k, v in tower.named_parameters()}
            for index, label in [(0, 'linear_1'), (2, 'linear_2')]:
                image_parameters.update({'multi_modal_projector.' + label + '.' + k: v for k, v in projector[index].named_parameters()})
            image = package / 'image'
            image.mkdir()
            save_file({k: v.detach().contiguous().clone() for k, v in image_parameters.items()}, str(image / 'model.safetensors'))
            token = vocab - 1
            write_json(package / 'config.json', {'model_type': 'hierarchos_image_text', 'schema': 1,
                       'text_profile': name, 'text_mode': 'encoder_decoder'})
            write_json(image / 'config.json', {'model_type': 'hierarchos_clip_adapter',
                       'vision_config': vision_config.to_dict(), 'vision_feature_layer': -1,
                       'vision_feature_select_strategy': 'default', 'projector_hidden_act': 'gelu',
                       'multimodal_projector_bias': True, 'image_token_index': token})
            processor = CLIPImageProcessor(do_resize=False, do_center_crop=False)
            processor.save_pretrained(image)
            rgb = np.random.default_rng(0x494D47 + sum(name.encode())).integers(0, 256, (2, 8, 8, 3), dtype=np.uint8)
            alternate_rgb = 255 - rgb
            pixels = processor(images=[Image.fromarray(x) for x in rgb], return_tensors='pt')['pixel_values'].requires_grad_()
            alternate = processor(images=[Image.fromarray(x) for x in alternate_rgb], return_tensors='pt')['pixel_values']
            ids = torch.tensor([[1,token,token,token,token,7,3,2], [2,token,token,token,token,3,7,1]])
            decoder_ids = torch.tensor([[0,4,5,6,7,3,2,1], [0,6,3,5,4,7,2,1]])
            mask = torch.ones_like(ids)
            decoder_mask = torch.ones_like(decoder_ids)
            if padded:
                mask[0,-1] = 0
                mask[1,0] = 0
                decoder_mask[0,-1] = 0
                decoder_mask[1,0] = 0
            positions = (ids == token).flatten().nonzero().flatten()
            targets = decoder_ids.roll(-1, dims=1)
            weights = decoder_mask.float()
            weights[:,-1] = 0
            weights[:,:-1] *= decoder_mask[:,1:]
            targets[weights == 0] = -100
            fixture = {'batch_size': 2, 'seq_len': 8, 'image_count': 2,
                       'input_ids': ids.flatten().tolist(), 'attention_mask': mask.float().flatten().tolist(),
                       'decoder_input_ids': decoder_ids.flatten().tolist(),
                       'decoder_attention_mask': decoder_mask.float().flatten().tolist(),
                       'targets': [x if x >= 0 else 0xffffffff for x in targets.flatten().tolist()],
                       'loss_weights': weights.flatten().tolist(), 'rgb': rgb.flatten().tolist(),
                       'alternate_rgb': alternate_rgb.flatten().tolist(), 'steps': 2,
                       'lr': .001, 'beta1': .9, 'beta2': .999, 'eps': .001, 'weight_decay': 0.}
            write_json(folder / 'input.json', fixture)
            holder = {'encoder_active': False}
            def enter_encoder(module, inputs):
                holder['encoder_active'] = True
            def leave_encoder(module, inputs, output):
                holder['encoder_active'] = False
            def inject(module, inputs, output):
                if not holder['encoder_active']:
                    return output
                replacement = holder['projected'] * getattr(module, 'embed_scale', 1.)
                return output.reshape(-1, width).index_copy(0, positions, replacement).reshape_as(output)
            handles = [encoder.register_forward_pre_hook(enter_encoder),
                       encoder.register_forward_hook(leave_encoder), embedding.register_forward_hook(inject)]
            def forward(values):
                features = tower(values, output_hidden_states=True).hidden_states[-1][:,1:,:].reshape(-1,16)
                projected = projector(features)
                holder['projected'] = projected
                result = text(input_ids=ids, attention_mask=mask, decoder_input_ids=decoder_ids,
                              decoder_attention_mask=decoder_mask, use_cache=False)
                return {'logits': result.logits, 'vision_features': features, 'projected_features': projected}
            def record(result):
                return {k: v.detach().flatten().tolist() for k, v in result.items()}
            with torch.no_grad():
                alternate_result = record(forward(alternate))
            optimizer = torch.optim.AdamW([*text.parameters(), *tower.parameters(), *projector.parameters()],
                                         lr=.001, betas=(.9,.999), eps=.001, weight_decay=0., foreach=False, fused=False)
            steps = []
            for step in [1,2]:
                optimizer.zero_grad(set_to_none=True)
                pixels.grad = None
                result = forward(pixels)
                holder['projected'].retain_grad()
                row_loss = torch.nn.functional.cross_entropy(result['logits'].reshape(-1,vocab), targets.flatten(), reduction='none').reshape_as(weights)
                loss = (row_loss * weights).sum() / weights.sum()
                loss.backward()
                if pixels.grad is None or not pixels.grad.isfinite().all() or pixels.grad.abs().max() == 0:
                    raise ValueError('joint CPU loss did not reach image pixels')
                steps.append({'forward': record(result), 'loss': loss.item(),
                              'pixel_gradient': pixels.grad.flatten().tolist(),
                              'projected_gradient': holder['projected'].grad.flatten().tolist()})
                optimizer.step()
                parameters = {'text.'+k: v for k,v in text.named_parameters()}
                parameters.update({'image.'+k: v for k,v in image_parameters.items()})
                save_file({k: v.detach().contiguous().clone() for k,v in parameters.items()}, str(folder/f'cpu-step-{step}.safetensors'))
            with torch.no_grad():
                final = record(forward(pixels))
            for handle in handles:
                handle.remove()
            visible = decoder_mask.bool().reshape(-1,1).expand(-1,vocab).flatten().tolist()
            write_json(folder/'cpu.json', {'pixels': pixels.detach().flatten().tolist(),
                       'alternate': alternate_result, 'steps': steps, 'final': final,
                       'visible': visible, 'text_visible': visible,
                       'aliases': {'text.'+k: 'text.'+v for k,v in getattr(text,'all_tied_weights_keys',{}).items()}})
            manifest['cases'].append({'profile':name, 'name':case,
                                     'files':{p.relative_to(out).as_posix(): digest(p) for p in folder.rglob('*') if p.is_file()}})
            write_json(out/'manifest.json', manifest)
            print('CPU', case, 'joint encoder/decoder image training frozen', flush=True)
    source = out/'oracle-source'
    source.mkdir()
    for name in [Path(__file__).name, 'qualify_shared_image_pipeline.py', 'qualify_multimodal_training.py',
                 'verify_hf_logits.py', 'kimi_k3_reference.py']:
        shutil.copyfile(VALIDATION/name, source/name)
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(reference.TRANSFORMERS_SOURCE):
                target = source/'transformers'/path.relative_to(reference.TRANSFORMERS_SOURCE)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    manifest['oracle']['source_sha256'] = {p.relative_to(out).as_posix(): digest(p) for p in source.rglob('*') if p.is_file()}
    manifest['complete'] = True
    write_json(out/'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profiles', nargs='+')
    p = sub.add_parser('run')
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profiles', nargs='+')
    p.add_argument('--target', choices=['gen9','rdna3'], required=True)
    p.add_argument('--device-id', type=lambda x:int(x,0))
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else shared.run)(args)


if __name__ == '__main__':
    main()
