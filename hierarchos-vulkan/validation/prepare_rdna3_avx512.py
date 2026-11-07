"""Freeze separate AVX-512 image oracles using the maintained CPU generators.

Torch 2.6 CPU scatter requires int64 indices; newer Transformers emits int32
Qwen4 selection indices. Widen indices without changing any floating-point op.
The adapter and its hash are retained alongside the ordinary source snapshot.
"""
import argparse
import importlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['shared', 'seq2seq', 'gemma3', 'gemma4', 'llava', 'deepseek_v41'], default='shared')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profiles', nargs='+')
    parser.add_argument('--cases', nargs='+')
    args = parser.parse_args()
    if args.suite != 'deepseek_v41':
        import torch
        if torch.backends.cpu.get_cpu_capability() != 'AVX512':
            raise RuntimeError('AVX-512 oracle requires native AVX512 CPU capability')
        original_scatter = torch.Tensor.scatter

        def scatter(self, dim, index, *values, **kwargs):
            if self.device.type == 'cpu' and index.dtype == torch.int32:
                index = index.to(torch.int64)
            return original_scatter(self, dim, index, *values, **kwargs)

        torch.Tensor.scatter = scatter
    module = importlib.import_module('qualify_' + {
        'shared': 'shared_image_pipeline', 'seq2seq': 'seq2seq_image_pipeline',
        'gemma3': 'gemma3_pipeline', 'gemma4': 'gemma4_pipeline',
        'llava': 'llava_pipeline', 'deepseek_v41': 'deepseek_v41_shared_image',
    }[args.suite])
    module.prepare(args)
    manifest_path = args.output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    dest = args.output / 'oracle-source' / Path(__file__).name
    dest.parent.mkdir(exist_ok=True)
    shutil.copyfile(__file__, dest)
    from qualify_multimodal_training import digest, write_json
    manifest['oracle']['source_sha256'][dest.relative_to(args.output).as_posix()] = digest(dest)
    manifest['oracle']['index_adapter'] = 'CPU int32 scatter indices widened to int64; floating-point graph unchanged'
    write_json(manifest_path, manifest)


if __name__ == '__main__':
    main()
