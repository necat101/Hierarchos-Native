"""Additional eligible-shape Mixed qualification, separate from frozen FP32 gates.

CPU inputs come from unmodified upstream models prepared without native output.
Every family must execute cooperative forward, dW and dX, update FP32 masters,
preserve reload identity and meet the existing Mixed max/RMS policy. Wide FP32
CPU diagnostics retain the original strict outcome without replacing 93 gates.

A geometry that misses the unchanged policy is recorded as FAIL and the remaining
families still run. This scope is not promoted to PASS when any case fails; the
original frozen FP32 gates remain separate evidence.
"""
import argparse, json, math, os, subprocess
from pathlib import Path
from safetensors.numpy import load_file
from qualify_precision_training import digest, metric, require, write, ROOT


def evaluate_case(family, profile, name, folder, out, binaries, commands):
    cpu = json.loads((folder / 'cpu.json').read_text())
    arms = {}
    checks = {}
    for mode in ('off', 'mixed'):
        label = family + '-' + name.replace('/', '-') + '-' + mode
        dest = out / label
        command = list(map(str, [binaries[family], '--model', folder / 'model',
                                 '--fixture', folder / 'input.json', '--output', dest]))
        env = {**os.environ, 'HIERARCHOS_WMMA_MODE': mode, 'HIERARCHOS_ATEN_VECTOR_WIDTH': '8'}
        run = subprocess.run(command, env=env, capture_output=True, text=True)
        (out / (label + '.log')).write_text(run.stdout + run.stderr)
        commands.append(dict(command=command, mode=mode, exit_code=run.returncode))
        require(run.returncode == 0, f'{label}: process failed: {(run.stdout + run.stderr)[-600:]}')
        actual = json.loads((dest / 'report.json').read_text())
        arms[mode] = actual
        precision = actual['precision']
        require(precision['requested'] == mode, 'wrong requested precision')
        accum = precision.get('accumulation_dispatches', [
            precision.get('mixed_fp32_accumulation', 0),
            precision.get('full_fp16_accumulation', 0)])
        phases = precision.get('phases', [precision.get('forward', 0),
                                          precision.get('weight_grad', 0),
                                          precision.get('input_grad', 0)])
        require(accum[1] == 0 and ((accum[0] > 0) == (mode == 'mixed')),
                'wrong accumulator or no physical acceleration')
        if mode == 'mixed':
            require(all(n > 0 for n in phases), 'missing cooperative forward/dW/dX')
        require(precision['master_weights'] == 'fp32' and precision['optimizer_state'] == 'fp32',
                'narrowed masters/state')
        require(len(actual['steps']) == 2, 'incomplete joint training')
        for i, step in enumerate(actual['steps']):
            require(step['training']['text']['step'] == i + 1
                    and math.isfinite(step['training']['text']['loss']),
                    'skipped/nonfinite normal update')
            reference = cpu['steps'][i]
            cpu_training = reference.get('training', reference)
            for key, values in step['forward'].items():
                if key not in reference['forward']:
                    continue
                expected = reference['forward'][key]
                if key == 'logits':
                    # Use the CPU fixture's existing observable-value contract,
                    # also used by its frozen FP32 qualifier. All values,
                    # including excluded padding, must be finite.
                    metric(expected, values, float('inf'))
                    visible = cpu['visible']
                    require(len(visible) == len(values), 'invalid CPU visibility')
                    expected = [x for x, keep in zip(expected, visible) if keep]
                    values = [x for x, keep in zip(values, visible) if keep]
                row = metric(expected, values, .02 if mode == 'mixed' else float('inf'),
                             .003 if mode == 'mixed' else None)
                if mode == 'off':
                    row.update(strict_limit=2e-7,
                               within_strict_limit=row['max_abs'] < 2e-7,
                               role='additional CPU diagnostic')
                checks[f'{mode}/step-{i + 1}/{key}-vs-CPU'] = row
            for key in ('pixel_gradient', 'patch_gradient', 'projected_gradient'):
                if key not in step['training'] or key not in cpu_training:
                    continue
                row = metric(cpu_training[key], step['training'][key],
                             .02 if mode == 'mixed' else float('inf'),
                             .003 if mode == 'mixed' else None)
                require(any(x != 0 for x in step['training'][key]), 'image gradient disappeared')
                if mode == 'off':
                    row.update(strict_limit=2e-7,
                               within_strict_limit=row['max_abs'] < 2e-7,
                               role='additional CPU diagnostic')
                checks[f'{mode}/step-{i + 1}/{key}-vs-CPU'] = row
        for key, values in actual['final'].items():
            if key not in actual['reload']:
                continue
            row = metric(values, actual['reload'][key], 2e-7)
            require(row['max_abs'] == 0, 'stale export/reload mirror')
            checks[mode + '/reload/' + key] = row
        shared = (folder / 'model/image').is_dir()

        def package_tensors(root):
            if shared:
                return {component + '.' + k: v
                        for component in ('text', 'image')
                        for k, v in load_file(str(root / component / 'model.safetensors')).items()}
            return load_file(str(root / 'model.safetensors'))

        initial = package_tensors(folder / 'model')
        trained = package_tensors(dest / 'step-2')
        require(set(initial) <= set(trained), 'export dropped a trained parameter')
        aliases = cpu.get('aliases', {})
        for key in set(trained) - set(initial):
            target = aliases.get(key)
            require(target in trained, 'unexpected exported parameter ' + key)
            require((trained[key] == trained[target]).all(), 'tied export mismatch ' + key)
        changed = [k for k in initial if (initial[k] != trained[k]).any()]
        require(changed, 'no master weights updated')
        require(all(str(v.dtype) == 'float32' for v in trained.values()),
                'master weights lost FP32')
        for k, v in load_file(str(folder / 'cpu-step-2.safetensors')).items():
            native = k if k in trained else cpu.get('aliases', {}).get(k, k)
            require(native in trained, 'missing independent CPU parameter ' + k)
            row = metric(v, trained[native], .02 if mode == 'mixed' else float('inf'),
                         .003 if mode == 'mixed' else None)
            if mode == 'off':
                row.update(strict_limit=2e-7,
                           within_strict_limit=row['max_abs'] < 2e-7,
                           role='additional CPU diagnostic')
            checks[mode + '/masters/' + k] = row
    return dict(family=family, profile=profile, case=name, status='PASS_SCOPED',
                precision={k: v['precision'] for k, v in arms.items()}, checks=checks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('llava', 'gemma3', 'gemma4', 'seq2seq'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--bins', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--families', nargs='+',
                   choices=('llava', 'gemma3', 'gemma4', 'seq2seq'),
                   help='families to execute (default: all four)')
    a = p.parse_args()
    requested = a.families or ('llava', 'gemma3', 'gemma4', 'seq2seq')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    fixtures = {}
    sources = {}
    for family in requested:
        root = getattr(a, family).resolve()
        manifest = json.loads((root / 'manifest.json').read_text())
        require(manifest['complete'], 'incomplete independent fixture')
        fixtures[family] = {f.relative_to(root).as_posix(): digest(f)
                            for f in root.rglob('*') if f.is_file()}
        for row in manifest['cases']:
            name = row.get('name', row.get('case'))
            require(name is not None, 'unnamed CPU case')
            cases.append((family, row.get('profile', family), name, root / name))
    for path in subprocess.check_output(
            ['git', '-C', str(ROOT), 'ls-files', 'hierarchos-vulkan/src',
             'hierarchos-vulkan/shaders'], text=True).splitlines():
        sources[path] = digest(ROOT / path)
    sources[Path(__file__).relative_to(ROOT).as_posix()] = digest(__file__)
    binaries = {family: a.bins / ({'seq2seq': 'image_text_pipeline_probe',
                                   'llava': 'llava_pipeline_probe',
                                   'gemma3': 'gemma3_vision_probe',
                                   'gemma4': 'gemma4_vision_probe'}[family] + '.exe')
                for family in fixtures}
    binary_hashes = {k: digest(v) for k, v in binaries.items()}
    results = []
    commands = []
    failures = []
    for family, profile, name, folder in cases:
        try:
            result = evaluate_case(family, profile, name, folder, out, binaries, commands)
        except Exception as error:
            result = dict(family=family, profile=profile, case=name, status='FAIL',
                          error=str(error))
            failures.append(result)
            print('FAIL', family, name, str(error), flush=True)
        results.append(result)
        if result['status'] == 'PASS_SCOPED':
            print('PASS_SCOPED', family, name, flush=True)
    require(all(digest(v) == binary_hashes[k] for k, v in binaries.items()),
            'binary changed during qualification')
    require(all(digest(ROOT / p) == h for p, h in sources.items()),
            'source changed during qualification')
    require(all(digest(getattr(a, family) / p) == h
                for family, files in fixtures.items() for p, h in files.items()),
            'independent CPU fixture changed')
    status = 'PASS_SCOPED' if not failures else 'FAIL'
    write(out / 'qualification.json', dict(
        schema=1, status=status, results=results, failures=failures, commands=commands,
        source_files=sources, binary_sha256=binary_hashes, fixture_sha256=fixtures,
        policy=dict(mixed_max_abs=.02, mixed_rms=.003, original_fp32_max_abs=2e-7, rtol=0),
        limitations=['Additional wide FP32 CPU diagnostics do not replace the original strict 93-image gate',
                     'FP32 suboperations remain explicit; this does not qualify Full FP16 or universal AMP scaling']))
    print(status, len(results), 'cases', len(failures), 'failed', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
