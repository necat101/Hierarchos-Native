"""Additional eligible-shape Mixed qualification, separate from frozen FP32 gates.

CPU inputs come from unmodified upstream models prepared without native output.
Every family must execute cooperative forward, dW and dX, update FP32 masters,
preserve reload identity and meet the existing Mixed max/RMS policy. Wide FP32
CPU diagnostics retain the original strict outcome without replacing 93 gates.

A geometry that misses the unchanged policy is recorded as FAIL and the remaining
families still run. This scope is not promoted to PASS when any case fails; the
original frozen FP32 gates remain separate evidence.

Every case also records the execution route its own counters observed. At the
released CPU-scale geometries the unchanged WMMA profitability policy declines
the cooperative projections, so the `mixed` request executes the FP32 kernels and
the case is described as `mixed_configured_non_cooperative_route`. That is a
numerical qualification of the route that ran; it is not an independent
current-source qualification of cooperative Mixed WMMA dispatch, and
`qualification_scope` says so in machine-checkable form. The sealed pre-gate
receipt whose wide rows did dispatch cooperatively is disclosed from its own
release root and is never relabelled as current-source evidence.
"""
import argparse, hashlib, json, math, os, subprocess
from pathlib import Path
from safetensors.numpy import load_file
from qualify_precision_training import digest, metric, require, write, ROOT


POLICY = dict(wmma_min_profitable_macs=16_000_000, full_fp16_max_reduction=64)
CLAIM = ('wide multimodal numerical behaviour of all nine model-family fixtures '
         'on the measured execution route')
NOT_CLAIMED = ['current-source cooperative Mixed WMMA accumulator dispatch']
SEPARATE_EVIDENCE = ['precision', 'wmma_mixed', 'wmma_full_original', 'wmma_full_dot2']
FALLBACK_KEYS = ('cooperative_dispatches', 'shape_fallbacks', 'policy_fallbacks',
                 'full_fp16_reduction_fallbacks')
ROUTE_COOPERATIVE = 'mixed_cooperative_route'
ROUTE_NON_COOPERATIVE = 'mixed_configured_non_cooperative_route'
SCOPE_COOPERATIVE = 'wide_cooperative_mixed_numerical'
SCOPE_NON_COOPERATIVE = 'wide_numerical_non_cooperative_route'


def dispatch_counts(precision):
    return list(precision.get('accumulation_dispatches', [
        precision.get('mixed_fp32_accumulation', 0),
        precision.get('full_fp16_accumulation', 0)]))


def phase_inventory(precision):
    return list(precision.get('phases', [precision.get('forward', 0),
                                         precision.get('weight_grad', 0),
                                         precision.get('input_grad', 0)]))


def measured_route(precision):
    """Describe the route this arm's own counters observed.

    A `mixed` request that dispatched nothing is a configured non-cooperative
    route: the kernels that ran were the FP32 ones. It is never reported as
    cooperative WMMA execution merely because Mixed was selected.
    """
    counts = dispatch_counts(precision)
    phase_counts = phase_inventory(precision)
    require(sum(phase_counts) == sum(counts),
            'phase inventory disagrees with the dispatch counters')
    cooperative = counts[0] > 0
    if not cooperative:
        require(counts == [0, 0] and phase_counts == [0, 0, 0],
                'non-cooperative route recorded low-precision accumulation')
        require(precision.get('cooperative_dispatches', 0) == 0,
                'non-cooperative route recorded cooperative dispatches')
    return dict(configured_precision=precision['requested'],
                accumulation_dispatches=list(counts), phases=list(phase_counts),
                cooperative_mixed_wmma_dispatched=cooperative,
                full_fp16_wmma_dispatched=counts[1] > 0,
                observed_route=ROUTE_COOPERATIVE if cooperative else ROUTE_NON_COOPERATIVE,
                fallback_telemetry={key: precision[key] for key in FALLBACK_KEYS
                                    if key in precision},
                scope=SCOPE_COOPERATIVE if cooperative else SCOPE_NON_COOPERATIVE)


def case_provenance(files, name):
    """Bind one case's fixture inputs and pinned CPU oracle inside the family map."""
    prefix = name.rstrip('/') + '/'
    rows = {path: sha for path, sha in files.items() if path.startswith(prefix)}
    require(rows, name + ': fixture case has no bound files')
    inputs = {path: sha for path, sha in rows.items() if path.endswith('input.json')}
    require(inputs, name + ': fixture input is missing')
    oracle = {path: sha for path, sha in rows.items()
              if path.endswith('/cpu.json') or '/cpu-step-' in path}
    require(oracle, name + ': pinned CPU oracle is missing')
    return dict(fixtures=inputs, oracle=oracle)


def historical_routes(path, source_revision):
    """Disclose a sealed pre-gate wide component without reusing its route."""
    release = json.loads(Path(path).read_text())
    binding = release['components']['wide']
    component = Path(path).parent / binding['path']
    require(digest(component) == binding['sha256'],
            'historical wide component digest changed')
    record = json.loads(component.read_text())
    routes = []
    for result in record['results']:
        precision = result['precision']['mixed']
        counts = dispatch_counts(precision)
        require(counts[0] > 0, 'historical disclosure is not a cooperative Mixed receipt')
        routes.append(dict(family=result['family'], case=result['case'],
                           accumulation_dispatches=counts,
                           phases=phase_inventory(precision)))
    require(release['source_revision'] != source_revision,
            'the disclosed pre-gate receipt is the current source')
    return dict(release_root=Path(path).parent.name,
                release_source_revision=release['source_revision'],
                component_path=binding['path'], component_sha256=binding['sha256'],
                routes=routes,
                note='pre-gate receipt from before the WMMA profitability gate; '
                     'not current-source dispatch evidence')


def evaluate_case(family, profile, name, folder, out, binaries, commands, provenance):
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
        accum = dispatch_counts(precision)
        phase_counts = phase_inventory(precision)
        require(sum(phase_counts) == sum(accum),
                'phase inventory disagrees with the dispatch counters')
        below_threshold = accum == [0, 0] and all(n == 0 for n in phase_counts)
        if mode != 'off' and below_threshold:
            # This fixture stays below WMMA_MIN_PROFITABLE_MACS, so the profitability
            # gate must decline cooperative execution and keep FP32 masters/state.
            require(precision.get('cooperative_dispatches', 0) == 0,
                    'below-threshold fixture dispatched WMMA')
        else:
            require(accum[1] == 0 and ((accum[0] > 0) == (mode == 'mixed')),
                    'wrong accumulator or no physical acceleration')
            if mode == 'mixed':
                require(all(n > 0 for n in phase_counts), 'missing cooperative forward/dW/dX')
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
                precision={k: v['precision'] for k, v in arms.items()}, checks=checks,
                provenance=provenance, route=measured_route(arms['mixed']['precision']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('llava', 'gemma3', 'gemma4', 'seq2seq'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--bins', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--families', nargs='+',
                   choices=('llava', 'gemma3', 'gemma4', 'seq2seq'),
                   help='families to execute (default: all four)')
    p.add_argument('--historical-release', type=Path, required=True,
                   help='sealed pre-gate release-qualification.json whose wide rows are disclosed, not reused')
    a = p.parse_args()
    requested = a.families or ('llava', 'gemma3', 'gemma4', 'seq2seq')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    fixtures = {}
    oracles = {}
    sources = {}
    source_revision = subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    for family in requested:
        root = getattr(a, family).resolve()
        manifest = json.loads((root / 'manifest.json').read_text())
        require(manifest['complete'], 'incomplete independent fixture')
        fixtures[family] = {f.relative_to(root).as_posix(): digest(f)
                            for f in root.rglob('*') if f.is_file()}
        oracle = manifest['oracle']
        require(oracle.get('cpu') and oracle.get('threads') == 1,
                family + ': unpinned CPU oracle')
        oracles[family] = dict(cpu=oracle['cpu'], threads=oracle['threads'],
                               torch=oracle.get('torch'),
                               transformers_revision=oracle.get('transformers_revision'),
                               manifest_sha256=fixtures[family]['manifest.json'],
                               oracle_sha256=hashlib.sha256(json.dumps(
                                   oracle, sort_keys=True).encode()).hexdigest())
        for row in manifest['cases']:
            name = row.get('name', row.get('case'))
            require(name is not None, 'unnamed CPU case')
            cases.append((family, row.get('profile', family), name, root / name,
                          case_provenance(fixtures[family], name)))
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
    for family, profile, name, folder, provenance in cases:
        try:
            result = evaluate_case(family, profile, name, folder, out, binaries, commands,
                                   provenance)
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
    routes = [r['route'] for r in results if r['status'] == 'PASS_SCOPED']
    require(len(routes) == len(results), 'a failing case has no measured route')
    flags = {r['cooperative_mixed_wmma_dispatched'] for r in routes}
    labels = {r['observed_route'] for r in routes}
    kinds = {r['scope'] for r in routes}
    require(len(flags) == len(labels) == len(kinds) == 1,
            'cases measured different execution routes')
    cooperative = flags.pop()
    scope = dict(kind=kinds.pop(), claim=CLAIM, measured_route=labels.pop(),
                 cooperative_mixed_executed=cooperative, policy=POLICY,
                 not_claimed=[] if cooperative else NOT_CLAIMED,
                 separate_wmma_evidence=SEPARATE_EVIDENCE,
                 historical_pre_gate_receipts=[historical_routes(
                     a.historical_release.resolve(), source_revision)])
    status = 'PASS_SCOPED' if not failures else 'FAIL'
    write(out / 'qualification.json', dict(
        schema=1, status=status, source_revision=source_revision,
        results=results, failures=failures, commands=commands,
        source_files=sources, binary_sha256=binary_hashes, fixture_sha256=fixtures,
        oracle_contract=oracles, qualification_scope=scope,
        policy=dict(mixed_max_abs=.02, mixed_rms=.003, original_fp32_max_abs=2e-7, rtol=0),
        limitations=['Additional wide FP32 CPU diagnostics do not replace the original strict 93-image gate',
                     'FP32 suboperations remain explicit; this does not qualify Full FP16 or universal AMP scaling',
                     'All nine fixtures are numerical qualifications of the route their counters measured; '
                     'at the released CPU-scale geometries the unchanged profitability policy dispatches no '
                     'cooperative accumulation, so this component is not an independent current-source '
                     'cooperative Mixed WMMA dispatch qualification',
                     'The pre-gate Phase 25 wide rows are disclosed from their own sealed release root and '
                     'are not current-source dispatch evidence']))
    print(status, len(results), 'cases', len(failures), 'failed', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
