"""Re-derive balanced packed timings from every raw construction rotation.

Each role occupies each physical allocation position once. Equal-length raw
traces are pooled, retaining the requested total observations per leg. No run,
sample or rotation is selected according to which route wins.
"""
import copy
import math
import statistics

PROTOCOL = 'allocation-balanced-shipped-v2'
BACKWARD_PROTOCOL = 'allocation-balanced-fusion-v1'
TENSORS = {'outputs', 'final_packed_state', 'grad_x', 'grad_initial_packed_state', 'token_feature_grad'}
METADATA = ('device', 'width', 'head_size', 'batch', 'steps', 'chain',
            'parameter_storage', 'precision_policy', 'default_route', 'recorded_route')
SCHEDULES = ('staged_schedule', 'fused_schedule', 'folded_schedule', 'default_schedule', 'available_schedules')


def timing_arms(chain):
    if chain == 'backward':
        return ('staged', 'fused', 'default', 'folded')
    if chain in ('forward', 'remat'):
        return ('legacy', 'packed', 'default')
    raise ValueError('unknown balanced timing chain')


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def merge_allocation_runs(runs, *, samples, warmup):
    if not runs:
        raise ValueError('missing construction rotations')
    first = runs[0]
    backward = first.get('chain') == 'backward'
    arms = timing_arms(first.get('chain'))
    roles = len(arms)
    if len(runs) != roles or samples < roles or samples % roles:
        raise ValueError('balanced timing requires equal samples in every construction rotation')
    if [r.get('allocation_rotation') for r in runs] != list(range(roles)):
        raise ValueError('missing, duplicate or reordered allocation rotation')
    count = samples // roles
    pooled = {arm: [] for arm in arms}
    metadata = METADATA + (SCHEDULES if backward else ())
    for run in runs:
        if any(key not in run or run[key] != first[key] for key in metadata):
            raise ValueError('allocation rotation metadata changed')
        if run.get('samples') != count or run.get('warmup') != warmup:
            raise ValueError('allocation rotation sample count or warmup changed')
        tensors = run.get('tensors', [])
        if len(tensors) != len(TENSORS) or {t.get('label') for t in tensors} != TENSORS:
            raise ValueError('allocation rotation tensor inventory changed')
        if run.get('worst_max_abs') != 0 or any(
                t.get('max_abs') != 0 or not isinstance(t.get('values'), int) or t['values'] <= 0
                or not isinstance(t.get('worst_excess'), (int, float))
                or not math.isfinite(t['worst_excess']) or t['worst_excess'] > 0 for t in tensors):
            raise ValueError('allocation rotation failed exact tensor parity')
        if backward:
            for key, limit in (('folded_max_abs', 1e-6), ('folded_parameter_delta', 1e-6),
                               ('folded_key_parameter_max_abs', 0.0)):
                value = run.get(key)
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
                    raise ValueError('allocation rotation failed folded numeric gate: '+key)
            if not positive(run.get('staged_key_parameter_max_abs')):
                raise ValueError('allocation rotation has a zero/invalid staged gradient reference')
        traces = run.get('timing_samples')
        if not isinstance(traces, list) or len(traces) != roles:
            raise ValueError('missing raw allocation timing traces')
        for index, arm in enumerate(pooled):
            values = traces[index]
            if not isinstance(values, list) or len(values) != count or not all(map(positive, values)):
                raise ValueError('invalid raw allocation timing observations')
            leg = run.get(arm, {})
            label = run['default_route'] if arm == 'default' else arm
            measured = statistics.median(values)
            if leg.get('label') != label or not positive(leg.get('median_ms')) or not math.isclose(
                    leg['median_ms'], measured, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('allocation median disagrees with raw timing observations')
            pooled[arm].extend(values)
    merged = copy.deepcopy(first)
    merged.pop('allocation_rotation')
    merged['samples'] = samples
    merged['warmup'] = warmup
    merged['timing_protocol'] = BACKWARD_PROTOCOL if backward else PROTOCOL
    merged['allocation_runs'] = copy.deepcopy(runs)
    merged['timing_samples'] = list(pooled.values())
    for arm, values in pooled.items():
        measured = statistics.median(values)
        merged[arm]['median_ms'] = measured
        merged[arm]['ms_per_step'] = measured / merged['steps']
    if backward:
        for arm in ('fused', 'folded'):
            merged[arm+'_over_staged'] = merged[arm]['median_ms'] / merged['staged']['median_ms']
        for key in ('folded_max_abs', 'folded_parameter_delta', 'folded_key_parameter_max_abs'):
            merged[key] = max(run[key] for run in runs)
        merged['staged_key_parameter_max_abs'] = min(run['staged_key_parameter_max_abs'] for run in runs)
    else:
        merged['packed_over_legacy'] = merged['packed']['median_ms'] / merged['legacy']['median_ms']
    return merged
