"""Re-derive balanced packed timings from every raw construction rotation.

Each role occupies each physical allocation position once. Equal-length raw
traces are pooled, retaining the requested total observations per leg. No run,
sample or rotation is selected according to which route wins.
"""
import copy
import math
import statistics

PROTOCOL = 'allocation-balanced-shipped-v2'
TENSORS = {'outputs', 'final_packed_state', 'grad_x', 'grad_initial_packed_state', 'token_feature_grad'}


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def merge_allocation_runs(runs, *, samples, warmup):
    if len(runs) != 3 or samples < 3 or samples % 3:
        raise ValueError('balanced timing requires three equal construction rotations')
    if [r.get('allocation_rotation') for r in runs] != [0, 1, 2]:
        raise ValueError('missing, duplicate or reordered allocation rotation')
    count = samples // 3
    first = runs[0]
    pooled = {arm: [] for arm in ('legacy', 'packed', 'default')}
    metadata = ('device', 'width', 'head_size', 'batch', 'steps', 'chain',
                'parameter_storage', 'precision_policy', 'default_route', 'recorded_route')
    for run in runs:
        if any(key not in run or run[key] != first[key] for key in metadata):
            raise ValueError('allocation rotation metadata changed')
        if run['chain'] not in ('forward', 'remat'):
            raise ValueError('allocation protocol is only for forward/rematerialization')
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
        traces = run.get('timing_samples')
        if not isinstance(traces, list) or len(traces) != 3:
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
    merged['timing_protocol'] = PROTOCOL
    merged['allocation_runs'] = copy.deepcopy(runs)
    merged['timing_samples'] = list(pooled.values())
    for arm, values in pooled.items():
        measured = statistics.median(values)
        merged[arm]['median_ms'] = measured
        merged[arm]['ms_per_step'] = measured / merged['steps']
    merged['packed_over_legacy'] = merged['packed']['median_ms'] / merged['legacy']['median_ms']
    return merged
