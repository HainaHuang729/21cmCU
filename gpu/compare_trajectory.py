"""Compare sampled cells at every internal step and smoothing scale."""
import argparse
from collections import defaultdict
import ctypes
import json
import math
from pathlib import Path


COORDS = [(23, 5, 19), (2, 6, 123), (8, 59, 88), (115, 74, 46),
          (24, 49, 72), (127, 35, 44), (121, 7, 45)]
STEP_FIELDS = ('previous_z', 'xhi', 'dNrec', 'gamma12', 'z_reion', 'mfp',
               'density', 'xe', 'Tk', 'Ts')
SCALE_FIELDS = ('radius', 'last', 'density', 'xe', 'filtered_rec', 'raw_fcoll',
                'normalization', 'efficiency', 'collapse_floor', 'prior_xhi', 'prior_gamma12',
                'min_density', 'max_density', 'gamma_prefactor')


def parse(path, steps, scales):
    for line in path.read_text().splitlines():
        event, redshift, index, *data = line.split()
        redshift = ctypes.c_float(float(redshift)).value
        index = int(index)
        values = STEP_FIELDS if event == 'step' else SCALE_FIELDS
        assert len(data) == len(values), (path, line)
        row = dict(zip(values, map(float, data), strict=True))
        if event == 'step':
            steps[index][redshift] = row
        elif event == 'scale':
            scales[index][redshift].append(row)
        else:
            raise ValueError(event)


def margin(row):
    lhs = max(row['normalization'] * row['raw_fcoll'], row['collapse_floor']) * row['efficiency']
    rhs = (1 - row['xe']) * (1 + row['filtered_rec'] / (1 + row['density']))
    return lhs - rhs


def discrepancy(reference, actual):
    diff = abs(reference - actual)
    allowed = 1e-8 + 1e-4 * abs(reference)
    return {'cpu': reference, 'gpu': actual, 'abs_diff': diff,
            'allowed': allowed, 'ratio_to_tolerance': diff / allowed}


def compare(cpu_steps, gpu_steps, cpu_scales, gpu_scales, partial=False):
    results = []
    for i, coords in enumerate(COORDS):
        zs = sorted(cpu_steps[i].keys() & gpu_steps[i].keys(), reverse=True)
        if not partial:
            assert len(zs) == len(cpu_steps[i]) == len(gpu_steps[i]), (i, len(zs))
        first = {}
        flips = []
        first_full_ionization_difference = None
        for z in zs:
            cpu, gpu = cpu_steps[i][z], gpu_steps[i][z]
            assert math.isclose(cpu['previous_z'], gpu['previous_z'], abs_tol=1e-6)
            for k in STEP_FIELDS[1:]:
                delta = discrepancy(cpu[k], gpu[k])
                if delta['ratio_to_tolerance'] > 1 and k not in first:
                    first[k] = {'z': z, **delta}
            if (cpu['xhi'] == 0) != (gpu['xhi'] == 0) and first_full_ionization_difference is None:
                first_full_ionization_difference = {'z': z, 'cpu': cpu, 'gpu': gpu}
            if z not in cpu_scales[i] or z not in gpu_scales[i]:
                continue
            cr, gr = cpu_scales[i][z], gpu_scales[i][z]
            assert len(cr) == len(gr) and len(cr), (i, z, len(cr), len(gr))
            for radius_index, (c, g) in enumerate(zip(cr, gr, strict=True)):
                assert math.isclose(c['radius'], g['radius'], rel_tol=2e-7)
                assert c['last'] == g['last']
                cm, gm = margin(c), margin(g)
                if (cm > 0) != (gm > 0):
                    flips.append({'z': z, 'radius_index': radius_index, 'radius': c['radius'],
                                  'cpu_margin': cm, 'gpu_margin': gm,
                                  'cpu': c, 'gpu': g,
                                  'step_final': {'cpu': cpu, 'gpu': gpu}})
        results.append({'coords': coords, 'matched_steps': len(zs),
                        'first_tolerance_failure': first,
                        'first_fully_ionized_decision_difference': first_full_ionization_difference,
                        'different_scale_decisions': flips})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cpu', type=Path)
    parser.add_argument('gpu', type=Path)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cs, gs = defaultdict(dict), defaultdict(dict)
    cr = defaultdict(lambda: defaultdict(list))
    gr = defaultdict(lambda: defaultdict(list))
    parse(args.cpu / 'trajectory.tsv', cs, cr)
    parse(args.gpu / 'trajectory.tsv', gs, gr)
    parse(args.gpu / 'gpu-scales.tsv', gs, gr)
    results = compare(cs, gs, cr, gr, args.partial)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + '\n')
    for cell in results:
        first = cell['first_tolerance_failure']
        print(cell['coords'], cell['matched_steps'], {k: (round(v['z'], 6), v['cpu'], v['gpu'])
              for k, v in first.items()})
        slips = cell['different_scale_decisions']
        if slips:
            x = slips[0]
            print('  first barrier crossing mismatch', len(slips), x['z'], x['radius'],
                  x['cpu_margin'], x['gpu_margin'])
        d = cell['first_fully_ionized_decision_difference']
        if d:
            print('  first full-ionization mismatch', d['z'], d['cpu']['xhi'], d['gpu']['xhi'])


if __name__ == '__main__':
    main()
