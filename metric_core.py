"""Pooled instance metrics and temporal-group confidence intervals."""
import numpy as np

def counts(iou):
    iou = np.asarray(iou, dtype=float)
    hit, overlap = (iou > 0.5, iou > 0)
    return np.array([hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum(), iou[hit].sum(), (overlap.sum(1) > 1).sum(), (overlap.sum(0) > 1).sum()], float)

def report(total):
    tp, fp, fn, summed, split, merge = map(float, total)
    denom = tp + 0.5 * fp + 0.5 * fn
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), sum_iou=summed, split_gt=int(split), merged_predictions=int(merge), pq=summed / denom if denom else 0.0, sq=summed / tp if tp else 0.0, rq=tp / denom if denom else 0.0)

def grouped_interval(baseline_counts, candidate_counts, groups, seed=2026, draws=1000):
    """Paired temporal-group bootstrap of pooled PQ differences."""
    base, candidate = (np.asarray(baseline_counts), np.asarray(candidate_counts))
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        return None
    a = np.stack([base[groups == group].sum(0) for group in unique])
    b = np.stack([candidate[groups == group].sum(0) for group in unique])
    rng = np.random.default_rng(seed)
    delta = []
    for _ in range(draws):
        sample = rng.integers(0, len(unique), len(unique))
        delta.append(report(b[sample].sum(0))['pq'] - report(a[sample].sum(0))['pq'])
    low, high = np.quantile(delta, [0.025, 0.975])
    return {'lower': float(low), 'upper': float(high), 'draws': draws, 'groups': len(unique)}
