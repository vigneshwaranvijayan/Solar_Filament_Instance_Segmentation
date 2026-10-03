"""Geometry and leakage checks, kept separate for CPU testing."""
import numpy as np


def context_crop(box, width, height, padding=1.5, minimum=48):
    x1, y1, x2, y2 = map(float, box)
    side = min(max((x2-x1)*padding, (y2-y1)*padding, minimum), max(width, height))
    cx, cy = (x1+x2)/2, (y1+y2)/2
    xa = max(0, min(int(np.floor(cx-side/2)), max(0, width-int(np.ceil(side)))))
    ya = max(0, min(int(np.floor(cy-side/2)), max(0, height-int(np.ceil(side)))))
    xb, yb = min(width, max(xa+1, int(np.ceil(xa+side)))), min(height, max(ya+1, int(np.ceil(ya+side))))
    return xa, ya, xb, yb


def box_iou(box, boxes):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if not len(boxes): return np.zeros(0)
    low = np.maximum(np.asarray(box)[:2], boxes[:, :2]); high = np.minimum(np.asarray(box)[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(0, high-low), axis=1)
    a = max(0, box[2]-box[0])*max(0, box[3]-box[1])
    b = np.prod(np.maximum(0, boxes[:, 2:]-boxes[:, :2]), axis=1)
    return intersection / np.maximum(a+b-intersection, 1e-9)


def assert_partitions(partitions, groups):
    seen_names, seen_groups = set(), set()
    for key in ('train', 'calibration', 'validation'):
        names = set(partitions[key]); current = {groups[n] for n in names}
        if names & seen_names or current & seen_groups:
            raise ValueError('Physical images or temporal groups overlap between partitions')
        if not names: raise ValueError('Empty partition: '+key)
        seen_names |= names; seen_groups |= current


def ordered_disjoint(patches, shape, minimum):
    """Highest score owns overlap; fragments of an instance stay in one mask."""
    occupied = np.zeros(shape, bool); kept = []
    for record in sorted(patches, key=lambda r: -r['score']):
        x1, y1, x2, y2 = record['crop']; mask = np.asarray(record['mask'], bool)
        if mask.shape != (y2-y1, x2-x1): raise ValueError('Crop/mask shape mismatch')
        local = mask & ~occupied[y1:y2, x1:x2]
        if local.sum() < minimum: continue
        occupied[y1:y2, x1:x2] |= local
        kept.append({**record, 'mask': local})
    return kept
