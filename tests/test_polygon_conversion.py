"""Exercise native COCO/OpenCV conversion, including the column-major regression."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
READY = all(importlib.util.find_spec(p) is not None for p in ('torch', 'ultralytics', 'cv2', 'pycocotools'))
if READY:
    from pycocotools import mask as coco
    from prepare_experiment import polygon_for


@unittest.skipUnless(READY, 'Install the project dependencies to run native polygon tests')
class PolygonTests(unittest.TestCase):
    def test_coco_column_major_mask_can_be_drawn(self):
        mask = np.zeros((64, 80), np.uint8); mask[5:25, 8:20] = 1
        rle = coco.encode(np.asfortranarray(mask))
        self.assertTrue(coco.decode(rle).flags.f_contiguous)
        points, quality = polygon_for(rle)
        self.assertEqual(quality, 1.)
        self.assertTrue(np.isfinite(points).all())

    def test_disconnected_instance_has_valid_polygon(self):
        mask = np.zeros((64, 80), np.uint8)
        mask[5:25, 8:20] = 1; mask[40:55, 50:70] = 1
        points, quality = polygon_for(coco.encode(np.asfortranarray(mask)))
        self.assertTrue(0 < quality <= 1)
        self.assertTrue(((points >= 0) & (points <= 1)).all())
        self.assertEqual(points.size % 2, 0)


if __name__ == '__main__':
    unittest.main()
