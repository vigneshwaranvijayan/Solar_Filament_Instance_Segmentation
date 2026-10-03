import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry import context_crop, assert_partitions, ordered_disjoint
from metric_core import counts, report, grouped_interval


class MetricTests(unittest.TestCase):
    def test_matching_is_strictly_above_half(self):
        self.assertEqual(report(counts(np.array([[.5]])))['tp'], 0)
        self.assertEqual(report(counts(np.array([[.50001]])))['tp'], 1)

    def test_false_positives_and_misses(self):
        result = report(counts(np.array([[.8, 0], [0, 0]])))
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 1, 1))
        self.assertAlmostEqual(result['pq'], .4)

    def test_empty_predictions_still_count_misses(self):
        self.assertEqual(report(counts(np.empty((3, 0))))['fn'], 3)
        self.assertEqual(report(counts(np.empty((0, 2))))['fp'], 2)

    def test_paired_bootstrap_identical_systems(self):
        totals = np.array([[1, 1, 1, .8, 0, 0], [2, 0, 1, 1.5, 0, 0]])
        result = grouped_interval(totals, totals, ['date_a', 'date_b'], draws=100)
        self.assertEqual((result['lower'], result['upper']), (0., 0.))


class GeometryTests(unittest.TestCase):
    def test_crop_contains_object_at_disk_image_edge(self):
        box = [2010, 2000, 2048, 2048]
        x1, y1, x2, y2 = context_crop(box, 2048, 2048)
        self.assertTrue(0 <= x1 <= box[0] < box[2] <= x2 <= 2048)
        self.assertTrue(0 <= y1 <= box[1] < box[3] <= y2 <= 2048)

    def test_date_group_leakage_is_rejected(self):
        partitions = {'train': ['a'], 'calibration': ['b'], 'validation': ['c']}
        with self.assertRaises(ValueError):
            assert_partitions(partitions, {'a': 'same_date', 'b': 'same_date', 'c': 'other'})

    def test_stronger_instance_owns_overlap(self):
        high = np.zeros((10, 10), bool); high[:6, :6] = 1
        low = np.zeros_like(high); low[4:, 4:] = 1
        result = ordered_disjoint([{'crop': (0, 0, 10, 10), 'mask': low, 'score': .2}, {'crop': (0, 0, 10, 10), 'mask': high, 'score': .9}], high.shape, 8)
        self.assertEqual([int(r['mask'].sum()) for r in result], [36, 32])
        self.assertFalse((result[0]['mask'] & result[1]['mask']).any())


if __name__ == '__main__':
    unittest.main()
