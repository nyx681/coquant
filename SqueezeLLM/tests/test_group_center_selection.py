import itertools
import unittest

import numpy as np
import torch

from quantization.group_center_selection import assign, select_centers
from squeezellm.fake_quant import _dequantize_rows


class GroupSelectionTest(unittest.TestCase):
    def test_against_exhaustive_search(self):
        rng = np.random.default_rng(123)
        for _ in range(80):
            x = np.r_[rng.normal(size=15), -8., 8., 0.]
            s = rng.uniform(0, 3, len(x))
            s[::4] = 0
            candidates = np.unique(np.r_[rng.normal(size=7), 0.])
            k = int(rng.integers(1, 6))
            selected = select_centers(x, s, candidates, k)
            exhaustive = min(assign(x, s, c)[1] for c in itertools.combinations(candidates, k))
            self.assertAlmostEqual(assign(x, s, selected)[1], exhaustive, places=10)
            self.assertTrue(np.isin(selected, candidates).all())

    def test_duplicates_and_zero_weights(self):
        self.assertEqual(len(select_centers([0, 1], [1, 1], [0, 0, 1], 16)), 16)
        c = select_centers([0, 1], [0, 0], np.arange(32), 16)
        self.assertEqual(len(np.unique(c)), 16)
        with self.assertRaises(ValueError):
            select_centers([0], [-1], [0])

    def test_group_and_legacy_loading(self):
        rng = np.random.default_rng(4)
        centers = rng.normal(size=(3, 2, 16)).astype(np.float32)
        labels = rng.integers(0, 16, size=(3, 2, 128), dtype=np.uint8)
        lut = dict(format='squeezellm_group_lut_v1', group_size=128,
                   centers=centers, labels=labels)
        result = _dequantize_rows(lut, (3, 256), torch.float32, chunk_rows=2)
        expected = np.take_along_axis(centers, labels.astype(int), axis=2).reshape(3, 256)
        np.testing.assert_array_equal(result.numpy(), expected)
        legacy = [[(centers[r, 0], labels[r, 0])] for r in range(3)]
        old = _dequantize_rows(legacy, (3, 128), torch.float32)
        np.testing.assert_array_equal(old.numpy(), expected[:, :128])
        with self.assertRaises(ValueError):
            _dequantize_rows(lut, (4, 256), torch.float32)
        labels[0, 0, 0] = 16
        with self.assertRaises(ValueError):
            _dequantize_rows(lut, (3, 256), torch.float32)


if __name__ == '__main__':
    unittest.main()
