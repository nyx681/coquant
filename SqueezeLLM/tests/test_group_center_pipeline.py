import sys
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'quantization'))
from fuse_group_centers import process_row, validate_record, VARIANTS
from squeezellm.fake_quant import _dequantize_rows


class PipelineTest(unittest.TestCase):
    def test_native_mapping_validation(self):
        scale = torch.tensor([[.125, .25]])
        zero = torch.tensor([[8., 7.]])
        codes = torch.arange(256).reshape(1, 2, 128) % 16
        w = (scale[:, :, None]*(codes-zero[:, :, None])).reshape(1, 256).half()
        record = dict(weight=w, scale=scale, zero=zero, group_size=128,
                      maxq=15, static_groups=True)
        self.assertEqual(validate_record(record, w.shape), 0.)
        record['g_idx'] = torch.arange(256)//128
        record['g_idx'][0] = 1
        with self.assertRaises(ValueError):
            validate_record(record, w.shape)
        del record['g_idx']
        record['scale'] = scale * 1.37
        with self.assertRaises(ValueError):
            validate_record(record, w.shape)

    def test_round_trip_objectives(self):
        rng = np.random.default_rng(43)
        w = rng.normal(size=256).astype(np.float32)
        s = rng.uniform(.1, 4, 256).astype(np.float32)
        scales = np.array([.4, .5], np.float32)
        zeros = np.array([8., 8.], np.float32)
        native = np.concatenate([scales[g]*(np.clip(np.round(w[g*128:(g+1)*128]/scales[g])+zeros[g],0,15)-zeros[g]) for g in range(2)])
        books, labels, metrics = process_row((w,s,scales,zeros,native))
        for i,v in enumerate(VARIANTS):
            lut = dict(format='squeezellm_group_lut_v1', group_size=128,
                       centers=books[v][None], labels=labels[v][None])
            reconstructed = _dequantize_rows(lut, (1,256), torch.float32).numpy()[0]
            objective = np.dot(s.astype(float), (w.astype(float)-reconstructed)**2)
            self.assertAlmostEqual(objective, metrics[i,0], places=10)
        self.assertLessEqual(metrics[2,0], min(metrics[0,0],metrics[1,0])+1e-10)
        self.assertLessEqual(metrics[3,0], metrics[2,0]+1e-10)
        for g in range(2):
            candidates = np.r_[books['awq'][g],books['gptq_nearest'][g]]
            self.assertTrue(np.isin(books['selected'][g],candidates).all())


if __name__ == '__main__':
    unittest.main()
