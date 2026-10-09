
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

import numpy as np
import torch
from datasets import Dataset
from egquant.common import experiment_from_args, load_torch
from egquant.conversion import importance
from egquant.kernels import composition3, composition4, conversion3, conversion4
import generate_baselines
import prepare_tokens


class SyntheticTokenizer:
    def __call__(self, text, return_tensors='pt'):
        document, length = map(int, text.split(':'))
        return SimpleNamespace(input_ids=(document * 10000 + torch.arange(length))[None])


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.args = ['prepare_tokens.py', '--model', 'llama2_7b', '--bits', '3',
                     '--model-dir', str(self.root / 'model'), '--data-dir', str(self.root / 'data'),
                     '--work-dir', str(self.root / 'work')]
        self.cfg = experiment_from_args(SimpleNamespace(model='llama2_7b', bits=3,
                        model_dir=str(self.root / 'model'), data_dir=str(self.root / 'data'),
                        work_dir=str(self.root / 'work')))

    def prepare(self, method, dataset=None, cache=None):
        args = self.args + ['--dataset', method]
        if cache is not None:
            args += ['--from-cache', str(cache)]
        with patch.object(sys, 'argv', args), \
                patch('transformers.AutoTokenizer.from_pretrained', return_value=SyntheticTokenizer()), \
                patch('datasets.load_from_disk', return_value=dataset), \
                contextlib.redirect_stdout(io.StringIO()):
            prepare_tokens.main()
        return load_torch(self.cfg['paths'][method + '_tokens'])

    def test_matching_full_length_samples_are_reproducible(self):
        documents = [f'{i}:{2048 + (i % 4) * 100}' for i in range(140)]
        dataset = Dataset.from_dict({'text': [' ', '200:512'] + documents})
        awq = self.prepare('awq', dataset)
        gptq = self.prepare('gptq', dataset)
        repeated = self.prepare('awq', dataset)
        for method, cache in [('awq', awq), ('gptq', gptq)]:
            with self.subTest(method=method):
                spec = self.cfg['baselines'][method]
                self.assertEqual((spec['seed'], spec['samples'], spec['sequence_length']), (42, 128, 2048))
                self.assertEqual(tuple(cache['tokens'].shape), (128, 2048))
                self.assertEqual(cache['sampling_seed'], 42)
                ids = cache['tokens'] // 10000
                self.assertTrue(torch.all(ids == ids[:, :1]))
                self.assertEqual(len(torch.unique(ids[:, 0])), 128)
                self.assertTrue(torch.all(ids < 140))
                self.assertTrue(torch.all(torch.diff(cache['tokens'], dim=1) == 1))
        torch.testing.assert_close(awq['tokens'], gptq['tokens'], rtol=0, atol=0)
        torch.testing.assert_close(awq['tokens'], repeated['tokens'], rtol=0, atol=0)
        self.assertEqual(awq['token_sha256'], repeated['token_sha256'])

    def test_documents_of_exactly_2048_tokens_are_accepted(self):
        dataset = Dataset.from_dict({'text': [f'{i}:2048' for i in range(128)]})
        for method in ('awq', 'gptq'):
            with self.subTest(method=method):
                cache = self.prepare(method, dataset)
                self.assertEqual(tuple(cache['tokens'].shape), (128, 2048))
                self.assertTrue(torch.all(cache['tokens'][:, 0] % 10000 == 0))

    def test_insufficient_documents_are_rejected(self):
        dataset = Dataset.from_dict({'text': [f'{i}:2048' for i in range(127)]})
        for method in ('awq', 'gptq'):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, 'Not enough'):
                self.prepare(method, dataset)

    def test_imported_caches_require_128_by_2048(self):
        path = self.root / 'cache.pt'
        for method in ('awq', 'gptq'):
            for shape in [(32, 2048), (128, 512), (128 * 2048,)]:
                with self.subTest(method=method, shape=shape):
                    torch.save(torch.zeros(shape, dtype=torch.long), path)
                    with self.assertRaisesRegex(ValueError, 'calibration token shape mismatch'):
                        self.prepare(method, cache=path)
            torch.save(torch.zeros((128, 2048), dtype=torch.long), path)
            self.assertEqual(tuple(self.prepare(method, cache=path)['tokens'].shape), (128, 2048))

    def test_baseline_generation_rejects_legacy_short_cache_before_loading_model(self):
        for method in ('awq', 'gptq'):
            cache = dict(model_id='llama2_7b', dataset=method, tokens=torch.zeros((32, 2048), dtype=torch.long))
            import hashlib
            cache['token_sha256'] = hashlib.sha256(cache['tokens'].numpy().tobytes()).hexdigest()
            with self.subTest(method=method), patch('torch.cuda.is_available', return_value=True), \
                    patch.object(generate_baselines, 'load_torch', return_value=cache), \
                    patch.object(generate_baselines, 'get_model') as get_model:
                with self.assertRaisesRegex(ValueError, 'calibration sample count mismatch'):
                    generate_baselines.generate(self.cfg, method)
                get_model.assert_not_called()
                self.assertFalse(Path(self.cfg['paths'][method]).exists())


class CompositionTests(unittest.TestCase):
    def test_first_update_uses_nearest_shared_scale_reconstruction(self):
        for bits, conv, comp in [(3, conversion3, composition3), (4, conversion4, composition4)]:
            with self.subTest(bits=bits):
                rng = np.random.default_rng(0)
                x = rng.normal(0, 0.02, 2048).astype(np.float32)
                f = rng.lognormal(0, 1, 2048).astype(np.float32)
                qa = (np.round(x / 0.008) * 0.008).astype(np.float32)
                qg = (np.round((x + 0.001) / 0.007) * 0.007 - 0.001).astype(np.float32)
                ca, ta = conv.fit_group(x, f * (1 + importance(x, qa)))
                cg, tg = conv.fit_group(x, f * (1 + importance(x, qg)))
                captured = {}
                original_refine = comp.refine

                def check_initialization(xx, ff, cc, tt, zz):

                    expected = np.empty_like(zz)
                    for h, row in enumerate(xx):
                        for j, weight in enumerate(row):
                            expected[h, j] = min(range(2 ** bits), key=lambda k: (weight - tt[h] * cc[k])**2)
                    np.testing.assert_array_equal(zz, expected)
                    captured['initial_loss'] = float(np.sum(ff * (xx - tt[:, None] * cc[expected])**2))
                    captured['z'] = zz.copy()
                    result = original_refine(xx, ff, cc, tt, zz)
                    self.assertTrue(np.all(np.diff(result[2]) <= 1e-10))
                    return result

                with patch.object(comp, 'refine', side_effect=check_initialization) as refine:
                    c, t, selected, stats = comp.fit_group(x, f, ca, ta, cg, tg, audit=True)
                refine.assert_called_once()
                source_levels = np.concatenate((ta.astype(np.float64)[:, None] * ca,
                                                tg.astype(np.float64)[:, None] * cg), axis=1)[:, selected]
                error = (x.astype(np.float64).reshape(16, 128, 1) - source_levels[:, None, :])**2
                old_z = error.argmin(-1)
                self.assertGreater(np.count_nonzero(old_z != captured['z']), 0)
                self.assertAlmostEqual(stats[3], float(np.sum(f.reshape(16, 128) * error.min(-1))))
                self.assertAlmostEqual(stats[4], captured['initial_loss'] * float(f.max()))
                self.assertLessEqual(stats[5], stats[4] + 1e-8)
                self.assertTrue(np.isfinite(c).all() and np.isfinite(t).all())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
