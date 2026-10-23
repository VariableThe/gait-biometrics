import unittest

import numpy as np

from models.gait_metric import WithinClassMetric


class WithinClassMetricTests(unittest.TestCase):
    def test_normalized_roundtrip_and_no_inference_refit(self):
        rng = np.random.default_rng(4)
        values = rng.normal(size=(40, 8))
        subjects = np.repeat(np.arange(4), 10)
        metric = WithinClassMetric().fit(values, subjects)
        state = metric.state_dict()
        shifted = values + 100
        actual = metric.transform(shifted)
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-6)
        restored = WithinClassMetric.from_state_dict(state)
        np.testing.assert_allclose(restored.transform(shifted), actual)
        for key, value in state.items():
            np.testing.assert_array_equal(value, metric.state_dict()[key])

    def test_only_training_ids_and_rows_are_consumed(self):
        rng = np.random.default_rng(5)
        values = rng.normal(size=(20, 3))
        metric = WithinClassMetric(shrinkage=0.2).fit(values, np.repeat([3, 7], 10))
        self.assertEqual(metric.training_subject_ids_, [3, 7])
        self.assertEqual(metric.mean_.shape, (3,))
        with self.assertRaises(ValueError):
            metric.transform(np.ones((2, 4)))
        with self.assertRaises(ValueError):
            WithinClassMetric().fit(values, np.zeros(20, dtype=int))


if __name__ == "__main__":
    unittest.main()
