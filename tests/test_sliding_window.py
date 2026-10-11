import numpy as np

from ligandseek.inference.sliding_window import (
    WindowSpec,
    _build_gaussian_weight,
    _merge_probability,
)


def test_core_crop_removes_interior_faces_and_preserves_outer_faces():
    shape = (4, 4, 120)
    value_sum = np.zeros(shape, dtype=np.float32)
    weight_sum = np.zeros(shape, dtype=np.float32)
    weight = _build_gaussian_weight((4, 4, 80), .5)
    _merge_probability(
        value_sum, weight_sum, np.ones((4, 4, 80), dtype=np.float32),
        WindowSpec((0, 0, 0), (4, 4, 80)), weight, core_offset=5,
    )
    _merge_probability(
        value_sum, weight_sum, np.zeros((4, 4, 80), dtype=np.float32),
        WindowSpec((0, 0, 40), (4, 4, 80)), weight, core_offset=5,
    )
    probability = value_sum / weight_sum
    np.testing.assert_array_equal(probability[..., 0], 1.)
    np.testing.assert_array_equal(probability[..., 42], 1.)
    np.testing.assert_array_equal(probability[..., 77], 0.)
    np.testing.assert_array_equal(probability[..., 119], 0.)
    assert np.all(weight_sum > 0)
    assert np.all((probability[..., 60] > 0) & (probability[..., 60] < 1))
