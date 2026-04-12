"""Tests for the DynamicWeightManager."""

import numpy as np
import pytest

from aegis5 import DynamicWeightManager


def test_initial_weights_are_uniform():
    wm = DynamicWeightManager(n_classifiers=5, n_classes=3)
    w = wm.get_weights()
    assert w.shape == (5, 3)
    np.testing.assert_allclose(w, 1.0 / 5)


def test_weights_sum_to_one_per_class():
    wm = DynamicWeightManager(n_classifiers=3, n_classes=4)
    # Feed some predictions to classifier 0
    wm.update(0, [0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 0, 1])  # perfect
    wm.update(1, [0, 1, 2, 3, 0, 1], [1, 0, 3, 2, 1, 0])  # wrong
    wm.update(2, [0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 1, 0])  # mostly right

    w = wm.get_weights()
    for c in range(4):
        assert abs(w[:, c].sum() - 1.0) < 1e-10


def test_perfect_classifier_gets_highest_weight():
    wm = DynamicWeightManager(n_classifiers=3, n_classes=2, beta=2.0)
    y_true = [0, 1, 0, 1, 0, 1, 0, 1] * 10

    # Classifier 0: perfect
    wm.update(0, y_true, y_true)
    # Classifier 1: random
    wm.update(1, y_true, [1, 0, 1, 0, 1, 0, 1, 0] * 10)
    # Classifier 2: always predicts 0
    wm.update(2, y_true, [0] * 80)

    w = wm.get_weights()
    # Classifier 0 should have the highest weight for both classes
    assert w[0, 0] > w[1, 0]
    assert w[0, 0] > w[2, 0]
    assert w[0, 1] > w[1, 1]


def test_higher_beta_sharpens_weights():
    y_true = [0, 1, 0, 1] * 20
    y_good = y_true
    y_bad = [1, 0, 1, 0] * 20

    wm_low = DynamicWeightManager(n_classifiers=2, n_classes=2, beta=0.5)
    wm_low.update(0, y_true, y_good)
    wm_low.update(1, y_true, y_bad)

    wm_high = DynamicWeightManager(n_classifiers=2, n_classes=2, beta=5.0)
    wm_high.update(0, y_true, y_good)
    wm_high.update(1, y_true, y_bad)

    # Higher beta -> more extreme weight difference
    low_diff = abs(wm_low.get_weights()[0, 0] - wm_low.get_weights()[1, 0])
    high_diff = abs(wm_high.get_weights()[0, 0] - wm_high.get_weights()[1, 0])
    assert high_diff > low_diff


def test_sliding_window_forgets_old_predictions():
    # Need 2 classifiers so softmax can actually differentiate weights.
    # (Softmax of a single value is always 1.0 regardless of input.)
    wm = DynamicWeightManager(n_classifiers=2, n_classes=2, window_size=10)

    # Classifier 1 is consistently mediocre
    wm.update(1, [0, 1] * 5, [0, 0] * 5)

    # Classifier 0 starts perfect
    wm.update(0, [0, 1] * 5, [0, 1] * 5)
    w_after_good = wm.get_weights()[0, 1]  # class-1 weight for clf 0

    # Now classifier 0 becomes bad (window drops the good ones)
    wm.update(0, [0, 1] * 5, [1, 0] * 5)
    w_after_bad = wm.get_weights()[0, 1]

    assert w_after_bad < w_after_good


def test_get_weights_returns_copy():
    wm = DynamicWeightManager(n_classifiers=2, n_classes=2)
    w1 = wm.get_weights()
    w1[0, 0] = 999.0
    w2 = wm.get_weights()
    assert w2[0, 0] != 999.0
