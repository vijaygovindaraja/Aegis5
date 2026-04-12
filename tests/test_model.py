"""Tests for the Aegis5 model."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from aegis5 import Aegis5


def _make_dataset(n_samples=500, n_features=20, n_classes=3, random_state=42):
    """Generate a synthetic classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=12,
        n_redundant=4,
        n_classes=n_classes,
        random_state=random_state,
        flip_y=0.05,
    )
    return X, y


def test_fit_predict_returns_correct_shape():
    X, y = _make_dataset(n_samples=200, n_classes=3)
    model = Aegis5(
        use_feature_selection=False,
        use_pca=False,
        use_smote=False,
        random_state=0,
    )
    model.fit(X[:150], y[:150])
    preds = model.predict(X[150:])
    assert preds.shape == (50,)


def test_predict_returns_known_labels():
    X, y = _make_dataset(n_samples=200, n_classes=3)
    model = Aegis5(
        use_feature_selection=False,
        use_pca=False,
        use_smote=False,
        random_state=0,
    )
    model.fit(X[:150], y[:150])
    preds = model.predict(X[150:])
    assert set(preds).issubset(set(y))


def test_predict_proba_sums_to_one():
    X, y = _make_dataset(n_samples=200, n_classes=3)
    model = Aegis5(
        use_feature_selection=False,
        use_pca=False,
        use_smote=False,
        random_state=0,
    )
    model.fit(X[:150], y[:150])
    proba = model.predict_proba(X[150:])
    assert proba.shape == (50, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_accuracy_above_chance():
    X, y = _make_dataset(n_samples=400, n_classes=3)
    model = Aegis5(
        use_feature_selection=False,
        use_pca=False,
        use_smote=True,
        random_state=42,
    )
    model.fit(X[:300], y[:300])
    results = model.evaluate(X[300:], y[300:])
    # 3-class, chance = 0.33. Model should be well above.
    assert results['accuracy'] > 0.60


def test_evaluate_returns_all_metrics():
    X, y = _make_dataset(n_samples=200, n_classes=2)
    model = Aegis5(
        use_feature_selection=False,
        use_pca=False,
        use_smote=False,
        random_state=0,
    )
    model.fit(X[:150], y[:150])
    results = model.evaluate(X[150:], y[150:])
    assert 'accuracy' in results
    assert 'precision' in results
    assert 'recall' in results
    assert 'f1_score' in results
    assert 'confusion_matrix' in results
    assert 'predictions' in results


def test_with_smote():
    X, y = _make_dataset(n_samples=300, n_classes=3)
    model = Aegis5(
        use_feature_selection=False,
        use_pca=False,
        use_smote=True,
        random_state=42,
    )
    model.fit(X[:200], y[:200])
    preds = model.predict(X[200:])
    assert len(preds) == 100


def test_with_pca():
    X, y = _make_dataset(n_samples=300, n_features=30, n_classes=2)
    model = Aegis5(
        use_feature_selection=False,
        use_pca=True,
        pca_variance=0.95,
        use_smote=False,
        random_state=0,
    )
    model.fit(X[:200], y[:200])
    preds = model.predict(X[200:])
    assert len(preds) == 100


def test_handles_nan_in_features():
    X, y = _make_dataset(n_samples=200, n_classes=2)
    # Inject some NaNs
    rng = np.random.default_rng(0)
    mask = rng.random(X.shape) < 0.05
    X[mask] = np.nan

    model = Aegis5(
        use_feature_selection=False,
        use_pca=False,
        use_smote=False,
        random_state=0,
    )
    model.fit(X[:150], y[:150])
    preds = model.predict(X[150:])
    assert len(preds) == 50
    assert not any(np.isnan(p) for p in preds)


def test_get_dynamic_weights_returns_dataframe():
    X, y = _make_dataset(n_samples=200, n_classes=3)
    model = Aegis5(
        use_feature_selection=False,
        use_pca=False,
        use_smote=False,
        random_state=0,
    )
    model.fit(X[:150], y[:150])
    df = model.get_dynamic_weights()
    assert df is not None
    assert df.shape[0] == 5  # 5 classifiers
    assert df.shape[1] == 3  # 3 classes


def test_binary_classification():
    X, y = _make_dataset(n_samples=200, n_classes=2)
    model = Aegis5(
        use_feature_selection=False,
        use_pca=False,
        use_smote=False,
        random_state=0,
    )
    model.fit(X[:150], y[:150])
    results = model.evaluate(X[150:], y[150:])
    assert results['accuracy'] > 0.70


def test_reproducible_with_same_seed():
    X, y = _make_dataset(n_samples=200, n_classes=2)
    model_a = Aegis5(use_feature_selection=False, use_pca=False, use_smote=False, random_state=42)
    model_b = Aegis5(use_feature_selection=False, use_pca=False, use_smote=False, random_state=42)

    model_a.fit(X[:150], y[:150])
    model_b.fit(X[:150], y[:150])

    preds_a = model_a.predict(X[150:])
    preds_b = model_b.predict(X[150:])
    np.testing.assert_array_equal(preds_a, preds_b)
