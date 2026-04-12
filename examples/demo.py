"""
Demo script for the Aegis-5 framework.

Generates a synthetic IIoT intrusion detection dataset that mimics the
class distribution and feature characteristics of IoT-23, then trains
and evaluates the Aegis-5 ensemble.

Usage:
    python demo.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from aegis5 import Aegis5


def generate_synthetic_iiot_data(n_samples=10000, n_features=25, random_state=42):
    """
    Generate a synthetic IIoT dataset mimicking IoT-23 class distribution.

    Classes and approximate distribution (from the paper):
    - Benign: 54.3%
    - PartOfAHorizontalPortScan: 25.8%
    - C&C: 11.7%
    - Attack: 8.0%
    - Okiru: 0.34%
    - DDoS: 0.075%
    - FileDownload: 0.004%
    """
    rng = np.random.RandomState(random_state)

    classes = ['Benign', 'PortScan', 'C&C', 'Attack', 'Okiru', 'DDoS', 'FileDownload']
    proportions = [0.543, 0.258, 0.117, 0.06, 0.012, 0.008, 0.002]

    X_all, y_all = [], []

    for cls, prop in zip(classes, proportions):
        n = max(int(n_samples * prop), 10)  # at least 10 samples per class

        if cls == 'Benign':
            # Normal traffic: low variance, centered features
            X = rng.normal(loc=0.0, scale=0.5, size=(n, n_features))
            # Benign traffic has low byte counts, regular duration
            X[:, 0] = rng.exponential(50, n)     # orig_bytes
            X[:, 1] = rng.exponential(60, n)     # resp_bytes
            X[:, 2] = rng.exponential(1.0, n)    # duration
        elif cls == 'PortScan':
            # Port scan: many short connections, high SYN count
            X = rng.normal(loc=1.5, scale=0.8, size=(n, n_features))
            X[:, 0] = rng.exponential(10, n)     # small packets
            X[:, 1] = rng.exponential(5, n)
            X[:, 2] = rng.exponential(0.01, n)   # very short duration
            X[:, 3] = rng.poisson(50, n)          # high SYN count
        elif cls == 'C&C':
            # Command & Control: periodic beaconing, moderate bytes
            X = rng.normal(loc=-1.0, scale=1.0, size=(n, n_features))
            X[:, 0] = rng.exponential(200, n)
            X[:, 1] = rng.exponential(150, n)
            X[:, 2] = rng.exponential(30, n)      # long duration (beaconing)
            X[:, 4] = rng.normal(5.0, 0.5, n)     # regular interval
        elif cls == 'Attack':
            # General attacks: high variance, anomalous patterns
            X = rng.normal(loc=2.0, scale=1.5, size=(n, n_features))
            X[:, 0] = rng.exponential(500, n)     # large payloads
            X[:, 1] = rng.exponential(100, n)
        elif cls == 'Okiru':
            # Okiru malware: specific botnet signatures
            X = rng.normal(loc=-2.0, scale=0.6, size=(n, n_features))
            X[:, 0] = rng.exponential(80, n)
            X[:, 5] = rng.poisson(100, n)          # high packet rate
        elif cls == 'DDoS':
            # DDoS: extremely high volume, short bursts
            X = rng.normal(loc=3.0, scale=2.0, size=(n, n_features))
            X[:, 0] = rng.exponential(10000, n)   # massive bytes
            X[:, 1] = rng.exponential(100, n)
            X[:, 2] = rng.exponential(0.001, n)   # very short
            X[:, 3] = rng.poisson(200, n)          # flood SYN
        elif cls == 'FileDownload':
            # File download: large response bytes, single connection
            X = rng.normal(loc=0.5, scale=0.3, size=(n, n_features))
            X[:, 0] = rng.exponential(100, n)
            X[:, 1] = rng.exponential(50000, n)   # huge response
            X[:, 2] = rng.exponential(10, n)

        X_all.append(X)
        y_all.extend([cls] * n)

    X_all = np.vstack(X_all)
    y_all = np.array(y_all)

    # Shuffle
    idx = rng.permutation(len(y_all))
    return X_all[idx], y_all[idx]


def main():
    print("=" * 60)
    print("  Aegis-5: Intrusion Detection Framework Demo")
    print("  Synthetic IIoT Dataset (mimicking IoT-23)")
    print("=" * 60)

    # Generate synthetic data
    print("\nGenerating synthetic IIoT dataset...")
    X, y = generate_synthetic_iiot_data(n_samples=5000, n_features=25)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {cls}: {cnt} ({cnt/len(y)*100:.1f}%)")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Initialize and train Aegis-5
    model = Aegis5(
        confidence_threshold=0.95,
        beta=2.0,
        window_size=1000,
        use_feature_selection=False,  # skip RFECV for demo speed
        use_pca=False,                # skip PCA for demo speed
        use_smote=True,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    results = model.evaluate(X_test, y_test)

    # Show dynamic weights
    print("\n=== Dynamic Classifier Weights (per class) ===")
    weights_df = model.get_dynamic_weights()
    if weights_df is not None:
        print(weights_df.round(3).to_string())

    print("\n=== Demo Complete ===")
    return results


if __name__ == '__main__':
    main()
