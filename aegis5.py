"""
Aegis-5: A Hybrid Ensemble Framework for Intrusion Detection
in Industry 5.0 Driven Smart Manufacturing Environment

Published in ACM Transactions on Autonomous and Adaptive Systems, 2026
DOI: 10.1145/3787224

This module implements the Aegis-5 adaptive hybrid ensemble framework
integrating five classifiers with dynamic weighting and meta-learning.
"""

import numpy as np
import pandas as pd
from collections import deque
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import f_classif, RFECV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


class DynamicWeightManager:
    """
    Manages dynamic per-class weights for each base classifier using a
    sliding window of recent predictions. Weights are computed via softmax
    over per-class F1-scores (Eq. 2 in the paper).

    Parameters
    ----------
    n_classifiers : int
        Number of base classifiers (5 for Aegis-5).
    n_classes : int
        Number of target classes.
    window_size : int
        Size of the sliding window for tracking predictions (K=1000).
    beta : float
        Temperature parameter controlling weight sharpness (beta=2.0).
    """

    def __init__(self, n_classifiers, n_classes, window_size=1000, beta=2.0):
        self.n_classifiers = n_classifiers
        self.n_classes = n_classes
        self.window_size = window_size
        self.beta = beta
        # Sliding windows: store (true_label, predicted_label) per classifier
        self.windows = [deque(maxlen=window_size) for _ in range(n_classifiers)]
        # Initialize weights uniformly
        self.weights = np.ones((n_classifiers, n_classes)) / n_classifiers

    def update(self, classifier_idx, y_true, y_pred):
        """Record a batch of predictions for a classifier and recompute weights."""
        for yt, yp in zip(y_true, y_pred):
            self.windows[classifier_idx].append((yt, yp))
        self._recompute_weights()

    def _recompute_weights(self):
        """Recompute weights using softmax over per-class F1-scores."""
        f1_scores = np.ones((self.n_classifiers, self.n_classes)) * 0.5  # prior

        for i in range(self.n_classifiers):
            if len(self.windows[i]) == 0:
                continue
            records = list(self.windows[i])
            y_true = [r[0] for r in records]
            y_pred = [r[1] for r in records]
            per_class_f1 = f1_score(
                y_true, y_pred, average=None,
                labels=list(range(self.n_classes)), zero_division=0.0
            )
            f1_scores[i, :len(per_class_f1)] = per_class_f1

        # Softmax with temperature beta (Eq. 2)
        for c in range(self.n_classes):
            scores = self.beta * f1_scores[:, c]
            exp_scores = np.exp(scores - np.max(scores))  # numerical stability
            self.weights[:, c] = exp_scores / exp_scores.sum()

    def get_weights(self):
        """Return current weight matrix (n_classifiers x n_classes)."""
        return self.weights.copy()


class Aegis5:
    """
    Aegis-5: Adaptive Ensemble for Guarding Industrial Systems in Industry 5.0.

    Integrates five base classifiers (RF, GBM, XGBoost, SVM, KNN) with:
    - Dynamic per-class weighting based on sliding-window F1-scores
    - Meta-learner (Logistic Regression) for prediction synthesis
    - Hybrid voting protocol (soft voting at high confidence, hard voting otherwise)

    Parameters
    ----------
    confidence_threshold : float
        Threshold tau for hybrid voting (default 0.95).
    beta : float
        Temperature for dynamic weight softmax (default 2.0).
    window_size : int
        Sliding window size K for weight updates (default 1000).
    use_feature_selection : bool
        Whether to apply ANOVA F-test + RFECV feature selection.
    use_pca : bool
        Whether to apply PCA after feature selection.
    pca_variance : float
        Variance ratio to retain with PCA (default 0.987).
    use_smote : bool
        Whether to apply SMOTE for class imbalance (default True).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        confidence_threshold=0.95,
        beta=2.0,
        window_size=1000,
        use_feature_selection=True,
        use_pca=True,
        pca_variance=0.987,
        use_smote=True,
        random_state=42
    ):
        self.confidence_threshold = confidence_threshold
        self.beta = beta
        self.window_size = window_size
        self.use_feature_selection = use_feature_selection
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.use_smote = use_smote
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = None
        self.feature_selector = None
        self.weight_manager = None
        self.meta_learner = None
        self.n_classes = None
        self.classes_ = None

        # Base classifiers with tuned hyperparameters (Table 4)
        self.base_classifiers = {
            'RF': RandomForestClassifier(
                n_estimators=200, max_depth=25, min_samples_split=5,
                random_state=random_state, n_jobs=-1
            ),
            'GBM': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=7,
                random_state=random_state
            ),
            'XGBoost': XGBClassifier(
                learning_rate=0.05, max_depth=10, subsample=0.8,
                colsample_bytree=0.8, random_state=random_state,
                use_label_encoder=False, eval_metric='mlogloss',
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf', C=10, gamma='scale',
                probability=True, random_state=random_state
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5, weights='distance', metric='minkowski',
                n_jobs=-1
            )
        }

        # Meta-learner (Logistic Regression)
        self.meta_learner = LogisticRegression(
            solver='lbfgs', C=1.0, penalty='l2',
            max_iter=1000, random_state=random_state, n_jobs=-1,
            multi_class='multinomial'
        )

    def _preprocess(self, X, y=None, fit=False):
        """
        Preprocessing pipeline:
        1. Handle missing values (median imputation)
        2. Feature scaling (StandardScaler)
        3. Feature selection (ANOVA F-test + RFECV) [optional]
        4. PCA dimensionality reduction [optional]
        5. SMOTE for class imbalance [optional, training only]
        """
        X = X.copy()

        # Median imputation
        if fit:
            self._medians = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            for col in range(X.shape[1]):
                X[nan_mask[:, col], col] = self._medians[col]

        # Feature scaling
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        # Feature selection with ANOVA F-test + RFECV
        if self.use_feature_selection and fit and y is not None:
            estimator = RandomForestClassifier(
                n_estimators=50, max_depth=10, random_state=self.random_state, n_jobs=-1
            )
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            self.feature_selector = RFECV(
                estimator=estimator, step=1, cv=cv, scoring='f1_weighted',
                min_features_to_select=10, n_jobs=-1
            )
            self.feature_selector.fit(X, y)
            X = self.feature_selector.transform(X)
            print(f"  Features selected by RFECV: {self.feature_selector.n_features_} / {self.feature_selector.n_features_in_}")
        elif self.use_feature_selection and self.feature_selector is not None:
            X = self.feature_selector.transform(X)

        # PCA
        if self.use_pca and fit:
            self.pca = PCA(n_components=self.pca_variance, random_state=self.random_state)
            X = self.pca.fit_transform(X)
            print(f"  PCA components: {self.pca.n_components_} (variance retained: {sum(self.pca.explained_variance_ratio_):.4f})")
        elif self.use_pca and self.pca is not None:
            X = self.pca.transform(X)

        # SMOTE (training only)
        if self.use_smote and fit and y is not None:
            smote = SMOTE(random_state=self.random_state)
            X, y = smote.fit_resample(X, y)
            print(f"  After SMOTE: {X.shape[0]} samples")

        return (X, y) if (fit and y is not None) else X

    def fit(self, X, y):
        """
        Train the Aegis-5 ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.
        """
        print("=== Aegis-5 Training ===")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        self.n_classes = len(self.classes_)

        print(f"Classes ({self.n_classes}): {list(self.classes_)}")
        print(f"Training samples: {X.shape[0]}, Features: {X.shape[1]}")

        # Preprocess
        print("\nPreprocessing...")
        X_processed, y_processed = self._preprocess(X, y_encoded, fit=True)

        # Split for meta-learner training (80/20)
        X_base, X_meta, y_base, y_meta = train_test_split(
            X_processed, y_processed, test_size=0.2,
            stratify=y_processed, random_state=self.random_state
        )

        # Initialize dynamic weight manager
        self.weight_manager = DynamicWeightManager(
            n_classifiers=len(self.base_classifiers),
            n_classes=self.n_classes,
            window_size=self.window_size,
            beta=self.beta
        )

        # Train base classifiers
        print("\nTraining base classifiers...")
        classifier_names = list(self.base_classifiers.keys())
        for i, (name, clf) in enumerate(self.base_classifiers.items()):
            print(f"  Training {name}...")
            clf.fit(X_base, y_base)

            # Update dynamic weights using validation predictions
            y_pred = clf.predict(X_meta)
            self.weight_manager.update(i, y_meta, y_pred)

            acc = accuracy_score(y_meta, y_pred)
            f1 = f1_score(y_meta, y_pred, average='weighted')
            print(f"    Validation - Accuracy: {acc:.4f}, F1: {f1:.4f}")

        # Build meta-features for meta-learner training
        print("\nTraining meta-learner (Logistic Regression)...")
        meta_features = self._build_meta_features(X_meta)

        # Apply dynamic weights to meta-features
        weights = self.weight_manager.get_weights()
        meta_features_weighted = self._apply_dynamic_weights(meta_features, y_meta, weights)

        self.meta_learner.fit(meta_features_weighted, y_meta)
        meta_pred = self.meta_learner.predict(meta_features_weighted)
        meta_acc = accuracy_score(y_meta, meta_pred)
        meta_f1 = f1_score(y_meta, meta_pred, average='weighted')
        print(f"  Meta-learner Validation - Accuracy: {meta_acc:.4f}, F1: {meta_f1:.4f}")

        print("\n=== Training Complete ===")
        return self

    def _build_meta_features(self, X):
        """
        Build meta-feature matrix from base classifier probability outputs.
        Each classifier contributes n_classes probability columns.
        """
        meta = []
        for clf in self.base_classifiers.values():
            proba = clf.predict_proba(X)
            # Ensure consistent shape
            if proba.shape[1] < self.n_classes:
                padded = np.zeros((X.shape[0], self.n_classes))
                padded[:, :proba.shape[1]] = proba
                proba = padded
            meta.append(proba)
        return np.hstack(meta)

    def _apply_dynamic_weights(self, meta_features, y_true, weights):
        """
        Apply dynamic per-class weights to meta-features.
        For each sample, scale the classifier's probability outputs by
        the weight for the predicted class.
        """
        weighted = meta_features.copy()
        n_clf = len(self.base_classifiers)

        for i in range(n_clf):
            start = i * self.n_classes
            end = start + self.n_classes
            # Weight by per-class weights for each classifier
            for c in range(self.n_classes):
                weighted[:, start + c] *= weights[i, c]

        return weighted

    def predict(self, X):
        """
        Predict class labels using the hybrid voting protocol (Algorithm 1).

        High-confidence predictions (P >= tau) use soft voting via meta-learner.
        Low-confidence predictions fall back to hard voting among base classifiers.
        """
        X_processed = self._preprocess(X)
        meta_features = self._build_meta_features(X_processed)

        weights = self.weight_manager.get_weights()
        meta_features_weighted = self._apply_dynamic_weights(
            meta_features, None, weights
        )

        # Meta-learner probabilities
        meta_proba = self.meta_learner.predict_proba(meta_features_weighted)

        predictions = np.zeros(X_processed.shape[0], dtype=int)

        for idx in range(X_processed.shape[0]):
            p_max = np.max(meta_proba[idx])
            c_max = np.argmax(meta_proba[idx])

            if p_max >= self.confidence_threshold:
                # High confidence: soft voting (meta-learner decision)
                predictions[idx] = c_max
            else:
                # Low confidence: hard voting (majority vote from base classifiers)
                votes = []
                for clf in self.base_classifiers.values():
                    votes.append(clf.predict(X_processed[idx:idx+1])[0])
                predictions[idx] = np.bincount(votes, minlength=self.n_classes).argmax()

        return self.label_encoder.inverse_transform(predictions)

    def predict_proba(self, X):
        """Return meta-learner probability estimates."""
        X_processed = self._preprocess(X)
        meta_features = self._build_meta_features(X_processed)
        weights = self.weight_manager.get_weights()
        meta_features_weighted = self._apply_dynamic_weights(
            meta_features, None, weights
        )
        return self.meta_learner.predict_proba(meta_features_weighted)

    def evaluate(self, X, y):
        """
        Evaluate the model and print detailed metrics.

        Returns dict with accuracy, precision, recall, f1, and classification report.
        """
        y_pred = self.predict(X)

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

        print("\n=== Aegis-5 Evaluation Results ===")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"\nClassification Report:\n")
        print(classification_report(y, y_pred, zero_division=0))

        cm = confusion_matrix(y, y_pred)
        print(f"Confusion Matrix:\n{cm}")

        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred
        }

    def get_dynamic_weights(self):
        """Return the current dynamic weight matrix with classifier and class labels."""
        if self.weight_manager is None:
            return None
        weights = self.weight_manager.get_weights()
        return pd.DataFrame(
            weights,
            index=list(self.base_classifiers.keys()),
            columns=self.classes_
        )
