"""
model.py
--------
Defines the local ML model used by every node.
Uses logistic regression (SGDClassifier) intentionally kept small so the
simulation runs in seconds on CPU with no GPU dependency.
"""

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss


# ---------------------------------------------------------------------------
# Constants shared across the project
# ---------------------------------------------------------------------------
N_FEATURES = 20       # synthetic dataset feature count
N_CLASSES  = 2        # binary classification
WEIGHT_CLIP = 10.0    # clip weights to [-WEIGHT_CLIP, WEIGHT_CLIP]


class LogisticModel:
    """
    Thin wrapper around sklearn SGDClassifier providing:
      • get_weights() / set_weights() for federated aggregation
      • gradient clipping to prevent explosion from malicious updates
      • NaN / Inf sanitisation on weight setting
    """

    def __init__(self, n_features: int = N_FEATURES, random_state: int = 42):
        self.n_features = n_features
        self.classes_   = np.array([0, 1])

        self.clf = SGDClassifier(
            loss        = "log_loss",
            alpha       = 0.01,       # L2 regularisation
            max_iter    = 1,
            tol         = None,
            warm_start  = True,
            random_state= random_state,
        )

        # --- Force-initialise internal arrays so set_weights works immediately ---
        self.clf.coef_      = np.zeros((1, n_features))
        self.clf.intercept_ = np.zeros(1)
        self.clf.classes_   = self.classes_
        self._fitted        = True          # pretend we've done at least one fit

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 3) -> None:
        """Run `epochs` passes of SGD on the local data."""
        for _ in range(epochs):
            self.clf.partial_fit(X, y, classes=self.classes_)
        self._clip_weights()

    # ------------------------------------------------------------------
    # Inference / evaluation
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.clf.score(X, y))

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        try:
            proba = self.clf.predict_proba(X)
            return float(log_loss(y, proba))
        except Exception:
            return 1.0

    # ------------------------------------------------------------------
    # Weight serialisation (used by P2P / FL aggregation)
    # ------------------------------------------------------------------
    def get_weights(self) -> np.ndarray:
        """Return a flat copy of [coef | intercept]."""
        coef      = self.clf.coef_.flatten()
        intercept = self.clf.intercept_.flatten()
        return np.concatenate([coef, intercept])

    def set_weights(self, weights: np.ndarray) -> None:
        """
        Accept a flat weight vector.
        Validates shape, sanitises NaN/Inf, then clips to WEIGHT_CLIP.
        """
        expected = self.n_features + 1
        if len(weights) != expected:
            raise ValueError(
                f"Weight shape mismatch: expected {expected}, got {len(weights)}"
            )
        # Sanitise
        if not np.all(np.isfinite(weights)):
            weights = np.nan_to_num(
                weights, nan=0.0, posinf=WEIGHT_CLIP, neginf=-WEIGHT_CLIP
            )
        weights = np.clip(weights, -WEIGHT_CLIP, WEIGHT_CLIP)
        self.clf.coef_      = weights[: self.n_features].reshape(1, self.n_features)
        self.clf.intercept_ = weights[self.n_features :].reshape(1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _clip_weights(self) -> None:
        self.clf.coef_      = np.clip(self.clf.coef_,      -WEIGHT_CLIP, WEIGHT_CLIP)
        self.clf.intercept_ = np.clip(self.clf.intercept_, -WEIGHT_CLIP, WEIGHT_CLIP)

    def weight_dim(self) -> int:
        """Total number of scalar parameters."""
        return self.n_features + 1

    def copy_weights_to(self, other: "LogisticModel") -> None:
        other.set_weights(self.get_weights().copy())
