import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from typing import Any
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


class BiasInprocessor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _decorrelate_features(self, X: np.ndarray, sensitive_attr: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X_decorr = X.copy()
        s = LabelEncoder().fit_transform(sensitive_attr).astype(float)
        s = (s - s.mean()) / (s.std() + 1e-8)
        denom = np.dot(s, s) + 1e-8
        for j in range(X.shape[1]):
            beta = np.dot(s, X[:, j]) / denom
            X_decorr[:, j] = X[:, j] - beta * s
        return X_decorr

    def adversarial_debiasing(self, X: np.ndarray, y: np.ndarray, sensitive_attr: np.ndarray):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed; adversarial debiasing is unavailable. "
                "Install torch or select 'fairlearn_eg' instead."
            )
        return self._train_adversarial_model(X, y, sensitive_attr)

    def _train_adversarial_model(self, X: np.ndarray, y: np.ndarray, sensitive_attr: np.ndarray):
        if not _TORCH_AVAILABLE:
            raise ImportError("Adversarial training requested but torch is unavailable.")

        try:
            if torch.cuda.is_available() and hasattr(torch, 'use_deterministic_algorithms'):
                torch.use_deterministic_algorithms(True)
        except Exception:
            pass

        X_array = np.array(X)
        y_array = np.array(y)
        sensitive_array = np.array(sensitive_attr)

        X_tensor = torch.FloatTensor(X_array)
        y_tensor = torch.FloatTensor(y_array).unsqueeze(1)

        le = LabelEncoder()
        sensitive_encoded = le.fit_transform(sensitive_array)
        sensitive_tensor = torch.LongTensor(sensitive_encoded)

        n_features = X_array.shape[1]
        n_sensitive_classes = len(np.unique(sensitive_encoded))
        hidden_dim = min(64, max(16, n_features // 2))

        class Predictor(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.shared_layer = nn.Linear(input_dim, hidden_dim)
                self.classifier = nn.Linear(hidden_dim, 1)
                self.dropout = nn.Dropout(0.2)
            def forward(self, x):
                hidden = torch.relu(self.shared_layer(x))
                hidden = self.dropout(hidden)
                pred = torch.sigmoid(self.classifier(hidden))
                return pred, hidden

        class Discriminator(nn.Module):
            def __init__(self, hidden_dim, n_classes):
                super().__init__()
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim // 2, n_classes)
                )
            def forward(self, hidden):
                return self.classifier(hidden)

        predictor = Predictor(n_features, hidden_dim)
        discriminator = Discriminator(hidden_dim, n_sensitive_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predictor.to(device); discriminator.to(device)

        pred_optimizer = optim.Adam(predictor.parameters(), lr=1e-3, weight_decay=1e-4)
        disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3, weight_decay=1e-4)

        pred_criterion = nn.BCELoss()
        disc_criterion = nn.CrossEntropyLoss()

        n_epochs = 60
        lambda_adv = 0.1
        max_grad_norm = 5.0

        predictor.train(); discriminator.train()
        dataset = TensorDataset(X_tensor, y_tensor, sensitive_tensor)
        loader  = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False)

        best_pred_loss = float("inf"); patience = 6; bad = 0
        for epoch in range(n_epochs):
            epoch_pred_loss = 0.0; batches = 0
            for xb, yb, sb in loader:
                xb = xb.to(device); yb = yb.to(device); sb = sb.to(device)
                disc_optimizer.zero_grad()
                with torch.no_grad():
                    _, hidden = predictor(xb)
                disc_pred = discriminator(hidden)
                disc_loss = disc_criterion(disc_pred, sb)
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm)
                disc_optimizer.step()

                pred_optimizer.zero_grad()
                y_hat, hidden = predictor(xb)
                pred_loss = pred_criterion(y_hat, yb)
                disc_pred = discriminator(hidden)
                adv_loss  = -disc_criterion(disc_pred, sb)
                total_loss = pred_loss + lambda_adv * adv_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_grad_norm)
                pred_optimizer.step()
                epoch_pred_loss += pred_loss.item(); batches += 1

            mean_pred = epoch_pred_loss / max(batches, 1)
            if mean_pred < best_pred_loss - 1e-4:
                best_pred_loss = mean_pred; bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        class AdversarialModel:
            def __init__(self, predictor_net, label_encoder):
                self.predictor = predictor_net
                self.label_encoder = label_encoder
                self.predictor.eval()
                self.device = next(self.predictor.parameters()).device
            def predict(self, X):
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(np.array(X)).to(self.device)
                    pred, _ = self.predictor(X_tensor)
                    return (pred.detach().cpu().numpy() > 0.5).astype(int).flatten()
            def predict_proba(self, X):
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(np.array(X)).to(self.device)
                    pred, _ = self.predictor(X_tensor)
                    p = pred.detach().cpu().numpy().flatten()
                    return np.column_stack([1 - p, p])
            def get_params(self, deep: bool = False) -> dict:
                return {}
            def set_params(self, **params: Any):
                return self

        return AdversarialModel(predictor, le)

    def fairlearn_exponentiated_gradient(self, X: np.ndarray=None, y: np.ndarray=None,
                                       sensitive_attr: np.ndarray=None,
                                       constraint: str = 'demographic_parity') -> object:
        if constraint == 'demographic_parity':
            fairness_constraint = DemographicParity()
        elif constraint == 'equalized_odds':
            fairness_constraint = EqualizedOdds()
        else:
            raise ValueError("Constraint must be 'demographic_parity' or 'equalized_odds'")

        class FairlearnWrapper(BaseEstimator):
            def __init__(self, mitigator=None, constraint_type='equalized_odds', eps=0.03, max_iter=100, random_state=0):
                self.mitigator = mitigator
                self.constraint_type = constraint_type
                self.eps = eps
                self.max_iter = max_iter
                self.random_state = random_state
            def fit(self, X, y, sensitive_features=None):
                if self.constraint_type == 'demographic_parity':
                    fairness_constraint = DemographicParity()
                else:
                    fairness_constraint = EqualizedOdds()
                base = LogisticRegression(random_state=self.random_state, max_iter=1000)
                self.mitigator = ExponentiatedGradient(
                    estimator=base,
                    constraints=fairness_constraint,
                    eps=self.eps,
                    max_iter=self.max_iter
                )
                self.mitigator.fit(X, y, sensitive_features=sensitive_features)
                return self
            def predict(self, X):
                return np.asarray(self.mitigator.predict(X)).astype(int)
            def predict_proba(self, X):
                import numpy as _np
                n = _np.asarray(X).shape[0]
                try:
                    p1 = _np.asarray(self.mitigator._pmf_predict(X), dtype=float).reshape(-1)
                except Exception:
                    preds = _np.asarray(self.mitigator.predict(X)).astype(float).reshape(-1)
                    p1 = preds
                if p1.shape[0] != n:
                    if p1.ndim == 2 and p1.shape[0] == 2 and p1.shape[1] == n:
                        p1 = p1[1, :]
                    else:
                        p1 = p1.ravel()[:n]
                proba = _np.column_stack([1.0 - p1, p1])
                return _np.clip(proba, 0.0, 1.0)
            def get_params(self, deep=False):
                return {
                    'mitigator': self.mitigator,
                    'constraint_type': self.constraint_type,
                    'eps': self.eps,
                    'max_iter': self.max_iter,
                    'random_state': self.random_state
                }
            def set_params(self, **params):
                for k, v in params.items():
                    setattr(self, k, v)
                return self

        return FairlearnWrapper(
            mitigator=None,
            constraint_type=constraint,
            eps=0.03,
            max_iter=100,
            random_state=self.random_state
        )


