import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import optuna

from fairness_metrics import FairnessEvaluator


class HyperparameterTuner:
    def __init__(self, random_state: int = 42):
        from hyperparameter_tuning import HyperparameterTuner as OldTuner
        self._old = OldTuner(random_state=random_state)

    def comprehensive_tuning(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                             sensitive_attrs: Dict[str, pd.Series], methods: List[str] = ('bayesian',),
                             fairness_aware: bool = True, n_trials: Optional[int] = None, cv: int = 5) -> Dict[str, Any]:
        return self._old.comprehensive_tuning(model, X, y, sensitive_attrs, methods, fairness_aware, n_trials, cv)

    def get_tuning_summary(self) -> pd.DataFrame:
        return self._old.get_tuning_summary()


