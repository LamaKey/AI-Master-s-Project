import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from fairness_metrics import FairnessEvaluator

import optuna

class FairnessAwareScorer:
    
    def __init__(self, performance_weight=0.7, fairness_weight=0.3,
                    fairness_metric='intersectional',
                    sensitive_attr_name='sex',
                    performance_metric='accuracy'):
        self.performance_weight = performance_weight
        self.fairness_weight = fairness_weight
        self.fairness_metric = fairness_metric
        self.sensitive_attr_name = sensitive_attr_name
        self.evaluator = FairnessEvaluator()
        self.performance_metric = performance_metric

    def __call__(self, estimator, X, y, sensitive_attrs=None):
            import numpy as np
            from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

            y_pred = estimator.predict(X)
            if self.performance_metric == 'balanced_accuracy':
                performance_score = balanced_accuracy_score(y, y_pred)
            elif self.performance_metric == 'f1':
                performance_score = f1_score(y, y_pred, zero_division=0)
            else:
                performance_score = accuracy_score(y, y_pred)
            if not np.isfinite(performance_score):
                return 0.0

            fairness_score = 1.0
            if sensitive_attrs:
                if self.fairness_metric == 'intersectional':
                    if len(sensitive_attrs) >= 2:
                        inter = self.evaluator.intersectional_fairness(
                            np.asarray(y).ravel(), y_pred,
                            {k: np.asarray(v).ravel() for k, v in sensitive_attrs.items()}
                        )
                        disp = inter.get('intersectional_disparities', {}).get('max_disparity', 0.0)
                        fairness_score = 1.0 - float(np.clip(disp, 0.0, 1.0))
                    else:
                        fairness_score = 0.5
                elif self.fairness_metric in ('demographic_parity', 'equalized_odds'):
                    if self.sensitive_attr_name in sensitive_attrs:
                        s = np.asarray(sensitive_attrs[self.sensitive_attr_name]).ravel()
                        if len(s) == len(y_pred):
                            if self.fairness_metric == 'demographic_parity':
                                dp = self.evaluator.demographic_parity(y_pred, s)
                                disparity = dp['demographic_parity_difference']
                            else:
                                eo = self.evaluator.equalized_odds(np.asarray(y).ravel(), y_pred, s)
                                disparity = eo['equalized_odds_difference']
                            fairness_score = float(np.exp(-5.0 * abs(disparity)))

            combined = self.performance_weight * performance_score + self.fairness_weight * fairness_score
            return float(combined) if np.isfinite(combined) else 0.0

class HyperparameterTuner:
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_params_ = {}
        self.best_score_ = {}
        self.cv_results_ = {}
        self.tuning_history = []
    
    def _resolve_logreg_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        resolved = dict(params)
        penalty = resolved.get('penalty')
        solver = resolved.get('solver')

        if penalty == 'elasticnet':
            resolved['solver'] = 'saga'
        elif penalty == 'l1' and solver == 'lbfgs':
            resolved['solver'] = 'liblinear'

        if penalty != 'elasticnet' and 'l1_ratio' in resolved:
            resolved.pop('l1_ratio', None)

        return resolved
    
    def get_logistic_regression_param_grid(self):
        return {
            'C': (0.001, 1000.0),
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['lbfgs', 'liblinear', 'saga'],
            'max_iter': (500, 10000),
            'class_weight': [None, 'balanced']
        }
    def comprehensive_tuning(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_attrs: Dict[str, np.ndarray],
        methods: List[str] = ('bayesian',),
        fairness_aware: bool = True,
        n_trials: Optional[int] = None,
        cv: int = 5
    ) -> Dict[str, Any]:
        all_results = {}
        print(f"Starting comprehensive hyperparameter tuning for {type(model).__name__}")
        print(f"Methods: {methods}")
        print(f"Fairness-aware: {fairness_aware}")
        print("=" * 60)

        if 'bayesian' in methods:
            print("[Tuning] Running Bayesian Optimization...")
            bayes_results = self.bayesian_optimization(
                model, X, y, sensitive_attrs,
                n_trials=n_trials, cv=cv, fairness_aware=fairness_aware
            )
            all_results['bayesian_optimization'] = bayes_results
            print(f"Bayesian Optimization completed. Best score: {bayes_results['best_score']:.4f}")
        overall_best = all_results['bayesian_optimization']

        print(f"\nSELECTED METHOD: bayesian_optimization")
        print(f"BEST SCORE: {overall_best['best_score']:.4f}")
        print(f"BEST PARAMS: {overall_best['best_params']}")

        return {
            'all_results': all_results,
            'best_method': 'bayesian_optimization',
            'best_overall': overall_best
        }


    def get_tuning_summary(self) -> pd.DataFrame:
        rows = []
        for entry in self.tuning_history:
            model = entry.get('model', '')
            method = entry.get('method', '')
            res = entry.get('results', {}) or {}
            rows.append({
                'Model': model,
                'Method': method,
                'Best Score': float(res.get('best_score', float('nan'))),
                'Best Params': res.get('best_params', {})
            })
        if not rows:
            return pd.DataFrame(columns=['Model', 'Method', 'Best Score', 'Best Params'])
        df = pd.DataFrame(rows)
        with pd.option_context('mode.use_inf_as_na', True):
            df = df.sort_values(by='Best Score', ascending=False)
        return df.reset_index(drop=True)

    def bayesian_optimization(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                            sensitive_attrs: Dict[str, np.ndarray],
                            n_trials: int = 100, cv: int = 5,
                            fairness_aware: bool = False) -> Dict[str, Any]:
        
        if fairness_aware:
            scorer = FairnessAwareScorer(fairness_metric='intersectional',
                                        performance_weight=0.7, fairness_weight=0.3)
        
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.001, 1000.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'saga']),
                'max_iter': trial.suggest_int('max_iter', 500, 10000),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
            }
            
            if params['penalty'] == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
            
            resolved_params = self._resolve_logreg_params(params)
            trial.set_user_attr('resolved_params', dict(resolved_params))
            
            model_instance = type(model)(**resolved_params, random_state=self.random_state)
            
            cv_scores = []
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold = X.iloc[train_idx]
                X_val_fold   = X.iloc[val_idx]

                y_train_fold = y.iloc[train_idx]
                y_val_fold   = y.iloc[val_idx]
                
                model_instance.fit(X_train_fold, y_train_fold)
                
                if fairness_aware:
                    sub_attrs = {k: v.iloc[val_idx] if hasattr(v,"iloc") else v[val_idx] for k,v in sensitive_attrs.items()}
                    score = scorer(model_instance, X_val_fold, y_val_fold, sub_attrs)
                else:
                    score = accuracy_score(y_val_fold, model_instance.predict(X_val_fold))
                
                cv_scores.append(score)
            
            return np.mean(cv_scores)
        
        if n_trials is None:
            n_trials = int(os.getenv("OPTUNA_TRIALS", "60"))
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_raw_params = study.best_params
        best_resolved_params = study.best_trial.user_attrs.get(
            'resolved_params', self._resolve_logreg_params(best_raw_params)
        )
        best_model = type(model)(**best_resolved_params, random_state=self.random_state)
        best_model.fit(X, y)
        
        model_name = type(model).__name__
        self.best_params_[model_name] = best_resolved_params
        self.best_score_[model_name] = study.best_value
        
        results = {
            'best_params': best_resolved_params,
            'best_score': study.best_value,
            'best_estimator': best_model,
            'study': study,
            'method': 'bayesian_optimization'
        }
        
        self.tuning_history.append({
            'model': model_name,
            'method': 'bayesian_optimization',
            'results': results
        })
        
        return results