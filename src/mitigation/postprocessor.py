import numpy as np
from sklearn.metrics import roc_curve
from fairlearn.postprocessing import ThresholdOptimizer
import warnings
warnings.filterwarnings('ignore')


class BiasPostprocessor:
    def __init__(self):
        self.thresholds = {}
        self.calibration_factors_ = None

    def fit_group_calibration(self, y_true, y_prob, sensitive_attr):
        factors = {}
        y_true = np.asarray(y_true).ravel()
        y_prob = np.asarray(y_prob).ravel()
        sens   = np.asarray(sensitive_attr).ravel()
        for g in np.unique(sens):
            m = (sens == g)
            if m.any():
                actual = float(y_true[m].mean())
                pred   = float(y_prob[m].mean())
                factors[g] = 1.0 if pred <= 1e-12 else float(actual / pred)
        self.calibration_factors_ = factors
        return factors

    def apply_group_calibration(self, y_prob, sensitive_attr, factors=None,
                                threshold=0.5, clip=True, return_proba=False):
        y_prob = np.asarray(y_prob).ravel()
        sens   = np.asarray(sensitive_attr).ravel()
        if factors is None:
            if self.calibration_factors_ is None:
                raise ValueError("Call fit_group_calibration() first or pass factors=...")
            factors = self.calibration_factors_
        adjusted = y_prob.copy()
        for g, f in factors.items():
            m = (sens == g)
            if m.any():
                adjusted[m] = y_prob[m] * float(f)
        if clip:
            adjusted = np.clip(adjusted, 0.0, 1.0)
        y_pred = (adjusted >= threshold).astype(int)
        return (y_pred, adjusted) if return_proba else y_pred

    def equalized_odds_postprocessing(self, y_true: np.ndarray, y_prob: np.ndarray,
                                    sensitive_attr: np.ndarray) -> np.ndarray:
        groups = np.unique(sensitive_attr)
        y_pred_adjusted = np.zeros_like(y_prob, dtype=int)
        for group in groups:
            group_mask = (sensitive_attr == group)
            if np.sum(group_mask) > 0:
                try:
                    fpr, tpr, thresholds = roc_curve(y_true[group_mask], y_prob[group_mask])
                    balanced_accuracy = (tpr + (1 - fpr)) / 2
                    optimal_idx = np.argmax(balanced_accuracy)
                    optimal_threshold = thresholds[optimal_idx]
                except Exception:
                    optimal_threshold = 0.5
                optimal_threshold = float(np.clip(optimal_threshold, 0.0, 1.0))
                self.thresholds[str(group)] = optimal_threshold
                y_pred_adjusted[group_mask] = (y_prob[group_mask] >= optimal_threshold).astype(int)
        return y_pred_adjusted.astype(int)

    def demographic_parity_postprocessing(self, y_prob: np.ndarray,
                                        sensitive_attr: np.ndarray) -> np.ndarray:
        target_rate = float(np.mean(y_prob))
        groups = np.unique(sensitive_attr)
        y_pred_adjusted = np.zeros_like(y_prob, dtype=int)
        for group in groups:
            group_mask = (sensitive_attr == group)
            if np.sum(group_mask) > 0:
                group_probs = y_prob[group_mask]
                sorted_probs = np.sort(group_probs)[::-1]
                n_positive = int(target_rate * len(group_probs))
                if n_positive > 0 and n_positive < len(sorted_probs):
                    threshold = sorted_probs[n_positive - 1]
                else:
                    threshold = 0.5
                self.thresholds[str(group)] = threshold
                y_pred_adjusted[group_mask] = (group_probs >= threshold).astype(int)
        return y_pred_adjusted

    def calibration_postprocessing(self, y_true, y_prob, sensitive_attr, return_proba=False):
        groups = np.unique(sensitive_attr)
        y_pred_adjusted = np.zeros_like(y_prob, dtype=int)
        adjusted_probs = np.copy(y_prob)
        for g in groups:
            m = (sensitive_attr == g)
            if np.sum(m) > 0:
                actual = np.mean(y_true[m])
                pred   = np.mean(y_prob[m])
                if pred > 0:
                    factor = actual / pred
                    adjusted_probs[m] = np.clip(y_prob[m] * factor, 0.0, 1.0)
                y_pred_adjusted[m] = (adjusted_probs[m] >= 0.5).astype(int)
        return (y_pred_adjusted, adjusted_probs) if return_proba else y_pred_adjusted

    def reliability_diagram(self, y_true, y_prob, sensitive_attr=None, n_bins=15, save_path=None):
        import matplotlib.pyplot as plt
        if y_prob is None:
            return
        y_true = np.asarray(y_true).ravel()
        y_prob = np.asarray(y_prob).ravel()
        ok = np.isfinite(y_true) & np.isfinite(y_prob)
        if not np.all(ok):
            y_true, y_prob = y_true[ok], y_prob[ok]
            if sensitive_attr is not None:
                sensitive_attr = np.asarray(sensitive_attr).ravel()[ok]
        def plot_one(mask, label):
            bins = np.linspace(0,1,n_bins+1)
            xs, ys = [], []
            for b in range(n_bins):
                lo, hi = bins[b], bins[b+1]
                m = mask & (y_prob >= lo) & (y_prob < hi)
                if m.any():
                    xs.append(y_prob[m].mean()); ys.append(y_true[m].mean())
            plt.plot([0,1],[0,1],'--',alpha=.4)
            plt.plot(xs, ys, marker='o', label=label)
        plt.figure(figsize=(6,5))
        if sensitive_attr is None:
            mask = np.ones_like(y_true, dtype=bool); plot_one(mask, 'all')
        else:
            for g in np.unique(sensitive_attr):
                plot_one(sensitive_attr==g, f'group={g}')
        plt.xlabel('Predicted probability'); plt.ylabel('Empirical positive rate'); plt.title('Reliability diagram')
        plt.legend(); plt.grid(alpha=.3)
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')

    def equalized_odds_via_fairlearn(self, base_estimator, X, y_true, sensitive_attr):
        post = ThresholdOptimizer(
            estimator=base_estimator,
            constraints="equalized_odds",
            prefit=True,
        )
        post.fit(X, y_true, sensitive_features=sensitive_attr)
        return post

    def optimize_group_thresholds(self, y_true, y_prob, sensitive_attr,
                                  dp_max=None, eo_max=None, ece_max=None,
                                  grid=None, max_iter=8, restarts=5, random_state: int = 42):
        import numpy as _np
        from src.fairness_metrics import FairnessEvaluator, group_ece

        y_true = _np.asarray(y_true).ravel()
        p = _np.asarray(y_prob).ravel()
        s = _np.asarray(sensitive_attr).ravel()
                                
                                                                                       
        p_mask = _np.isfinite(p)
        y_mask = _np.isfinite(y_true)
        if s.dtype.kind in ('f', 'i', 'u'):
            s_mask = _np.isfinite(s)
        else:
                                                                          
            s_mask = _np.array([(_s is not None) and (not (isinstance(_s, float) and _np.isnan(_s))) for _s in s], dtype=bool)
        mask = p_mask & y_mask & s_mask
        if mask.sum() < len(p):
            p = p[mask]; s = s[mask]; y_true = y_true[mask]
        groups = _np.unique(s)
        grid = _np.asarray(grid if grid is not None else _np.linspace(0.1, 0.9, 17))
        ev = FairnessEvaluator()

        def _predict(thr_map):
            yhat = _np.zeros_like(p, dtype=int)
            for g in groups:
                m = (s == g)
                yhat[m] = (p[m] >= float(thr_map[g])).astype(int)
            return yhat

        def _score(yhat):
            acc = (yhat == y_true).mean()
            dp  = ev.demographic_parity(yhat, s)['demographic_parity_difference']
            eo  = ev.equalized_odds(y_true, yhat, s)['equalized_odds_difference']
            ece = group_ece(y_true, p, s).get('ece_diff', _np.nan)
                                       
            pen = 0.0
            if dp_max is not None and dp > dp_max: pen += 1e3*(dp - dp_max)
            if eo_max is not None and eo > eo_max: pen += 1e3*(eo - eo_max)
            if ece_max is not None and _np.isfinite(ece) and ece > ece_max: pen += 1e3*(ece - ece_max)
                                            
            return acc - pen, acc, dp, eo, ece

                                                                                
        global_best = None
        rng = _np.random.default_rng(random_state)
        for r in range(int(max(0, restarts)) + 1):
            if r == 0:
                thr = {g: 0.5 for g in groups}
            else:
                thr = {g: float(grid[rng.integers(0, len(grid))]) for g in groups}
            local_best = None
            for _ in range(max_iter):
                improved = False
                for g in groups:
                    best_g = thr[g]; best_local = None
                    for t in grid:
                        thr[g] = float(t)
                        yhat = _predict(thr)
                        val = _score(yhat)
                        if best_local is None or val[0] > best_local[0]:
                            best_local = val; best_g = float(t); local_best = (val, yhat, dict(thr))
                    if thr[g] != best_g:
                        thr[g] = best_g; improved = True
                if not improved:
                    break
            if global_best is None or (local_best and local_best[0][0] > global_best[0][0]):
                global_best = local_best

        (obj, acc, dp, eo, ece), yhat, thr_map = global_best
                          
        self.thresholds = {str(k): float(v) for k, v in thr_map.items()}
        return {
            'y_pred': yhat.astype(int),
            'y_prob': p,
            'thresholds': self.thresholds,
            'metrics': {'accuracy': float(acc), 'dp_diff': float(dp), 'eo_diff': float(eo), 'ece_diff': float(ece)},
            'mask': mask
        }

    def calibration_constrained_fairness(self, y_true, y_prob, sensitive_attr,
                                         dp_max=None, eo_max=None, ece_max=None,
                                         grid=None, max_iter=8, restarts=5, random_state: int = 42):
        return self.optimize_group_thresholds(
            y_true=y_true,
            y_prob=y_prob,
            sensitive_attr=sensitive_attr,
            dp_max=dp_max,
            eo_max=eo_max,
            ece_max=ece_max,
            grid=grid,
            max_iter=max_iter,
            restarts=restarts,
            random_state=random_state
        )


