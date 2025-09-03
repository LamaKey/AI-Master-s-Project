import numpy as np

                                                                             
                                                                                          
def bootstrap_cis(y_true, y_pred, y_prob, sensitive_attrs, n_boot=1000, seed=42, attribute: str = 'sex'):
    from fairness_metrics import FairnessEvaluator, group_ece
    from sklearn.metrics import f1_score

                                                    
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1] if y_prob.shape[1] >= 2 else y_prob.ravel()
        else:
            if y_prob.size == 2 * len(y_true):
                try:
                    y_prob = y_prob.reshape(len(y_true), 2)[:, 1]
                except Exception:
                    y_prob = y_prob.ravel()[: len(y_true)]
            else:
                y_prob = y_prob.ravel()
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]; y_pred = y_pred[:n]
    if y_prob is not None:
        y_prob = y_prob[:n]

    rng = np.random.default_rng(seed)
    acc, f1s, dp_diffs, eo_diffs, eces = [], [], [], [], []
    ev = FairnessEvaluator()
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        acc.append((yt == yp).mean())
        f1s.append(f1_score(yt, yp, zero_division=0))
                                                                 
        if attribute in sensitive_attrs:
            s_all = np.asarray(sensitive_attrs[attribute]).ravel()[:n]
            s = s_all[idx]
            dp = ev.demographic_parity(yp, s)['demographic_parity_difference']
            eo = ev.equalized_odds(yt, yp, s)['equalized_odds_difference']
            dp_diffs.append(dp); eo_diffs.append(eo)
        if y_prob is not None and attribute in sensitive_attrs:
            eces.append(group_ece(yt, y_prob[idx], s).get('ece_diff', np.nan))

    def ci(a):
        a = np.array(a, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (np.nan, np.nan, np.nan)
        lo, hi = np.percentile(a, [2.5, 97.5])
        return float(lo), float(a.mean()), float(hi)

    return {
        'accuracy_CI': ci(acc), 'f1_CI': ci(f1s),
        'dp_diff_CI': ci(dp_diffs),
        'eo_diff_CI': ci(eo_diffs),
        'ece_diff_CI': ci(eces)
    }

                                                  
                                                                        
def inject_label_noise(y, rate=0.05, seed=42):
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    flip = rng.random(len(y_noisy)) < rate
    y_noisy[flip] = 1 - y_noisy[flip]
    return y_noisy


