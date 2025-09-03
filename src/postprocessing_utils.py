import numpy as np

def postproc_proba(post, X, sensitive):
    proba = None
    try:
        proba = post.predict_proba(X, sensitive_features=sensitive)
    except Exception:
        pass
    if proba is None:
        try:
            p1 = np.asarray(post._pmf_predict(X, sensitive_features=sensitive)).ravel()
            proba = np.column_stack([1.0 - p1, p1])
        except Exception:
            pass
    if proba is None:
        preds = np.asarray(post.predict(X, sensitive_features=sensitive)).astype(float).ravel()
        proba = np.column_stack([1.0 - preds, preds])
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    proba = np.clip(proba, 0.0, 1.0)
    if np.isnan(proba).any():
        col_means = np.nanmean(proba, axis=0)
        inds = np.where(np.isnan(proba))
        proba[inds] = np.take(col_means, inds[1])
    return proba[:, 1]


