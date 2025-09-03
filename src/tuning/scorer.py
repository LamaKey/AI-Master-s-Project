from typing import Dict
import numpy as np

def fairness_aware_score(accuracy: float, disparities: Dict[str, float],
                         weight_accuracy: float = 0.6, weight_fairness: float = 0.4) -> float:
    disp_vals = [v for v in disparities.values() if v is not None and not np.isnan(v)]
    disp = float(np.nanmean(disp_vals)) if disp_vals else 1.0
    fairness = 1.0 - float(np.clip(disp, 0.0, 1.0))
    return float(weight_accuracy) * float(accuracy) + float(weight_fairness) * fairness


