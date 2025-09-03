import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FairnessEvaluator:
    def __init__(self):
        self.metrics_history = []
        
    def demographic_parity(self, y_pred: np.ndarray, sensitive_attr: np.ndarray) -> Dict[str, float]:
        groups = np.unique(sensitive_attr)
        group_rates = {}
        
        for group in groups:
            group_mask = (sensitive_attr == group)
            if np.sum(group_mask) > 0:
                positive_rate = np.mean(y_pred[group_mask])
                group_rates[str(group)] = positive_rate
        
        rates_values = list(group_rates.values())
        max_rate = max(rates_values) if rates_values else 0
        min_rate = min(rates_values) if rates_values else 0
        
        return {
            'group_rates': group_rates,
            'demographic_parity_difference': max_rate - min_rate,
            'demographic_parity_ratio': min_rate / max_rate if max_rate > 0 else 0
        }
    
    def equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      sensitive_attr: np.ndarray) -> Dict[str, float]:
        groups = np.unique(sensitive_attr)
        group_metrics = {}
        
        for group in groups:
            group_mask = (sensitive_attr == group)
            if np.sum(group_mask) > 0:
                y_true_group = y_true[group_mask]
                y_pred_group = y_pred[group_mask]
                
                cm = confusion_matrix(y_true_group, y_pred_group)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                else:
                    tpr, fpr = 0, 0
                
                group_metrics[str(group)] = {'tpr': tpr, 'fpr': fpr}
        
        tprs = [metrics['tpr'] for metrics in group_metrics.values()]
        fprs = [metrics['fpr'] for metrics in group_metrics.values()]
        
        tpr_diff = max(tprs) - min(tprs) if tprs else 0
        fpr_diff = max(fprs) - min(fprs) if fprs else 0
        
        return {
            'group_metrics': group_metrics,
            'tpr_difference': tpr_diff,
            'fpr_difference': fpr_diff,
            'equalized_odds_difference': max(tpr_diff, fpr_diff)
        }

    def equal_opportunity(self, y_true: np.ndarray, y_pred: np.ndarray,
                          sensitive_attr: np.ndarray) -> Dict[str, float]:
        groups = np.unique(sensitive_attr)
        tprs = {}
        for g in groups:
            m = (sensitive_attr == g)
            if np.sum(m) > 0:
                yt = y_true[m]
                yp = y_pred[m]
                tp = int(((yt == 1) & (yp == 1)).sum())
                fn = int(((yt == 1) & (yp == 0)).sum())
                tprs[str(g)] = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        vals = list(tprs.values())
        return {
            'group_tpr': tprs,
            'equal_opportunity_difference': (max(vals) - min(vals)) if vals else 0.0
        }
    
    def calibration_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, 
                           sensitive_attr: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        y_true = np.asarray(y_true).ravel()
        y_prob = np.asarray(y_prob)
        if y_prob.ndim == 2:
            if y_prob.shape[1] >= 2:
                y_prob = y_prob[:, 1]
            else:
                y_prob = y_prob.ravel()
        else:
            if y_prob.size == 2 * len(y_true):
                try:
                    y_prob = y_prob.reshape(len(y_true), 2)[:, 1]
                except Exception:
                    y_prob = y_prob.ravel()[: len(y_true)]
            else:
                y_prob = y_prob.ravel()

        groups = np.unique(sensitive_attr)
        group_calibration = {}
        
        for group in groups:
            group_mask = (sensitive_attr == group)
            if np.sum(group_mask) > 0:
                y_true_group = y_true[group_mask]
                y_prob_group = y_prob[group_mask]
                
                try:
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        y_true_group, y_prob_group, n_bins=n_bins
                    )
                    
                    calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                    
                    group_calibration[str(group)] = {
                        'calibration_error': calibration_error,
                        'fraction_of_positives': fraction_of_positives,
                        'mean_predicted_value': mean_predicted_value
                    }
                except:
                    group_calibration[str(group)] = {
                        'calibration_error': np.nan,
                        'fraction_of_positives': np.array([]),
                        'mean_predicted_value': np.array([])
                    }
        
        calibration_errors = [
            v['calibration_error']
            for v in group_calibration.values()
            if not np.isnan(v['calibration_error'])
        ]
        calibration_difference = (
            max(calibration_errors) - min(calibration_errors) if calibration_errors else 0
        )
        
        return {
            'group_calibration': group_calibration,
            'calibration_difference': calibration_difference
        }
    
    def performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_prob: np.ndarray = None) -> Dict[str, float]:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }

        if y_prob is not None:
            try:
                yp = np.asarray(y_prob)
                if yp.ndim == 2:
                    yp = yp[:, 1] if yp.shape[1] >= 2 else yp.ravel()
                else:
                    if yp.size == 2 * len(y_true):
                        yp = yp.reshape(len(y_true), 2)[:, 1]
                    else:
                        yp = yp.ravel()
                if not np.allclose(yp, np.round(yp)):
                    metrics['auc'] = roc_auc_score(y_true, yp)
                else:
                    metrics['auc'] = np.nan
            except Exception:
                metrics['auc'] = np.nan

        return metrics
    
    def intersectional_fairness(self, y_true: np.ndarray, y_pred: np.ndarray,
                                sensitive_attrs: Dict[str, np.ndarray],
                                min_group_size: int = 25) -> Dict[str, Any]:
        if len(sensitive_attrs) < 2:
            return {"error": "Intersectional analysis requires at least 2 sensitive attributes"}

        attr_names = list(sensitive_attrs.keys())
        attr_values = list(sensitive_attrs.values())
        attr_arrays = [np.array(attr) for attr in attr_values]

        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)

        intersectional_groups = []
        for i in range(len(y_true_array)):
            group_combination = " × ".join(str(attr_arrays[j][i]) for j in range(len(attr_arrays)))
            intersectional_groups.append(group_combination)

        intersectional_groups = np.array(intersectional_groups)
        unique_groups = np.unique(intersectional_groups)
        total_groups = int(len(unique_groups))

        group_metrics = {}
        excluded_count = 0
        for group in unique_groups:
            group_mask = (intersectional_groups == group)
            group_size = int(np.sum(group_mask))
            if group_size < min_group_size:
                excluded_count += 1
                continue

            group_y_true = y_true_array[group_mask]
            group_y_pred = y_pred_array[group_mask]

            positive_rate = np.mean(group_y_pred) if group_size > 0 else 0
            accuracy = np.mean(group_y_true == group_y_pred) if group_size > 0 else 0

            if group_size > 0 and len(np.unique(group_y_true)) > 1:
                cm = confusion_matrix(group_y_true, group_y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                else:
                    tpr = fpr = precision = recall = 0
            else:
                tpr = fpr = precision = recall = 0

            group_metrics[str(group)] = {
                'size': group_size,
                'positive_rate': positive_rate,
                'accuracy': accuracy,
                'tpr': tpr,
                'fpr': fpr,
                'precision': precision,
                'recall': recall
            }

        if group_metrics:
            positive_rates = [m['positive_rate'] for m in group_metrics.values()]
            accuracies     = [m['accuracy'] for m in group_metrics.values()]
            tprs           = [m['tpr'] for m in group_metrics.values()]
            fprs           = [m['fpr'] for m in group_metrics.values()]

            intersectional_disparities = {
                'positive_rate_disparity': max(positive_rates) - min(positive_rates),
                'accuracy_disparity':      max(accuracies)     - min(accuracies),
                'tpr_disparity':           max(tprs)           - min(tprs),
                'fpr_disparity':           max(fprs)           - min(fprs),
                'max_disparity': max(
                    max(positive_rates) - min(positive_rates),
                    max(accuracies)     - min(accuracies),
                    max(tprs)           - min(tprs),
                    max(fprs)           - min(fprs)
                )
            }
        else:
            intersectional_disparities = {}

        return {
            'attributes': attr_names,
            'group_metrics': group_metrics,
            'intersectional_disparities': intersectional_disparities,
            'num_groups': len(group_metrics),
            'min_group_size': min_group_size,
            'total_groups_raw': total_groups,
            'excluded_groups_small_n': excluded_count
        }


    def comprehensive_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               sensitive_attrs: Dict[str, np.ndarray], 
                               y_prob: np.ndarray = None, 
                               model_name: str = "Model") -> Dict[str, Any]:
        results = {
            'model_name': model_name,
            'performance': self.performance_metrics(y_true, y_pred, y_prob),
            'fairness': {}
        }
        results['fairness'] = self.evaluate(y_true, y_pred, y_prob, sensitive_attrs)

        if len(sensitive_attrs) >= 2:
            intersectional_results = self.intersectional_fairness(
                y_true, y_pred, sensitive_attrs, min_group_size=50
            )
            results['fairness']['intersectional'] = intersectional_results
        
        self.metrics_history.append(results)
        
        return results

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_prob: np.ndarray, sensitive_attrs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for attr_name, attr_values in sensitive_attrs.items():
            attr_results: Dict[str, Any] = {}

            dp_metrics = self.demographic_parity(y_pred, attr_values)
            attr_results['demographic_parity'] = dp_metrics

            eo_metrics = self.equalized_odds(y_true, y_pred, attr_values)
            attr_results['equalized_odds'] = eo_metrics

            eopp_metrics = self.equal_opportunity(y_true, y_pred, attr_values)
            attr_results['equal_opportunity'] = eopp_metrics

            try:
                overall_pr = float(np.mean(y_pred))
                rates = dp_metrics.get('group_rates', {})
                if rates:
                    worst_g = max(rates.items(), key=lambda kv: abs(float(kv[1]) - overall_pr))
                    attr_results['worst_dp_group'] = {
                        'group': worst_g[0],
                        'gap': float(abs(float(worst_g[1]) - overall_pr))
                    }
            except Exception:
                pass

            if y_prob is not None:
                cal_metrics = self.calibration_metrics(y_true, y_prob, attr_values, n_bins=15)
                ece_metrics = group_ece(y_true, y_prob, attr_values, n_bins=15)
                cal_metrics['ece'] = ece_metrics
                attr_results['calibration'] = cal_metrics

            out[attr_name] = attr_results
        return out
    
    def compare_models(self, results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        comparison_data = []
        
        for results in results_list:
            row = {
                'Model': results['model_name'],
                'Accuracy': results['performance']['accuracy'],
                'Precision': results['performance']['precision'],
                'Recall': results['performance']['recall'],
                'F1': results['performance']['f1']
            }
            
            if 'auc' in results['performance']:
                row['AUC'] = results['performance']['auc']
            
            for attr_name, attr_metrics in results['fairness'].items():
                if 'demographic_parity' in attr_metrics:
                    row[f'{attr_name}_DP_Diff'] = attr_metrics['demographic_parity']['demographic_parity_difference']
                
                if 'equalized_odds' in attr_metrics:
                    row[f'{attr_name}_EO_Diff'] = attr_metrics['equalized_odds']['equalized_odds_difference']

                if 'equal_opportunity' in attr_metrics:
                    row[f'{attr_name}_EOpp_Diff'] = attr_metrics['equal_opportunity']['equal_opportunity_difference']

                if 'worst_dp_group' in attr_metrics:
                    row[f'{attr_name}_WorstDP_Group'] = attr_metrics['worst_dp_group'].get('group')
                    row[f'{attr_name}_WorstDP_Gap'] = attr_metrics['worst_dp_group'].get('gap')
                
                if 'calibration' in attr_metrics:
                    row[f'{attr_name}_Cal_Diff'] = attr_metrics['calibration']['calibration_difference']
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        print("="*60)
        print(f"EVALUATION SUMMARY: {results['model_name']}")
        print("="*60)
        
        print("\nPERFORMANCE METRICS:")
        print("-" * 20)
        for metric, value in results['performance'].items():
            if not np.isnan(value):
                print(f"{metric.upper()}: {value:.4f}")
        
        print("\nFAIRNESS METRICS:")
        print("-" * 20)
        
        for attr_name, attr_metrics in results['fairness'].items():
            print(f"\n{attr_name.upper()} ATTRIBUTE:")
            
            if 'demographic_parity' in attr_metrics:
                dp = attr_metrics['demographic_parity']
                print(f"  Demographic Parity Difference: {dp['demographic_parity_difference']:.4f}")
                print(f"  Group Rates: {dp['group_rates']}")
            
            if 'equalized_odds' in attr_metrics:
                eo = attr_metrics['equalized_odds']
                print(f"  Equalized Odds Difference: {eo['equalized_odds_difference']:.4f}")
                print(f"  TPR Difference: {eo['tpr_difference']:.4f}")
                print(f"  FPR Difference: {eo['fpr_difference']:.4f}")
            
            if 'calibration' in attr_metrics:
                cal = attr_metrics['calibration']
                print(f"  Calibration Difference: {cal['calibration_difference']:.4f}")
        
        print("="*60)
    
    def plot_fairness_comparison(self, results_list: List[Dict[str, Any]], 
                               save_path: str = None):
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        comparison_df = self.compare_models(results_list)

        def _abbr(name: str) -> str:
            m = {
                'Baseline (Tuned)': 'Baseline',
                'Preprocessing (Sex)': 'Pre(Sex)',
                'Preprocessing (Race)': 'Pre(Race)',
                'Adversarial (Sex)': 'Adv(Sex)',
                'Adversarial (Race)': 'Adv(Race)',
                'Fairlearn EG (Sex)': 'EG(Sex)',
                'Fairlearn EG (Race)': 'EG(Race)',
                'Fairlearn DP (Sex)': 'DP(Sex)',
                'Fairlearn DP (Race)': 'DP(Race)',
                'Post-processing EO (Sex)': 'PostEO(Sex)',
                'Post-processing Cal (Sex)': 'PostCal(Sex)',
                'Post-processing EO (Race)': 'PostEO(Race)',
                'Post-processing Cal (Race)': 'PostCal(Race)',
                'Fairlearn EG (Both)': 'EG(Both)'
            }
            return m.get(name, name)

        comparison_df = comparison_df.copy()
        comparison_df['Model'] = comparison_df['Model'].map(_abbr)

        fairness_cols = [col for col in comparison_df.columns 
                        if any(suffix in col for suffix in ['_DP_Diff', '_EO_Diff', '_EOpp_Diff', '_Cal_Diff'])]
        if not fairness_cols:
            print("No fairness metrics found for plotting.")
            return

        num_models = len(comparison_df)
        nrows = len(fairness_cols)
        fig_height = max(3 * nrows, min(0.5 * num_models * nrows + 1, 18))
        fig_width = max(10, 12 if num_models > 10 else 10)
        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(fig_width, fig_height))
        if nrows == 1:
            axes = [axes]

        for i, col in enumerate(fairness_cols):
            ax = axes[i]
            series = comparison_df.set_index('Model')[col].copy()
            series = series.reindex(series.abs().sort_values(ascending=False).index)
            ax.barh(series.index, series.values, color='steelblue', alpha=0.85)
            ax.set_title(col, fontsize=12)
            ax.set_xlabel('Difference (lower is better)')
            ax.grid(True, axis='x', alpha=0.3)
            ax.invert_yaxis()
            ax.axvline(0.0, color='black', linewidth=0.8, alpha=0.6)
            ax.tick_params(axis='y', labelsize=8 if num_models > 10 else 9)
            plt.subplots_adjust(left=0.28 if num_models > 12 else 0.22)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Fairness comparison plot saved to: {save_path}")
        try:
            plt.show()
        except Exception:
            pass 
    
    def plot_performance_vs_fairness(self, results_list: List[Dict[str, Any]], 
                                   fairness_metric: str = 'sex_DP_Diff',
                                   performance_metric: str = 'Accuracy',
                                   save_path: str = None):
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        comparison_df = self.compare_models(results_list)

        if fairness_metric not in comparison_df.columns:
            candidates = [c for c in comparison_df.columns if c.endswith('_DP_Diff')]
            if candidates:
                fairness_metric = candidates[0]
            else:
                print("No fairness metric columns found for plotting.")
                return
        if performance_metric not in comparison_df.columns:
            print(f"Metric {performance_metric} not found in results.")
            return

        def _abbr(name: str) -> str:
            m = {
                'Baseline (Tuned)': 'Baseline',
                'Preprocessing (Sex)': 'Pre(Sex)',
                'Preprocessing (Race)': 'Pre(Race)',
                'Adversarial (Sex)': 'Adv(Sex)',
                'Adversarial (Race)': 'Adv(Race)',
                'Fairlearn EG (Sex)': 'EG(Sex)',
                'Fairlearn EG (Race)': 'EG(Race)',
                'Fairlearn DP (Sex)': 'DP(Sex)',
                'Fairlearn DP (Race)': 'DP(Race)',
                'Post-processing EO (Sex)': 'PostEO(Sex)',
                'Post-processing Cal (Sex)': 'PostCal(Sex)',
                'Post-processing EO (Race)': 'PostEO(Race)',
                'Post-processing Cal (Race)': 'PostCal(Race)',
                'Fairlearn EG (Both)': 'EG(Both)'
            }
            return m.get(name, name)

        labels = [ _abbr(m) for m in comparison_df['Model'].tolist() ]

        plt.figure(figsize=(8.5, 5.5))
        for label, (_, row) in zip(labels, comparison_df.iterrows()):
            plt.scatter(row[fairness_metric], row[performance_metric], s=70, alpha=0.8, label=label)

        plt.xlabel(f'{fairness_metric} (Lower is Better)')
        plt.ylabel(f'{performance_metric} (Higher is Better)')
        plt.title('Performance vs Fairness Trade-off')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, frameon=False)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance vs fairness plot saved to: {save_path}")

        plt.show() 
    
def _ece(y_true, y_prob, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b+1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.any():
            conf = y_prob[mask].mean()
            acc  = y_true[mask].mean()
            ece += (mask.mean()) * abs(acc - conf)
    return float(ece)

def group_ece(y_true: np.ndarray, y_prob: np.ndarray, sensitive: np.ndarray, n_bins: int = 15) -> Dict[str, Any]:
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 2:
        if y_prob.shape[1] >= 2:
            y_prob = y_prob[:, 1]
        else:
            y_prob = y_prob.ravel()
    else:
        if y_prob.size == 2 * len(y_true):
            try:
                y_prob = y_prob.reshape(len(y_true), 2)[:, 1]
            except Exception:
                y_prob = y_prob.ravel()[: len(y_true)]
        else:
            y_prob = y_prob.ravel()

    sensitive = np.asarray(sensitive).ravel()
    n = min(len(y_true), len(y_prob), len(sensitive))
    y_true = y_true[:n]; y_prob = y_prob[:n]; sensitive = sensitive[:n]

    groups = np.unique(sensitive)
    per_group = {}
    for g in groups:
        m = (sensitive == g)
        if m.any():
            per_group[str(g)] = _ece(y_true[m], y_prob[m], n_bins)
    vals = list(per_group.values())
    return {
        "group_ece": per_group,
        "ece_max": max(vals) if vals else 0.0,
        "ece_min": min(vals) if vals else 0.0,
        "ece_diff": (max(vals) - min(vals)) if vals else 0.0
    }

def worst_case_subgroup(table: Dict[str, Dict[str, float]], key: str) -> Tuple[str, float]:
    if not table:
        return ("", 0.0)
    all_vals = np.array([row.get(key, np.nan) for row in table.values()], dtype=float)
    mu = np.nanmean(all_vals)
    diffs = {g: abs(row.get(key, np.nan) - mu) for g, row in table.items()}
    g_star = max(diffs, key=diffs.get)
    return g_star, diffs[g_star]