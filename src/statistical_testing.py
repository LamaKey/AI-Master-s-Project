import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class StatisticalTester:
    
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.cv_results = {}
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray,
                           sensitive_attrs: Dict[str, np.ndarray],
                           model_name: str,
                           preprocessor=None,
                           attr_for_preproc: str = None) -> Dict[str, Any]:
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        fold_results = {
            'accuracy': [],
            'f1': [],
            'fairness_metrics': {attr_name: [] for attr_name in sensitive_attrs.keys()}
        }
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            if hasattr(X, 'iloc'):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                
            if hasattr(y, 'iloc'):
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            else:
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            sensitive_train_fold = {}
            sensitive_val_fold = {}
            for attr_name, attr_values in sensitive_attrs.items():
                attr_arr = np.asarray(attr_values)
                sensitive_train_fold[attr_name] = attr_arr[train_idx]
                sensitive_val_fold[attr_name]   = attr_arr[val_idx]
                        
            try:
                import copy
                try:
                    model_fold = clone(model)
                except Exception:
                    try:
                        model_fold = copy.deepcopy(model)
                    except Exception:
                        model_fold = model
                
                X_train_fold_clean = pd.DataFrame(X_train_fold).reset_index(drop=True).values
                y_train_fold_clean = pd.Series(y_train_fold).reset_index(drop=True).values
                
                primary_sensitive_clean = None
                if sensitive_train_fold:
                    if 'sex' in sensitive_train_fold:
                        primary_sensitive = sensitive_train_fold['sex']
                    elif 'race' in sensitive_train_fold:
                        primary_sensitive = sensitive_train_fold['race']
                    else:
                        primary_sensitive = next(iter(sensitive_train_fold.values()))

                    if hasattr(primary_sensitive, 'reset_index'):
                        primary_sensitive_clean = primary_sensitive.reset_index(drop=True)
                    else:
                        primary_sensitive_clean = pd.Series(primary_sensitive).reset_index(drop=True)
                
                if hasattr(model_fold, 'fit'):
                    if preprocessor is not None and attr_for_preproc and attr_for_preproc in sensitive_train_fold:
                        s_tr = np.asarray(sensitive_train_fold[attr_for_preproc])
                        try:
                            Xw, yw, _, w = preprocessor.reweight_samples(
                                X_train_fold_clean, y_train_fold_clean, s_tr,
                                attr_name=attr_for_preproc
                            )
                            model_fold.fit(Xw, yw, sample_weight=w)
                        except Exception:
                            model_fold.fit(X_train_fold_clean, y_train_fold_clean)
                    elif hasattr(model_fold, 'constraint_type'):
                        model_fold.fit(X_train_fold_clean, y_train_fold_clean,
                                       sensitive_features=primary_sensitive_clean)
                    else:
                        model_fold.fit(X_train_fold_clean, y_train_fold_clean)
                else:
                    print("Warning: model has no .fit(); using as-is for CV fold.")
                
                X_val_fold_clean = pd.DataFrame(X_val_fold).reset_index(drop=True).values
                y_val_fold_clean = pd.Series(y_val_fold).reset_index(drop=True).values
                
                y_pred_fold = model_fold.predict(X_val_fold_clean)
                
                accuracy = accuracy_score(y_val_fold_clean, y_pred_fold)
                f1 = f1_score(y_val_fold_clean, y_pred_fold, average='binary', zero_division=0)
                
                fold_results['accuracy'].append(accuracy)
                fold_results['f1'].append(f1)
                
                from fairness_metrics import FairnessEvaluator
                evaluator = FairnessEvaluator()
                
                for attr_name, attr_values in sensitive_val_fold.items():
                    if hasattr(attr_values, 'reset_index'):
                        attr_values_clean = attr_values.reset_index(drop=True)
                    else:
                        attr_values_clean = pd.Series(attr_values).reset_index(drop=True)
                    
                    dp_metrics = evaluator.demographic_parity(y_pred_fold, attr_values_clean)
                    
                    eo_metrics = evaluator.equalized_odds(y_val_fold_clean, y_pred_fold, attr_values_clean)
                    
                    eopp_metrics = evaluator.equal_opportunity(y_val_fold_clean, y_pred_fold, attr_values_clean)
                    
                    fold_fairness = {
                        'demographic_parity_diff': dp_metrics['demographic_parity_difference'],
                        'equalized_odds_diff': eo_metrics['equalized_odds_difference'],
                        'equal_opportunity_diff': eopp_metrics['equal_opportunity_difference']
                    }
                    
                    fold_results['fairness_metrics'][attr_name].append(fold_fairness)

                try:
                    import os
                    os.makedirs('results/reports', exist_ok=True)
                    model_tag = str(model_name)
                    if model_tag in ('Preprocessing (Sex)', 'Preprocessing (Race)'):
                        import pandas as _pd
                        _pd.DataFrame({
                            'fold': [fold_idx],
                            'accuracy': [accuracy],
                            'f1': [f1],
                            'attr': ['sex' if 'Sex' in model_tag else 'race'],
                            'dp_diff': [fold_fairness.get('demographic_parity_diff')],
                            'eo_diff': [fold_fairness.get('equalized_odds_diff')],
                            'eopp_diff': [fold_fairness.get('equal_opportunity_diff')]
                        }).to_csv(f'results/reports/cv_folds_{"sex" if "Sex" in model_tag else "race"}.csv', mode='a', header=not os.path.exists(f'results/reports/cv_folds_{"sex" if "Sex" in model_tag else "race"}.csv'), index=False)
                except Exception:
                    pass
                
            except Exception as e:
                print(f"Error in fold {fold_idx}: {e}")
                fold_results['accuracy'].append(np.nan)
                fold_results['f1'].append(np.nan)
                
                for attr_name in sensitive_attrs.keys():
                    fold_results['fairness_metrics'][attr_name].append({
                        'demographic_parity_diff': np.nan,
                        'equalized_odds_diff': np.nan,
                        'equal_opportunity_diff': np.nan
                    })
        
        self.cv_results[model_name] = fold_results
        
        summary = {
            'model_name': model_name,
            'accuracy_mean': np.nanmean(fold_results['accuracy']),
            'accuracy_std': np.nanstd(fold_results['accuracy']),
            'f1_mean': np.nanmean(fold_results['f1']),
            'f1_std': np.nanstd(fold_results['f1']),
            'fairness_summary': {}
        }
        
        for attr_name, attr_results in fold_results['fairness_metrics'].items():
            dp_diffs = [result['demographic_parity_diff'] for result in attr_results if not np.isnan(result['demographic_parity_diff'])]
            eo_diffs = [result['equalized_odds_diff'] for result in attr_results if not np.isnan(result['equalized_odds_diff'])]
            eopp_diffs = [result.get('equal_opportunity_diff') for result in attr_results if not np.isnan(result.get('equal_opportunity_diff', np.nan))]
            
            summary['fairness_summary'][attr_name] = {
                'dp_diff_mean': np.mean(dp_diffs) if dp_diffs else np.nan,
                'dp_diff_std': np.std(dp_diffs) if dp_diffs else np.nan,
                'eo_diff_mean': np.mean(eo_diffs) if eo_diffs else np.nan,
                'eo_diff_std': np.std(eo_diffs) if eo_diffs else np.nan,
                'eopp_diff_mean': np.mean(eopp_diffs) if eopp_diffs else np.nan,
                'eopp_diff_std': np.std(eopp_diffs) if eopp_diffs else np.nan
            }
        
        return summary
    
    def paired_t_test(self, model1_name: str, model2_name: str, 
                     metric: str = 'accuracy') -> Dict[str, float]:
        if model1_name not in self.cv_results or model2_name not in self.cv_results:
            raise ValueError("Models must be cross-validated first")
        
        scores1 = self.cv_results[model1_name][metric]
        scores2 = self.cv_results[model2_name][metric]
        
        valid_pairs = [(s1, s2) for s1, s2 in zip(scores1, scores2) 
                      if not (np.isnan(s1) or np.isnan(s2))]
        
        if len(valid_pairs) < 2:
            return {'t_statistic': np.nan, 'p_value': np.nan, 'significant': False}
        
        scores1_clean, scores2_clean = zip(*valid_pairs)
        
        t_statistic, p_value = stats.ttest_rel(scores1_clean, scores2_clean)

        diff     = np.array(scores1_clean) - np.array(scores2_clean)
        sd_diff  = diff.std(ddof=1)
        cohen_d  = diff.mean() / sd_diff if sd_diff else 0.0
        
        return {
            't_statistic': t_statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': np.mean(scores1_clean) - np.mean(scores2_clean),
            'cohen_d': cohen_d
        }
    
    def wilcoxon_test(self, model1_name: str, model2_name: str,
                     metric: str = 'accuracy') -> Dict[str, float]:
        if model1_name not in self.cv_results or model2_name not in self.cv_results:
            raise ValueError("Models must be cross-validated first")
        
        scores1 = self.cv_results[model1_name][metric]
        scores2 = self.cv_results[model2_name][metric]
        
        valid_pairs = [(s1, s2) for s1, s2 in zip(scores1, scores2) 
                      if not (np.isnan(s1) or np.isnan(s2))]
        
        if len(valid_pairs) < 2:
            return {'statistic': np.nan, 'p_value': np.nan, 'significant': False}
        
        scores1_clean, scores2_clean = zip(*valid_pairs)
        
        try:
            statistic, p_value = stats.wilcoxon(scores1_clean, scores2_clean, alternative='two-sided')
        except ValueError:
            return {'statistic': 0, 'p_value': 1.0, 'significant': False}
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def fairness_significance_test(self, model1_name: str, model2_name: str,
                                attribute: str, fairness_metric: str = 'demographic_parity_diff') -> Dict[str, float]:
        if model1_name not in self.cv_results or model2_name not in self.cv_results:
            raise ValueError("Models must be cross-validated first")
        
        fairness1 = self.cv_results[model1_name]['fairness_metrics'][attribute]
        fairness2 = self.cv_results[model2_name]['fairness_metrics'][attribute]

        pairs = []
        for r1, r2 in zip(fairness1, fairness2):
            v1 = r1.get(fairness_metric, np.nan)
            v2 = r2.get(fairness_metric, np.nan)
            if not (np.isnan(v1) or np.isnan(v2)):
                pairs.append((v1, v2))

        if len(pairs) < 2:
            return {'t_statistic': np.nan, 'p_value': np.nan, 'significant': False}

        s1, s2 = map(np.array, zip(*pairs))
        t_statistic, p_value = stats.ttest_rel(s1, s2)

        return {
            't_statistic': t_statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'fairness_improvement': float(np.mean(s1) - np.mean(s2))
        }

    def comprehensive_comparison(self, baseline_model: str, 
                               comparison_models: List[str]) -> pd.DataFrame:
        results = []
        
        for model in comparison_models:
            result_row = {'Model': model}
            
            for metric in ['accuracy', 'f1']:
                t_test = self.paired_t_test(baseline_model, model, metric)
                wilcoxon = self.wilcoxon_test(baseline_model, model, metric)
                
                result_row[f'{metric}_t_test_p'] = t_test['p_value']
                result_row[f'{metric}_t_test_significant'] = t_test['significant']
                result_row[f'{metric}_wilcoxon_p'] = wilcoxon['p_value']
                result_row[f'{metric}_wilcoxon_significant'] = wilcoxon['significant']
                result_row[f'{metric}_effect_size'] = t_test.get('effect_size', np.nan)
            
            if baseline_model in self.cv_results and model in self.cv_results:
                baseline_fairness = self.cv_results[baseline_model]['fairness_metrics']
                for attr_name in baseline_fairness.keys():
                    for fairness_metric in ['demographic_parity_diff', 'equalized_odds_diff']:
                        fairness_test = self.fairness_significance_test(
                            baseline_model, model, attr_name, fairness_metric
                        )
                        result_row[f'{attr_name}_{fairness_metric}_p'] = fairness_test['p_value']
                        result_row[f'{attr_name}_{fairness_metric}_significant'] = fairness_test['significant']
                        result_row[f'{attr_name}_{fairness_metric}_improvement'] = fairness_test.get('fairness_improvement', np.nan)
                    eopp_test = self.fairness_significance_test(
                        baseline_model, model, attr_name, 'equal_opportunity_diff'
                    )
                    result_row[f'{attr_name}_equal_opportunity_diff_p'] = eopp_test['p_value']
                    result_row[f'{attr_name}_equal_opportunity_diff_significant'] = eopp_test['significant']
                    result_row[f'{attr_name}_equal_opportunity_diff_improvement'] = eopp_test.get('fairness_improvement', np.nan)
            
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def print_cv_summary(self):
        print("="*80)
        print("CROSS-VALIDATION SUMMARY")
        print("="*80)
        
        for model_name, results in self.cv_results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy: {np.nanmean(results['accuracy']):.4f} ± {np.nanstd(results['accuracy']):.4f}")
            print(f"  F1-Score: {np.nanmean(results['f1']):.4f} ± {np.nanstd(results['f1']):.4f}")
            
            for attr_name, attr_results in results['fairness_metrics'].items():
                dp_scores = [r['demographic_parity_diff'] for r in attr_results if not np.isnan(r['demographic_parity_diff'])]
                eo_scores = [r['equalized_odds_diff'] for r in attr_results if not np.isnan(r['equalized_odds_diff'])]
                try:
                    import numpy as _np
                    eopp_scores = [r.get('equal_opportunity_diff') for r in attr_results if not _np.isnan(r.get('equal_opportunity_diff', _np.nan))]
                except Exception:
                    eopp_scores = []
                
                if dp_scores:
                    print(f"  {attr_name} DP Diff: {np.mean(dp_scores):.4f} ± {np.std(dp_scores):.4f}")
                if eo_scores:
                    print(f"  {attr_name} EO Diff: {np.mean(eo_scores):.4f} ± {np.std(eo_scores):.4f}")
                if eopp_scores:
                    print(f"  {attr_name} EOpp Diff: {np.mean(eopp_scores):.4f} ± {np.std(eopp_scores):.4f}")
        
        print("="*80) 

from src.statistics.bootstrap import bootstrap_cis, inject_label_noise
