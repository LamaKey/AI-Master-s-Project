import os
import sys
import json
from datetime import datetime
import numpy as np
import pandas as pd

def save_results(all_results, cv_results, best_balanced, method_metadata=None):
                                                                     
                                                                                      
    results_df = []
    def _safe(dp,*keys):
        from collections.abc import Mapping
        cur = dp
        for k in keys:
            if isinstance(cur, Mapping) and k in cur:
                cur = cur[k]
            else:
                return np.nan
        return cur if isinstance(cur,(int,float,np.floating)) else np.nan

    for result in all_results:
        row = {
            'model_name': result['model_name'],
            'accuracy': result['performance']['accuracy'],
            'precision': result['performance']['precision'],
            'recall': result['performance']['recall'],
            'f1': result['performance']['f1'],
            'auc': result['performance']['auc'],
            'sex_demographic_parity_diff': _safe(result['fairness'],'sex','demographic_parity','demographic_parity_difference'),
            'sex_equalized_odds_diff' : _safe(result['fairness'],'sex','equalized_odds','equalized_odds_difference'),
            'sex_equal_opportunity_diff' : _safe(result['fairness'],'sex','equal_opportunity','equal_opportunity_difference'),
            'sex_calibration_diff'    : _safe(result['fairness'],'sex','calibration','calibration_difference'),
            'race_demographic_parity_diff': _safe(result['fairness'],'race','demographic_parity','demographic_parity_difference'),
            'race_equalized_odds_diff'    : _safe(result['fairness'],'race','equalized_odds','equalized_odds_difference'),
            'race_equal_opportunity_diff' : _safe(result['fairness'],'race','equal_opportunity','equal_opportunity_difference'),
            'race_calibration_diff'       : _safe(result['fairness'],'race','calibration','calibration_difference'),
            'intersectional_max_disparity': result['fairness'].get('intersectional', {}).get('intersectional_disparities', {}).get('max_disparity', 0),
            'intersectional_positive_rate_disparity': result['fairness'].get('intersectional', {}).get('intersectional_disparities', {}).get('positive_rate_disparity', 0),
            'intersectional_accuracy_disparity': result['fairness'].get('intersectional', {}).get('intersectional_disparities', {}).get('accuracy_disparity', 0)
        }
        results_df.append(row)

    os.makedirs('results/reports', exist_ok=True)
    pd.DataFrame(results_df).to_csv('results/reports/comprehensive_results.csv', index=False)

                                                                      
    try:
        def _pick(name):
            return next((r for r in all_results if r['model_name'] == name), None)
        from src.constants import BASELINE_NAME, POST_PROCESSING_EO_SEX_NAME
        baseline = _pick(BASELINE_NAME)
        ccf      = _pick('Post-processing CCF (Sex)')
        post_eo  = _pick(POST_PROCESSING_EO_SEX_NAME)
        rows = []
        for nam, res in [(BASELINE_NAME, baseline), ('CCF (Sex)', ccf), (POST_PROCESSING_EO_SEX_NAME, post_eo)]:
            if res is None:
                continue
            sex_dp = res['fairness'].get('sex', {}).get('demographic_parity', {}).get('demographic_parity_difference', None)
            sex_eo = res['fairness'].get('sex', {}).get('equalized_odds', {}).get('equalized_odds_difference', None)
            rows.append({
                'Model': nam,
                'Accuracy': res['performance']['accuracy'],
                'AUC': res['performance'].get('auc', None),
                'Sex_DP_Diff': sex_dp,
                'Sex_EO_Diff': sex_eo
            })
        if rows:
            dfp = pd.DataFrame(rows)
            dfp = dfp[['Model','Accuracy','AUC','Sex_DP_Diff','Sex_EO_Diff']].sort_values('Sex_EO_Diff')
            dfp.to_csv('results/reports/policy_comparison_sex.csv', index=False)
    except Exception:
        pass

                                                                
    try:
        if method_metadata and isinstance(method_metadata, dict):
            ccf = method_metadata.get('ccf', {})
            if ccf:
                os.makedirs('results/reports', exist_ok=True)
                with open('results/reports/ccf_thresholds.json', 'w') as _jf:
                    json.dump(ccf, _jf, indent=2)
    except Exception:
        pass

    if cv_results:
        rows = []
        for name, res in cv_results.items():
            rows.append({
                'model': name,
                'acc_mean': float(np.nanmean(res['accuracy'])),
                'acc_std':  float(np.nanstd(res['accuracy'])),
                'f1_mean':  float(np.nanmean(res['f1'])),
                'f1_std':   float(np.nanstd(res['f1'])),
            })
        pd.DataFrame(rows).to_csv('results/reports/cross_validation_results.csv', index=False)

                                                                           

    def thin(res):
        perf = res.get('performance', {})
        fair = res.get('fairness', {})
        inter = fair.get('intersectional', {})
        inter_disp = inter.get('intersectional_disparities', {})
        return {
            'model_name': res.get('model_name'),
            'performance': {
                k: float(perf.get(k)) if perf.get(k) is not None else None
                for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']
            },
            'fairness': {
                'sex': {
                    'dp_diff': fair.get('sex', {}).get('demographic_parity', {}).get('demographic_parity_difference', None),
                    'eo_diff': fair.get('sex', {}).get('equalized_odds', {}).get('equalized_odds_difference', None),
                    'cal_diff': fair.get('sex', {}).get('calibration', {}).get('calibration_difference', None),
                },
                'race': {
                    'dp_diff': fair.get('race', {}).get('demographic_parity', {}).get('demographic_parity_difference', None),
                    'eo_diff': fair.get('race', {}).get('equalized_odds', {}).get('equalized_odds_difference', None),
                    'cal_diff': fair.get('race', {}).get('calibration', {}).get('calibration_difference', None),
                },
                                                              
                'intersectional': {
                    'max_disparity': inter_disp.get('max_disparity', None),
                    'positive_rate_disparity': inter_disp.get('positive_rate_disparity', None),
                    'accuracy_disparity': inter_disp.get('accuracy_disparity', None),
                    'num_groups': inter.get('num_groups', None),
                    'attributes': inter.get('attributes', None),
                }
            }
        }

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'model_results': [thin(r) for r in all_results],
        'cross_validation_results': cv_results,
        'best_balanced_model': best_balanced,
        'method_metadata': method_metadata or {},
        'summary': {
            'total_models_evaluated': len(all_results),
            'best_accuracy_model': max(all_results, key=lambda x: x['performance']['accuracy'])['model_name'],
            'best_fairness_model': (
                min([r for r in all_results if 'intersectional' in r.get('fairness', {}) 
                     and 'intersectional_disparities' in r['fairness']['intersectional']], 
                    key=lambda x: x['fairness']['intersectional']['intersectional_disparities']['max_disparity'])['model_name']
                if any('intersectional' in r.get('fairness', {}) and 'intersectional_disparities' in r['fairness']['intersectional'] for r in all_results)
                else max(all_results, key=lambda x: x['performance']['f1'])['model_name']
            )
        }
    }

    json_report = convert_numpy(report)
    with open('results/reports/detailed_analysis_report.json', 'w') as f:
        json.dump(json_report, f, indent=2)

    print("Results saved to:")
    print("  - results/reports/comprehensive_results.csv")
    print("  - results/reports/cross_validation_results.csv")
    print("  - results/reports/detailed_analysis_report.json")
    print("  - results/plots/ (visualizations)")

def generate_plots(all_results, evaluator, interpreter):
                                                                                     
                                                                               
    import matplotlib.pyplot as plt
    os.makedirs("results/plots", exist_ok=True)
    print("Creating fairness comparison plot...")
    evaluator.plot_fairness_comparison(
        all_results, 
        save_path='results/plots/fairness_comparison.png'
    )

    print("Creating performance vs fairness plot(s)...")
                                                                                 
    import re
    def _slug(s: str) -> str:
        s = s.replace('×', 'x').replace('–','-').replace('—','-')
        s = s.replace(' ', '_')
        return re.sub(r'[^A-Za-z0-9_.-]+', '_', s)
    present_attrs = []
    if all_results:
        present_attrs = [k for k in all_results[0].get('fairness', {}).keys() if k not in ('intersectional','native-country')]
    for attr in present_attrs:
        out = f'results/plots/performance_vs_fairness_{_slug(attr)}.png'
        evaluator.plot_performance_vs_fairness(
            all_results,
            fairness_metric=f'{attr}_DP_Diff',
            save_path=out
        )

    print("Creating interpretability plots...")
    for result in all_results:
        model_name = result['model_name']
        safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        shap_key = f"{model_name}_shap"
        if shap_key in interpreter.interpretations:
            try:
                interpreter.plot_shap_summary(
                    model_name,
                    save_path=f'results/plots/shap_summary_{safe_name}.png'
                )
            except Exception as e:
                print(f"SHAP plot for {model_name} failed: {e}")

                                               
    try:
        lime_explanations = interpreter.explain_with_lime
    except Exception:
        lime_explanations = None

                                                                           
    try:
                                                                            
                                                              
        from src.constants import BASELINE_NAME
                                                                         
        if getattr(interpreter, 'lime_explainer', None) is not None:
                                                                                     
            pass
    except Exception:
        pass

    print("All visualizations completed.")

    sens_csv = []
    for result in all_results:
        model_name = result['model_name']
        analysis = interpreter.analyze_bias_in_features(model_name, ['sex','race','native-country'])
        for feat, stats in analysis.items():
            sens_csv.append({'model': model_name,
                            'feature': feat,
                            **stats})
    if sens_csv:
        pd.DataFrame(sens_csv).to_csv('results/reports/sensitive_feature_importance.csv', index=False)
        print("sensitive_feature_importance.csv written.")

                                                                        
    try:
        import pandas as _pd
        import matplotlib.pyplot as plt
        cisd = 'results/reports/bootstrap_cis.csv'
        if os.path.exists(cisd):
            df = _pd.read_csv(cisd)
            for metric_prefix in ['accuracy', 'f1', 'sex_dp_diff', 'sex_eo_diff', 'sex_ece_diff']:
                lo, mu, hi = f'{metric_prefix}_lo', f'{metric_prefix}_mean', f'{metric_prefix}_hi'
                if all(c in df.columns for c in [lo, mu, hi]):
                    plt.figure(figsize=(8,4))
                    xs = range(len(df))
                    plt.errorbar(xs, df[mu], yerr=[df[mu]-df[lo], df[hi]-df[mu]], fmt='o')
                    plt.xticks(xs, df['model_name'], rotation=45, ha='right')
                    plt.title(f'Bootstrap CI: {metric_prefix}')
                    plt.tight_layout()
                    outp = f"results/plots/ci_{metric_prefix}.png"
                    plt.savefig(outp, dpi=200, bbox_inches='tight')
    except Exception:
        pass

def generate_executive_summary(data_dict, all_results, best_balanced, method_metadata=None):                               
    import numpy as np
    import pandas as pd
    from datetime import datetime
                                              
    req_path = os.path.join(os.getcwd(), 'requirements.txt')
    env_packages = []
    if os.path.exists(req_path):
        try:
            with open(req_path, 'r') as rf:
                env_packages = [line.strip() for line in rf if line.strip() and not line.startswith('#')]
        except Exception:
            env_packages = []
    def intersectional_max_disparity(res):
        inter = res.get('fairness', {}).get('intersectional', {})
        if 'intersectional_disparities' in inter:
            return inter['intersectional_disparities'].get('max_disparity', np.nan)
        return np.nan

    def avg_attr_disparity(res):
        diffs = []
        for attr in ('sex', 'race', 'education'):
            m = res.get('fairness', {}).get(attr, {})
            dp = m.get('demographic_parity', {}).get('demographic_parity_difference', np.nan)
            if pd.notna(dp):
                diffs.append(abs(float(dp)))
        return float(np.nanmean(diffs)) if diffs else np.nan

    def overall_disparity(res):
        d = intersectional_max_disparity(res)
        if not np.isnan(d):
            return d
        d = avg_attr_disparity(res)
        return d if not np.isnan(d) else np.nan

    best_accuracy = max(all_results, key=lambda x: x['performance']['accuracy'])
    disparities = [(r, overall_disparity(r)) for r in all_results]
    disparities = [(r, d) for r, d in disparities if not np.isnan(d)]
    best_fairness = min(disparities, key=lambda t: t[1])[0] if disparities else max(all_results, key=lambda x: x['performance']['f1'])

    best_acc_disp = overall_disparity(best_accuracy)
    best_fair_disp = overall_disparity(best_fairness)

    n_train = data_dict.get('X_train', np.empty((0,))).shape[0]
    n_test  = data_dict.get('X_test',  np.empty((0,))).shape[0]
                                                                         
    rqs = [
        {
            'id': 'RQ1',
            'question': 'Can we reduce sex Equalized Odds disparity while retaining strong discrimination?',
            'success_criterion': 'Sex EO diff ≤ 0.10 with AUC ≥ 0.84 on test.'
        },
        {
            'id': 'RQ2',
            'question': 'Does pre-, in-, or post-processing yield the best fairness-accuracy trade-off?',
            'success_criterion': 'Best-balanced score among methods with no >3% absolute accuracy drop from baseline for DP/EO ≤ 0.12.'
        },
        {
            'id': 'RQ3',
            'question': 'Are models calibrated similarly across sex/race groups?',
            'success_criterion': 'Group-wise ECE differences ≤ 0.05 and near-diagonal reliability.'
        }
    ]

    methods = [r['model_name'] for r in all_results]
    metrics = ['Accuracy','Precision','Recall','F1','AUC',
               'Demographic Parity Diff (by sex/race/native-country)',
               'Equalized Odds Diff (by sex/race/native-country)',
               'Equal Opportunity Diff (by sex/race/native-country)',
               'Calibration Diff/ECE (by sex/race/native-country)',
               'Intersectional disparities (sex×race)']

    ethics = {
        'privileged_groups': {
            'sex': 'Male',
            'race': 'White'
        },
        'justification': 'Matches common practice in Adult dataset analyses and aligns with historical advantage patterns; used only for auditing/constraints.',
        'harms_tradeoffs': [
            'Mitigations may reduce accuracy on privileged groups to improve parity.',
            'Intersectional disparities can remain even when marginal DP/EO improves.',
            'Sensitive attributes are excluded from X to avoid disparate treatment; proxies may persist.'
        ],
        'dataset_biases': 'Adult dataset reflects historical income inequities; labels encode structural bias; results are context-limited.'
    }

    reproducibility = {
        'seeds': {'python_numpy_torch': 42},
        'environment': {
            'python': sys.version if 'sys' in globals() else '3.x',
            'packages': env_packages
        },
        'how_to_run': 'Activate venv, install requirements, then run: python main.py',
        'artifacts': {
            'figures': 'results/plots/',
            'tables': 'results/reports/*.csv and detailed_analysis_report.json'
        }
    }

    extras_scope = {
        'additional_models': 'Primary focus is Logistic Regression. RF/XGB not included to keep scope tight; can be added as future work.',
        'external_validation': 'Not performed; Adult is the single benchmark used.',
        'ui': 'Not included; command-line pipeline with saved artifacts.',
        'error_analysis': 'Local explanations via LIME for three representative cases; can expand to more cases if required.'
    }

    summary = {
        'executive_summary': {
            'project_overview': 'AI Fairness Analysis for Hiring Systems',
            'analysis_date': datetime.now().isoformat(),
            'dataset': {
                'name': 'UCI Adult',
                'post_clean_samples': int(n_train + n_test),
                'features_used': data_dict.get('feature_names', []),
                'sensitive_attributes': ['sex','race']
            },
            'methods': methods,
            'metrics': metrics,
            'research_questions': rqs,
            'key_findings': {
                'best_accuracy_model': {
                    'name': best_accuracy['model_name'],
                    'accuracy': best_accuracy['performance']['accuracy'],
                    'fairness_disparity': None if np.isnan(best_acc_disp) else best_acc_disp
                },
                'best_fairness_model': {
                    'name': best_fairness['model_name'],
                    'accuracy': best_fairness['performance']['accuracy'],
                    'fairness_disparity': None if np.isnan(best_fair_disp) else best_fair_disp
                },
                'recommended_model': {
                    'name': best_balanced[0],
                    'balanced_score': best_balanced[1],
                    'rationale': 'Best trade-off between performance and fairness'
                }
            },
            'method_metadata': method_metadata or {},
            'bias_analysis': {
                'sex_bias_detected': any(
                    abs(r.get('fairness', {}).get('sex', {}).get('demographic_parity', {}).get('demographic_parity_difference', 0)) > 0.1
                    for r in all_results if 'sex' in r.get('fairness', {})
                ),
                'race_bias_detected': any(
                    abs(r.get('fairness', {}).get('race', {}).get('demographic_parity', {}).get('demographic_parity_difference', 0)) > 0.1
                    for r in all_results if 'race' in r.get('fairness', {})
                ),
                'intersectional_bias_severity': (
                    'High' if any(
                        (res.get('fairness', {}).get('intersectional', {})
                         .get('intersectional_disparities', {})
                         .get('max_disparity', 0)) > 0.8
                        for res in all_results
                    ) else 'Medium'
                )
            },
            'ethics_and_risk': ethics,
            'reproducibility': reproducibility,
            'scope_and_extras': extras_scope,
            'next_steps': [
                'Evaluate RF/XGB as alternative families with the same fairness protocol.',
                'Extend LIME cases and add group-conditional SHAP summaries.',
                'Consider external dataset for validation if within scope/time.'
            ]
        }
    }

    os.makedirs('results/reports', exist_ok=True)
    with open('results/reports/executive_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Executive summary saved to: results/reports/executive_summary.json")
    return summary


