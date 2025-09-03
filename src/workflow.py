import os
import sys
import random
import logging
import io
import numpy as np
import pandas as pd
from datetime import datetime

                                                   
import matplotlib
matplotlib.use('Agg')

sys.path.append('src')

                 
from data_loader import AdultDataLoader
from fairness_metrics import FairnessEvaluator
from bias_mitigation import BiasMitigationPipeline
from statistical_testing import StatisticalTester, bootstrap_cis, inject_label_noise
from interpretability import ModelInterpreter
from hyperparameter_tuning import HyperparameterTuner
from sklearn.linear_model import LogisticRegression
from src.constants import (
    BASELINE_NAME, PREPROC_NAME, PREPROC_RACE_NAME,
    ADVERSARIAL_NAME, ADVERSARIAL_RACE_NAME,
    FAIRLEARN_EG_NAME, FAIRLEARN_EG_RACE_NAME,
    FAIRLEARN_DP_NAME, FAIRLEARN_DP_RACE_NAME,
    POST_PROCESSING_EO_SEX_NAME, POST_PROCESSING_EO_RACE_NAME,
    POST_PROCESSING_CAL_SEX_NAME, POST_PROCESSING_CAL_RACE_NAME,
    FAIRLEARN_EG_BOTH_NAME,
)
from src.postprocessing_utils import postproc_proba
from src.reporting import save_results, generate_plots, generate_executive_summary


def seed_everything(seed: int = 42) -> bool:
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        return True
    except Exception as e:
        print(f"PyTorch not available or failed to initialize ({e}). Adversarial debiasing will be skipped.")
        return False


def setup_logging():
    os.makedirs('results', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('results/fairness_analysis.log', mode='w')
        ],
        force=True
    )
                                                                                
    class StreamToLogger(io.TextIOBase):
        def __init__(self, logger, level=logging.INFO):
            self.logger = logger
            self.level = level
        def write(self, buf):
            if not buf:
                return 0
                                                          
            buf = buf.replace('\r', '')
            for line in buf.rstrip().splitlines():
                if line:
                    self.logger.log(self.level, line)
            return len(buf)
        def flush(self):
            pass
    sys.stdout = StreamToLogger(logging.getLogger('stdout'), logging.INFO)
                                                                                      
    sys.stderr = StreamToLogger(logging.getLogger('stderr'), logging.INFO)


def setup_directories():
    for d in ['results', 'results/plots', 'results/reports']:
        os.makedirs(d, exist_ok=True)


def run_pipeline():
    setup_directories(); setup_logging(); seed_everything(42)

    pipeline = BiasMitigationPipeline(); race_pipeline = BiasMitigationPipeline()
    evaluator = FairnessEvaluator(); tester = StatisticalTester(); interpreter = ModelInterpreter()
    tuner = HyperparameterTuner()

          
    data_loader = AdultDataLoader(
        drop_columns=['fnlwgt'],
        native_country_mode='none',
        keep_columns=['age','education','relationship','race','sex','native-country']
    );
    data_dict = data_loader.prepare_data()
    X_train = data_dict['X_train']; X_test = data_dict['X_test']
    y_train = data_dict['y_train']; y_test = data_dict['y_test']
    sensitive_train = data_dict['sensitive_train']; sensitive_test = data_dict['sensitive_test']

                     
    baseline_model = LogisticRegression(random_state=42)
    tuning_results = tuner.comprehensive_tuning(
        model=baseline_model,
        X=pd.DataFrame(X_train),
        y=pd.Series(y_train),
        sensitive_attrs={k: pd.Series(v) for k, v in sensitive_train.items()},
        methods=['bayesian'], fairness_aware=True, n_trials=100, cv=5
    )
    best_tuned_model = tuning_results['best_overall']['best_estimator']

                          
    baseline_model = best_tuned_model; baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_prob = baseline_model.predict_proba(X_test)[:, 1]
    base_train_prob = baseline_model.predict_proba(X_train)[:, 1]
    baseline_results = evaluator.comprehensive_evaluation(
        y_test, baseline_pred, sensitive_test, baseline_prob, BASELINE_NAME
    )

                              
    sex_train = sensitive_train['sex']; race_train = sensitive_train['race']
    sex_race_train = np.array([f"{s}×{r}" for s, r in zip(sex_train, race_train)])
    sex_race_test  = np.array([f"{s}×{r}" for s, r in zip(
        sensitive_test['sex'], sensitive_test['race']
    )])
    sensitive_test_plus = dict(sensitive_test); sensitive_test_plus['sex×race'] = sex_race_test
                                                                                                   

                       
    preprocessing_model = pipeline.train_preprocessing_model(
        X_train, y_train, sex_train, method='reweight', attr_name='sex')
    try:
        adversarial_model = pipeline.train_inprocessing_model(
            X_train, y_train, sex_train, method='adversarial')
    except ImportError:
        adversarial_model = None
    fairlearn_eg_model = pipeline.train_inprocessing_model(
        X_train, y_train, sex_train, method='fairlearn_eg')
    fairlearn_dp_model = pipeline.inprocessor.fairlearn_exponentiated_gradient(
        X_train, y_train, sex_train, constraint='demographic_parity')
    fairlearn_dp_model.fit(X_train, y_train, sensitive_features=sex_train)

    preprocessing_race_model = race_pipeline.train_preprocessing_model(
        X_train, y_train, race_train, method='reweight', attr_name='race')
    try:
        adversarial_race_model = race_pipeline.train_inprocessing_model(
            X_train, y_train, race_train, method='adversarial')
    except ImportError:
        adversarial_race_model = None
    fairlearn_eg_race_model = race_pipeline.train_inprocessing_model(
        X_train, y_train, race_train, method='fairlearn_eg')
    fairlearn_dp_race_model = race_pipeline.inprocessor.fairlearn_exponentiated_gradient(
        X_train, y_train, race_train, constraint='demographic_parity')
    fairlearn_dp_race_model.fit(X_train, y_train, sensitive_features=race_train)

                                  
    fairlearn_eg_both = pipeline.inprocessor.fairlearn_exponentiated_gradient(
        X_train, y_train, sex_race_train, constraint='equalized_odds')
    fairlearn_eg_both.fit(X_train, y_train, sensitive_features=sex_race_train)

                          
    for name, model in [
        (BASELINE_NAME, baseline_model), (PREPROC_NAME, preprocessing_model),
        (PREPROC_RACE_NAME, preprocessing_race_model), (ADVERSARIAL_NAME, adversarial_model),
        (ADVERSARIAL_RACE_NAME, adversarial_race_model), (FAIRLEARN_EG_NAME, fairlearn_eg_model),
        (FAIRLEARN_EG_RACE_NAME, fairlearn_eg_race_model), (FAIRLEARN_DP_NAME, fairlearn_dp_model),
        (FAIRLEARN_DP_RACE_NAME, fairlearn_dp_race_model)
    ]:
        if model is not None:
            try:
                interpreter.setup_shap_explainer(model, X_train, data_dict['feature_names'], name)
            except Exception:
                pass
    interpreter.setup_lime_explainer(X_train, data_dict['feature_names'])

                             
    postprocessor = pipeline.postprocessor; race_postprocessor = race_pipeline.postprocessor
    sex_test = sensitive_test['sex']; race_test = sensitive_test['race']
    eo_post_sex = postprocessor.equalized_odds_via_fairlearn(baseline_model, X_train, y_train, sex_train)
    y_pred_post_eq_sex  = eo_post_sex.predict(X_test, sensitive_features=sex_test).astype(int)
    y_prob_post_eq_sex  = postproc_proba(eo_post_sex,  X_test, sex_test)
    cal_factors_sex = postprocessor.fit_group_calibration(y_train, base_train_prob, sex_train)
    y_pred_post_cal_sex, y_prob_post_cal_sex = postprocessor.apply_group_calibration(
        baseline_prob, sex_test, factors=cal_factors_sex, return_proba=True)
    cal_factors_race = race_postprocessor.fit_group_calibration(y_train, base_train_prob, race_train)
    y_pred_post_cal_race, y_prob_post_cal_race = race_postprocessor.apply_group_calibration(
        baseline_prob, race_test, factors=cal_factors_race, return_proba=True)
    eo_post_race = race_postprocessor.equalized_odds_via_fairlearn(baseline_model, X_train, y_train, race_train)
    y_pred_post_eq_race = eo_post_race.predict(X_test, sensitive_features=race_test).astype(int)
    y_prob_post_eq_race = postproc_proba(eo_post_race, X_test, race_test)

                                                                               
    all_results = []

                                                                        
    try:
        print("[CCF] Starting Calibration-Constrained Fairness search for Sex...")
                                                                                                
        _, y_prob_cal_sex = postprocessor.apply_group_calibration(
            baseline_prob, sex_test, factors=cal_factors_sex, return_proba=True
        )
        ccf_sex = postprocessor.optimize_group_thresholds(
            y_true=y_test, y_prob=y_prob_cal_sex, sensitive_attr=sex_test,
            dp_max=0.12, eo_max=0.12, ece_max=0.07, grid=np.linspace(0.3, 0.7, 9), random_state=42
        )
        res_ccf_sex = evaluator.comprehensive_evaluation(
            y_test,
            ccf_sex['y_pred'],
            {'sex': sex_test, 'race': race_test},
            ccf_sex.get('y_prob', y_prob_cal_sex),
            model_name='Post-processing CCF (Sex)'
        )
        try:
            thr_log = ccf_sex.get('thresholds', {})
            met_log = ccf_sex.get('metrics', {})
            print(f"[CCF] Sex thresholds: {thr_log}; metrics: {met_log}")
        except Exception:
            pass
    except Exception as e:
        print(f"[CCF] Sex CCF failed: {e}. Proceeding without CCF (Sex).")
        ccf_sex = None
        res_ccf_sex = None

                                                                         
    try:
        print("[CCF] Starting Calibration-Constrained Fairness search for Race...")
                                                                                                
        _, y_prob_cal_race = race_postprocessor.apply_group_calibration(
            baseline_prob, race_test, factors=cal_factors_race, return_proba=True
        )
        ccf_race = race_postprocessor.optimize_group_thresholds(
            y_true=y_test, y_prob=y_prob_cal_race, sensitive_attr=race_test,
            dp_max=0.12, eo_max=0.12, ece_max=0.07, grid=np.linspace(0.3, 0.7, 9), random_state=42
        )
        res_ccf_race = evaluator.comprehensive_evaluation(
            y_test,
            ccf_race['y_pred'],
            {'sex': sex_test, 'race': race_test},
            ccf_race.get('y_prob', y_prob_cal_race),
            model_name='Post-processing CCF (Race)'
        )
        try:
            thr_log_r = ccf_race.get('thresholds', {})
            met_log_r = ccf_race.get('metrics', {})
            print(f"[CCF] Race thresholds: {thr_log_r}; metrics: {met_log_r}")
        except Exception:
            pass
    except Exception as e:
        print(f"[CCF] Race CCF failed: {e}. Proceeding without CCF (Race).")
        ccf_race = None
        res_ccf_race = None

                     
    models_to_evaluate = [
        (BASELINE_NAME, baseline_model, None, None),
        (PREPROC_NAME, preprocessing_model, None, None),
        *([(ADVERSARIAL_NAME, adversarial_model, None, None)] if adversarial_model is not None else []),
        (FAIRLEARN_EG_NAME, fairlearn_eg_model, None, None),
        (FAIRLEARN_DP_NAME, fairlearn_dp_model, None, None),
        (POST_PROCESSING_EO_SEX_NAME,  baseline_model, y_pred_post_eq_sex,  y_prob_post_eq_sex),
        (POST_PROCESSING_CAL_SEX_NAME, baseline_model, y_pred_post_cal_sex, y_prob_post_cal_sex),
        (PREPROC_RACE_NAME, preprocessing_race_model, None, None),
        *([(ADVERSARIAL_RACE_NAME, adversarial_race_model, None, None)] if adversarial_race_model is not None else []),
        (FAIRLEARN_EG_RACE_NAME, fairlearn_eg_race_model, None, None),
        (FAIRLEARN_DP_RACE_NAME, fairlearn_dp_race_model, None, None),
        (POST_PROCESSING_EO_RACE_NAME,  baseline_model, y_pred_post_eq_race,  y_prob_post_eq_race),
        (POST_PROCESSING_CAL_RACE_NAME, baseline_model, y_pred_post_cal_race, y_prob_post_cal_race),
        (FAIRLEARN_EG_BOTH_NAME, fairlearn_eg_both, None, None),
        *([('Post-processing CCF (Sex)', baseline_model, ccf_sex['y_pred'], ccf_sex.get('y_prob', y_prob_cal_sex))] if ccf_sex is not None else []),
        *([('Post-processing CCF (Race)', baseline_model, ccf_race['y_pred'], ccf_race.get('y_prob', y_prob_cal_race))] if ccf_race is not None else []),
    ]
    try:
        print("[Eval] Models to evaluate:" , [m[0] for m in models_to_evaluate])
    except Exception:
        pass

                                                                     
    sex_pre_cache  = pipeline.preprocessor.transform_test_data_multi(X_test, y_test, sensitive_test)
    race_pre_cache = race_pipeline.preprocessor.transform_test_data_multi(X_test, y_test, sensitive_test)

    for model_name, model, y_pred_override, y_prob_override in models_to_evaluate:
        if y_pred_override is not None:
            y_pred = y_pred_override; y_prob = y_prob_override
            X_eval, y_eval, sens_eval = X_test, y_test, sensitive_test_plus
        else:
            if "Preprocessing" in model_name:
                X_eval, y_eval, sens_eval = sex_pre_cache if "(Sex)" in model_name else race_pre_cache
            else:
                X_eval, y_eval, sens_eval = X_test, y_test, sensitive_test_plus
            y_pred = model.predict(X_eval)
            y_prob = model.predict_proba(X_eval)[:, 1] if hasattr(model, 'predict_proba') else None

                                                                                
        results = evaluator.comprehensive_evaluation(y_eval, y_pred, sens_eval, y_prob, model_name)
        try:
            imp = interpreter.analyze_feature_importance(
                model, X_eval, data_dict['feature_names'], model_name, y=y_eval
            )
            if imp:
                interpreter.interpretations[f"{model_name}_importance"] = imp
        except Exception:
            pass
        results['predictions'] = y_pred; results['prediction_probabilities'] = y_prob
        results['_y_eval'] = y_eval; results['_sensitive_eval'] = sens_eval
        all_results.append(results)

        if y_prob is not None and not model_name.startswith("Post-processing"):
                                                                                                        
            interpreter.explain_with_shap(model_name, X_eval, max_samples=200)
                                                                                          
            try:
                lime_expl = interpreter.explain_with_lime(model, X_eval, sample_indices=[3, 15, 42])
                interpreter.plot_lime_explanations(model_name, lime_expl, save_dir='results/plots')
            except Exception:
                pass

                                                                   

                           
    ci_rows = []
    for r in all_results:
        try:
                     
            cis_sex = bootstrap_cis(r.get('_y_eval', y_test), r.get('predictions'), r.get('prediction_probabilities'),
                                    r.get('_sensitive_eval', sensitive_test), n_boot=1000, seed=42, attribute='sex')
                      
            cis_race = bootstrap_cis(r.get('_y_eval', y_test), r.get('predictions'), r.get('prediction_probabilities'),
                                     r.get('_sensitive_eval', sensitive_test), n_boot=1000, seed=42, attribute='race')
        except Exception:
            cis_sex = {'accuracy_CI': (np.nan, np.nan, np.nan), 'f1_CI': (np.nan, np.nan, np.nan),
                       'dp_diff_CI': (np.nan, np.nan, np.nan), 'eo_diff_CI': (np.nan, np.nan, np.nan), 'ece_diff_CI': (np.nan, np.nan, np.nan)}
            cis_race = cis_sex
        def as_row(prefix, triple):
            lo, mu, hi = triple; return {f'{prefix}_lo': lo, f'{prefix}_mean': mu, f'{prefix}_hi': hi}
        row = {'model_name': r['model_name']}
        row.update(as_row('accuracy',     cis_sex.get('accuracy_CI', (np.nan,)*3)))
        row.update(as_row('f1',           cis_sex.get('f1_CI', (np.nan,)*3)))
        row.update(as_row('sex_dp_diff',  cis_sex.get('dp_diff_CI', (np.nan,)*3)))
        row.update(as_row('sex_eo_diff',  cis_sex.get('eo_diff_CI', (np.nan,)*3)))
        row.update(as_row('sex_ece_diff', cis_sex.get('ece_diff_CI', (np.nan,)*3)))
        row.update(as_row('race_dp_diff',  cis_race.get('dp_diff_CI', (np.nan,)*3)))
        row.update(as_row('race_eo_diff',  cis_race.get('eo_diff_CI', (np.nan,)*3)))
        row.update(as_row('race_ece_diff', cis_race.get('ece_diff_CI', (np.nan,)*3)))
        ci_rows.append(row)
    pd.DataFrame(ci_rows).to_csv('results/reports/bootstrap_cis.csv', index=False)

                                    
                                                                            
    tester.cross_validate_model(baseline_model,            X_train, y_train, sensitive_train, BASELINE_NAME)
    tester.cross_validate_model(fairlearn_dp_model,        X_train, y_train, sensitive_train, FAIRLEARN_DP_NAME)
    tester.cross_validate_model(fairlearn_eg_model,        X_train, y_train, sensitive_train, FAIRLEARN_EG_NAME)
    tester.cross_validate_model(preprocessing_model,       X_train, y_train, sensitive_train, PREPROC_NAME,
                                preprocessor=pipeline.preprocessor, attr_for_preproc='sex')
    tester.cross_validate_model(preprocessing_race_model,  X_train, y_train, sensitive_train, PREPROC_RACE_NAME,
                                preprocessor=race_pipeline.preprocessor, attr_for_preproc='race')
    compare_models = [FAIRLEARN_DP_NAME, FAIRLEARN_EG_NAME, PREPROC_NAME, PREPROC_RACE_NAME]
    try:
        comp_df = tester.comprehensive_comparison(BASELINE_NAME, compare_models)
        comp_df.to_csv('results/reports/statistical_comparisons.csv', index=False)
    except Exception:
        pass

                           
    generate_plots(all_results, evaluator, interpreter)
    balanced_scores = []
    for result in [r for r in all_results if not r['model_name'].startswith('Post-processing')]:
        accuracy = result['performance']['accuracy']
        inter = result.get('fairness', {}).get('intersectional', {})
        if 'intersectional_disparities' in inter:
            disp = inter['intersectional_disparities'].get('max_disparity', np.nan)
        else:
            diffs = []
            for attr_name, metrics in result.get('fairness', {}).items():
                if attr_name == 'intersectional':
                    continue
                dp = metrics.get('demographic_parity', {}).get('demographic_parity_difference', np.nan)
                eo = metrics.get('equalized_odds', {}).get('equalized_odds_difference', np.nan)
                if not np.isnan(dp): diffs.append(abs(dp))
                if not np.isnan(eo): diffs.append(abs(eo))
            disp = np.nanmean(diffs) if diffs else np.nan
        fairness = 1 - np.clip(disp, 0, 1) if not np.isnan(disp) else 0.5
        balanced_scores.append((result['model_name'], 0.6 * accuracy + 0.4 * fairness))
    best_balanced = max(balanced_scores, key=lambda x: x[1])

                                                                                           
    method_metadata = {
        'seed': 42,
        'reweighting': {
            'sex': getattr(pipeline.preprocessor, 'reweight_factors', {}),
            'race': getattr(race_pipeline.preprocessor, 'reweight_factors', {})
        }
    }
    try:
        method_metadata['shap_stability'] = {
            BASELINE_NAME: interpreter.shap_stability(BASELINE_NAME, X_test, repeats=5, top_k=10)
        }
    except Exception:
        method_metadata['shap_stability'] = {}

    if ccf_sex is not None or ccf_race is not None:
        method_metadata.setdefault('ccf', {}).update({
            'sex_thresholds': (ccf_sex.get('thresholds', {}) if ccf_sex is not None else {}),
            'race_thresholds': (ccf_race.get('thresholds', {}) if ccf_race is not None else {}),
            'constraints': {'dp_max': 0.12, 'eo_max': 0.12, 'ece_max': 0.07}
        })

    generate_executive_summary(data_dict, all_results, best_balanced, method_metadata)
    save_results(all_results, tester.cv_results, best_balanced, method_metadata)

                                                                                                 
    try:
        noise_rows = []
        for rate, seed in [(0.05, 1), (0.10, 2)]:
            y_train_noisy = inject_label_noise(y_train.copy(), rate=rate, seed=seed)
            baseline_noisy = best_tuned_model.__class__(**best_tuned_model.get_params())
            baseline_noisy.fit(X_train, y_train_noisy)
            pred = baseline_noisy.predict(X_test)
            prob = baseline_noisy.predict_proba(X_test)[:, 1]
            res = evaluator.comprehensive_evaluation(y_test, pred, sensitive_test, prob, f"Baseline (noise={int(rate*100)}%)")
            sex_dp = res['fairness']['sex']['demographic_parity']['demographic_parity_difference'] if 'sex' in res['fairness'] else np.nan
            sex_eo = res['fairness']['sex']['equalized_odds']['equalized_odds_difference'] if 'sex' in res['fairness'] else np.nan
            noise_rows.append({
                'noise_rate': rate,
                'accuracy': res['performance']['accuracy'],
                'f1': res['performance']['f1'],
                'sex_dp_diff': sex_dp,
                'sex_eo_diff': sex_eo
            })
        import pandas as _pd
        _pd.DataFrame(noise_rows).to_csv('results/reports/robustness_results.csv', index=False)

                                                                                          
        try:
            from src.mitigation.inprocessor import BiasInprocessor
            _inproc = BiasInprocessor()
            X_train_dec = _inproc._decorrelate_features(X_train, sex_train)
            X_test_dec  = _inproc._decorrelate_features(X_test, sex_test)
            base_dec = best_tuned_model.__class__(**best_tuned_model.get_params())
            base_dec.fit(X_train_dec, y_train)
            pred_dec = base_dec.predict(X_test_dec)
            prob_dec = base_dec.predict_proba(X_test_dec)[:, 1] if hasattr(base_dec, 'predict_proba') else None
            res_dec = evaluator.comprehensive_evaluation(y_test, pred_dec, sensitive_test, prob_dec, 'Baseline (decorrelated sex)')
            all_results.append(res_dec)
        except Exception:
            pass
    except Exception:
        pass

                                                                                    
    try:
        ablations = []
                                                                         
        X_test_pre, y_test_pre, sens_test_pre = pipeline.preprocessor.transform_test_data_multi(X_test, y_test, sensitive_test)
        yhat_pre = preprocessing_model.predict(X_test_pre)
        p_pre = preprocessing_model.predict_proba(X_test_pre)[:, 1] if hasattr(preprocessing_model, 'predict_proba') else None
        ablations.append(evaluator.comprehensive_evaluation(y_test_pre, yhat_pre, sens_test_pre, p_pre, 'Ablation: PreOnly(Sex)'))

                                            
        yhat_in = fairlearn_eg_model.predict(X_test)
        p_in = fairlearn_eg_model.predict_proba(X_test)[:, 1] if hasattr(fairlearn_eg_model, 'predict_proba') else None
        ablations.append(evaluator.comprehensive_evaluation(y_test, yhat_in, sensitive_test, p_in, 'Ablation: InOnly(FairlearnEG)'))

                                                           
        ablations.append(evaluator.comprehensive_evaluation(y_test, y_pred_post_eq_sex, sensitive_test, y_prob_post_eq_sex, 'Ablation: PostOnly(EO, Sex)'))

                                                          
        import pandas as _pd
        abl_rows = []
        for r in ablations:
            row = {
                'model_name': r['model_name'],
                'accuracy': r['performance']['accuracy'],
                'f1': r['performance']['f1'],
                'sex_dp_diff': r['fairness'].get('sex', {}).get('demographic_parity', {}).get('demographic_parity_difference', None),
                'sex_eo_diff': r['fairness'].get('sex', {}).get('equalized_odds', {}).get('equalized_odds_difference', None)
            }
            abl_rows.append(row)
        _pd.DataFrame(abl_rows).to_csv('results/reports/ablations_results.csv', index=False)
    except Exception:
        pass

                                                               
    try:
        post = pipeline.postprocessor
        if 'sex' in sensitive_test:
            post.reliability_diagram(y_test, baseline_prob, np.asarray(sensitive_test['sex']), save_path='results/plots/reliability_sex_baseline.png')
        if 'race' in sensitive_test:
            post.reliability_diagram(y_test, baseline_prob, np.asarray(sensitive_test['race']), save_path='results/plots/reliability_race_baseline.png')
                                                      
    except Exception:
        pass

    return all_results


