from multiprocessing.process import parent_process
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class ModelInterpreter:
    def __init__(self):
        self.shap_explainers = {}
        self.lime_explainer = None
        self.feature_names = []
        self.interpretations = {}
   
    def setup_shap_explainer(self, model, X_train: np.ndarray,
                            feature_names: List[str], model_name: str = "model"):
        self.feature_names = feature_names
        try:
            background_size = min(2000, len(X_train))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_train), size=background_size, replace=False)
            background_data = X_train[idx]
            masker = shap.maskers.Independent(background_data)
            if hasattr(model, "predict_proba"):
                f = lambda X: model.predict_proba(X)[:, 1]
            else:
                f = model.predict

            explainer = shap.Explainer(f, masker)
            self.shap_explainers[model_name] = explainer
            print(f"SHAP explainer set up for {model_name}")
        except Exception as e:
            print(f"Error setting up SHAP for {model_name}: {e}")
            try:
                explainer = shap.LinearExplainer(model, background_data)
                self.shap_explainers[model_name] = explainer
                print(f"Linear SHAP explainer set up for {model_name}")
            except Exception as e2:
                print(f"Failed to set up any SHAP explainer for {model_name}: {e2}")
    
    def setup_lime_explainer(self, X_train: np.ndarray, feature_names: List[str],
                           mode: str = 'tabular'):
        self.feature_names = feature_names
        
        try:
            if mode == 'tabular':
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train,
                    feature_names=feature_names,
                    class_names=['<=50K', '>50K'],
                    mode='classification',
                    discretize_continuous=False
                )
                print("LIME explainer set up successfully")
            
        except Exception as e:
            print(f"Error setting up LIME explainer: {e}")
    
    def explain_with_shap(self, model_name: str, X_test: np.ndarray, 
                         max_samples: int = 100) -> Dict[str, Any]:
        if model_name not in self.shap_explainers:
            print(f"SHAP explainer not found for {model_name}")
            return {}
        
        explainer = self.shap_explainers[model_name]
        
        try:
            X_explain = X_test[:max_samples]
            
            shap_values = explainer(X_explain)

            if hasattr(shap_values, 'values'):
                vals = shap_values.values
                if vals.ndim == 3: 
                    values = vals[:, :, -1]
                else:
                    values = vals
            else:
                values = shap_values
            
            feature_importance = np.abs(values).mean(axis=0)
            
            summary = {
                'shap_values': values,
                'feature_importance': feature_importance,
                'feature_names': self.feature_names,
                'mean_abs_shap': feature_importance,
                'top_features': [(self.feature_names[i], importance) 
                               for i, importance in enumerate(feature_importance)],
                'X_explain': X_explain
            }
            
            summary['top_features'].sort(key=lambda x: x[1], reverse=True)
            
            self.interpretations[f"{model_name}_shap"] = summary
            print(f"SHAP analysis completed for {model_name}")
            
            return summary
            
        except Exception as e:
            print(f"Error in SHAP explanation for {model_name}: {e}")
            return {}
    
    def explain_with_lime(self, model, X_test: np.ndarray, 
                         sample_indices: List[int] = None, 
                         num_features: int = 10) -> Dict[str, Any]:
        if self.lime_explainer is None:
            print("LIME explainer not set up")
            return {}
        
        if sample_indices is None:
            sample_indices = [0, 1, 2]
        
        explanations = {}
        
        try:
            for idx in sample_indices:
                if idx < len(X_test):
                    exp = self.lime_explainer.explain_instance(
                        X_test[idx], 
                        model.predict_proba,
                        num_features=num_features
                    )
                    
                    feature_weights = exp.as_list()
                    
                    explanations[f'sample_{idx}'] = {
                        'feature_weights': feature_weights,
                        'prediction_proba': model.predict_proba([X_test[idx]])[0],
                        'explanation_object': exp
                    }
            
            print(f"LIME analysis completed for {len(explanations)} samples")
            return explanations
            
        except Exception as e:
            print(f"Error in LIME explanation: {e}")
            return {}
    
    def plot_shap_summary(self, model_name: str, save_path: str = None):
        shap_key = f"{model_name}_shap"
        if shap_key not in self.interpretations:
            print(f"SHAP interpretations not found for {model_name}")
            return
        
        summary = self.interpretations[shap_key]
        
        try:
            plt.figure(figsize=(10, 8))
            
            feature_importance = summary['feature_importance']
            feature_names = summary['feature_names']

            def _pretty(n: str) -> str:
                return n.replace('_', ' ', 1).replace('_', ' ')
            
            sorted_indices = np.argsort(feature_importance)[-15:]
            
            plt.barh(range(len(sorted_indices)), 
                    feature_importance[sorted_indices],
                    color='skyblue', alpha=0.7)
            plt.yticks(range(len(sorted_indices)), 
                      [_pretty(feature_names[i]) for i in sorted_indices])
            plt.xlabel('Mean |SHAP Value|')
            plt.title(f'SHAP Feature Importance - {model_name}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"SHAP plot saved to: {save_path}")
            
            try:
                plt.show()
            except:
                print("Plot display skipped (non-interactive mode)")
                
        except Exception as e:
            print(f"Error plotting SHAP summary: {e}")

    def plot_lime_explanations(self, model_name: str, explanations: Dict[str, Any], save_dir: str = None):
        if not explanations:
            return
        try:
            import matplotlib.pyplot as plt
            import os
            os.makedirs(save_dir or 'results/plots', exist_ok=True)
            for key, data in explanations.items():
                exp = data.get('explanation_object')
                if exp is None:
                    continue
                fig = exp.as_pyplot_figure()
                fig.set_size_inches(6, 4)
                fig.tight_layout()
                if save_dir:
                    safe_model = model_name.replace(' ', '_').replace('(', '').replace(')', '')
                    path = os.path.join(save_dir, f'lime_{safe_model}_{key}.png')
                    fig.savefig(path, dpi=200, bbox_inches='tight')
                    print(f"LIME plot saved to: {path}")
                try:
                    plt.show()
                except Exception:
                    pass
        except Exception as e:
            print(f"Error plotting LIME explanations: {e}")
    
    def compare_model_interpretations(self, model_names: List[str], 
                                    top_k: int = 10) -> pd.DataFrame:
        comparison_data = []
        
        for model_name in model_names:
            shap_key = f"{model_name}_shap"
            if shap_key in self.interpretations:
                summary = self.interpretations[shap_key]
                feature_importance = summary['feature_importance']
                feature_names = summary['feature_names']
                
                top_indices = np.argsort(feature_importance)[-top_k:][::-1]
                
                for rank, idx in enumerate(top_indices):
                    comparison_data.append({
                        'Model': model_name,
                        'Feature': feature_names[idx],
                        'Importance': feature_importance[idx],
                        'Rank': rank + 1
                    })
        
        return pd.DataFrame(comparison_data)
    
    def delta_shap(self, base_model: str, other_model: str, top_k: int = 10):
        a = self.interpretations.get(f"{base_model}_shap")
        b = self.interpretations.get(f"{other_model}_shap")
        if not a or not b:
            return {}
        names = a['feature_names']
        diff = (b['feature_importance'] - a['feature_importance'])
        order = np.argsort(diff)[::-1]
        rows = []
        for i in order[:top_k]:
            rows.append({'feature': names[i], 'delta_mean_abs_shap': float(diff[i])})
        return rows

    def analyze_bias_in_features(self, model_name: str, sensitive_features: List[str]) -> Dict[str, Dict]:
        key = f"{model_name}_importance"
        if key not in self.interpretations:
            return {}

        summary = self.interpretations[key]
        feature_names = list(summary['feature_names'])
        feature_importance = np.asarray(summary['feature_importance'])

        analysis = {}
        abs_all = np.abs(feature_importance)
        max_abs = float(abs_all.max()) if abs_all.size else 0.0
        n_feats = len(abs_all)

        for base in sensitive_features:
            idxs = [i for i, n in enumerate(feature_names)
                    if n == base or n.startswith(base + '_')]
            if not idxs:
                continue

            parts = {feature_names[i]: float(abs(feature_importance[i])) for i in idxs}
            agg = float(sum(parts.values()))
            rank = int((abs_all > agg).sum() + 1) if n_feats else 1
            percentile = 100.0 * (1.0 - rank / n_feats) if n_feats else 0.0

            analysis[base] = {
                'absolute_importance': agg,
                'relative_importance': (agg / max_abs) if max_abs else 0.0,
                'rank': rank,
                'percentile': percentile,
                'components': parent_process
            }

        return analysis
    
    def plot_feature_comparison(self, model_names: List[str], 
                              save_path: str = None):
        comparison_df = self.compare_model_interpretations(model_names)
        
        if comparison_df.empty:
            print("No interpretation data available for comparison")
            return
        
        pivot_df = comparison_df.pivot(index='Feature', columns='Model', values='Importance')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Feature Importance'})
        plt.title('Feature Importance Comparison Across Models')
        plt.xlabel('Models')
        plt.ylabel('Features')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature comparison plot saved to: {save_path}")
        
        try:
            plt.show()
        except:
            print("Plot display skipped (non-interactive mode)")
    
    def generate_interpretation_report(self, model_names: List[str], 
                                     sensitive_features: List[str] = None) -> str:
        if sensitive_features is None:
            sensitive_features = ['sex', 'race']
        
        report = []
        report.append("="*80)
        report.append("MODEL INTERPRETABILITY ANALYSIS REPORT")
        report.append("="*80)
        
        for model_name in model_names:
            shap_key = f"{model_name}_shap"
            if shap_key in self.interpretations:
                report.append(f"\n{model_name.upper()} ANALYSIS:")
                report.append("-" * 40)
                
                summary = self.interpretations[shap_key]
                feature_importance = summary['feature_importance']
                feature_names = summary['feature_names']
                
                top_indices = np.argsort(feature_importance)[-5:][::-1]
                report.append("\nTop 5 Most Important Features:")
                for i, idx in enumerate(top_indices):
                    report.append(f"  {i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
                
                if sensitive_features:
                    bias_analysis = self.analyze_bias_in_features(model_name, sensitive_features)
                    if bias_analysis:
                        report.append(f"\nSensitive Feature Analysis:")
                        for feature, analysis in bias_analysis.items():
                            report.append(f"  {feature}:")
                            report.append(f"    Rank: {analysis['rank']}")
                            report.append(f"    Percentile: {analysis['percentile']:.1f}%")
                            report.append(f"    Relative Importance: {analysis['relative_importance']:.3f}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def print_interpretation_summary(self, model_names: List[str]):
        print(self.generate_interpretation_report(model_names))


    def analyze_feature_importance(self, model, X_test: np.ndarray,
                                feature_names: List[str],
                                model_name: str = "model",
                                y: np.ndarray = None,
                                prefer_permutation: bool = True) -> Dict[str, Any]:
        self.feature_names = list(feature_names)

        try:
            use_perm = prefer_permutation or (
                hasattr(model, 'penalty') and getattr(model, 'penalty', None) == 'l1'
            )

            if use_perm:
                if y is None:
                    raise ValueError("analyze_feature_importance requires y (true labels) for permutation importance")
                y = np.asarray(y).ravel()
                supports_sklearn_pi = hasattr(model, 'fit') and callable(getattr(model, 'fit'))
                if supports_sklearn_pi:
                    try:
                        pi = permutation_importance(model, X_test, y,
                                                    scoring='accuracy',
                                                    n_repeats=10,
                                                    random_state=42)
                        feat_imp = pi.importances_mean
                    except Exception:
                        feat_imp = self._permutation_importance(model, X_test, y)
                else:
                    feat_imp = self._permutation_importance(model, X_test, y)
            elif hasattr(model, 'coef_') and getattr(model, 'coef_', None) is not None:
                coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                feat_imp = np.abs(np.asarray(coefficients).ravel())
            else:
                if y is None:
                    raise ValueError("analyze_feature_importance requires y (true labels) for fallback permutation importance")
                feat_imp = self._permutation_importance(model, X_test, y)

            n = min(len(self.feature_names), len(feat_imp))
            feats = self.feature_names[:n]
            imps  = np.asarray(feat_imp)[:n]

            top_features = list(zip(feats, imps))
            top_features.sort(key=lambda x: x[1], reverse=True)

            summary = {
                'feature_importance': imps,
                'feature_names': feats,
                'top_features': top_features
            }
            self.interpretations[f"{model_name}_importance"] = summary
            return summary

        except Exception as e:
            print(f"Error in feature importance analysis: {e}")
            return {}
    
    def _permutation_importance(self, model, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        if y is None:
            y = model.predict(X)
        
        baseline_score = np.mean(model.predict(X) == y)
        importance_scores = []
        
        for i in range(X.shape[1]):
            X_permuted = X.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            permuted_score = np.mean(model.predict(X_permuted) == y)
            importance_scores.append(baseline_score - permuted_score)
        
        return np.array(importance_scores)
    
    def compare_models(self, model_names: List[str], top_k: int = 10) -> pd.DataFrame:
        comparison_data = []
        
        for model_name in model_names:
            key = f"{model_name}_importance"
            if key in self.interpretations:
                summary = self.interpretations[key]
                top_features = summary['top_features'][:top_k]
                
                for rank, (feature, importance) in enumerate(top_features):
                    comparison_data.append({
                        'Model': model_name,
                        'Feature': feature,
                        'Importance': importance,
                        'Rank': rank + 1
                    })
        
        return pd.DataFrame(comparison_data)
    
    def plot_feature_importance(self, model_name: str, top_k: int = 15, save_path: str = None):
        key = f"{model_name}_importance"
        if key not in self.interpretations:
            print(f"No interpretation data for {model_name}")
            return
        
        summary = self.interpretations[key]
        top_features = summary['top_features'][:top_k]
        
        features, importances = zip(*top_features)

        def _pretty(n: str) -> str:
            return n.replace('_', ' ', 1).replace('_', ' ')
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances, color='skyblue', alpha=0.7)
        plt.yticks(range(len(features)), [_pretty(f) for f in features])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_k} Features - {model_name}')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        try:
            plt.show()
        except:
            print("Plot display skipped (non-interactive mode)")
    
    def analyze_sensitive_features(self, model_name: str, 
                                 sensitive_features: List[str]) -> Dict[str, Dict]:
        key = f"{model_name}_importance"
        if key not in self.interpretations:
            return {}
        
        summary = self.interpretations[key]
        feature_names = summary['feature_names']
        feature_importance = summary['feature_importance']
        
        analysis = {}
        for sensitive_feature in sensitive_features:
            if sensitive_feature in feature_names:
                idx = feature_names.index(sensitive_feature)
                importance = feature_importance[idx]
                rank = len(feature_importance) - np.searchsorted(
                    np.sort(feature_importance), importance
                )
                
                analysis[sensitive_feature] = {
                    'importance': importance,
                    'rank': rank,
                    'percentile': (len(feature_importance) - rank) / len(feature_importance) * 100
                }
        
        return analysis

    def shap_stability(self, model_name: str, X_test: np.ndarray, repeats: int = 5, top_k: int = 10):
        if model_name not in self.shap_explainers:
            return {}
        ranks = []
        for _ in range(repeats):
            sv = self.shap_explainers[model_name](X_test[:200])
            vals = sv.values if hasattr(sv, 'values') else sv
            imp = np.abs(vals).mean(axis=0).ravel()
            ranks.append(np.argsort(imp)[::-1][:top_k])
        from itertools import combinations
        j = []
        for i, jdx in combinations(range(repeats), 2):
            a, b = set(ranks[i]), set(ranks[jdx])
            j.append(len(a & b) / len(a | b) if (a | b) else 1.0)
        import numpy as _np
        return {'top_k': top_k, 'mean_jaccard': float(_np.mean(j)) if j else 1.0}