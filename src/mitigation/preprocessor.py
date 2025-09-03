import numpy as np
import pandas as pd
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from aif360.datasets import StandardDataset
    from aif360.algorithms.preprocessing import Reweighing
    _AIF360_AVAILABLE = True
except Exception:
    _AIF360_AVAILABLE = False

class BiasPreprocessor:
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.reweight_factors = {}
        self.train_attr = None
        self._use_reweight_only = False

    def reweight_samples(self, X, y, sensitive_attr, attr_name: str = None, privileged_value=None):
        try:
            self.train_attr = attr_name
            unique_values = np.unique(sensitive_attr)
            print(f"Sensitive attribute values: {unique_values}")
            if privileged_value is None:
                if len(unique_values) == 2:
                    privileged_value = unique_values[1]
                else:
                    from collections import Counter
                    privileged_value = Counter(sensitive_attr).most_common(1)[0][0]
            print(f"Privileged group for {attr_name or 'attr'} -> {privileged_value}")
            
            if len(unique_values) == 2:
                if privileged_value is None:
                    privileged_value = unique_values[1]
                print(f"Encoded as binary: 0={unique_values[0]}, 1={unique_values[1]} (privileged={privileged_value})")
            else:
                if privileged_value is None:
                    from collections import Counter
                    counts = Counter(sensitive_attr)
                    privileged_value = counts.most_common(1)[0][0]
                print(f"Multi-class detected. Privileged group: {privileged_value} vs others")
            
            y_flat          = np.asarray(y).ravel()
            sensitive_array = np.asarray(sensitive_attr).ravel()

            unique_values = np.unique(sensitive_attr)
            if len(unique_values) == 2:
                other = next(v for v in unique_values if v != privileged_value)
                decode = {1: str(privileged_value), 0: str(other)}
            else:
                decode = {1: str(privileged_value), 0: f"Not-{privileged_value}"}

            df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
            df["label"] = y_flat

            prot = (sensitive_array == privileged_value).astype(int)
            self._prot_mapping = {"privileged": privileged_value, "decode": decode}
            df["prot"] = prot
            df["row_id"] = np.arange(len(df))

            before_rows = len(df)
            df = df.dropna()
            df = df[df['prot'].isin([0, 1])]
            after_rows = len(df)
            if len(df) < len(X) * 0.8:
                print(f"AIF360 would drop too much data ({len(df)}/{len(X)}), using simple reweighting")
                self.reweight_factors = {
                    "mean": 1.0,
                    "std": 0.0,
                    "min": 1.0,
                    "max": 1.0,
                    "original_size": int(len(X)),
                    "cleaned_size": int(len(df)),
                    "dropped_rows": int(len(X) - len(df)),
                    "used": "simple_reweight_fallback"
                }
                return self._simple_reweight(X, y, sensitive_attr)
            if not _AIF360_AVAILABLE:
                print("AIF360 not available – falling back to simple reweighting.")
                return self._simple_reweight(X, y, sensitive_attr)

            feature_cols = [c for c in df.columns if c not in ["label", "prot"]]
            if "row_id" not in feature_cols:
                feature_cols = feature_cols + ["row_id"]

            aif_data = StandardDataset(
                df,
                label_name="label",
                favorable_classes=[1],
                protected_attribute_names=["prot"],
                privileged_classes=[[1]],
                features_to_keep=feature_cols
            )

            unprivileged_groups = [{"prot": 0}]
            privileged_groups   = [{"prot": 1}]
            rw = Reweighing(unprivileged_groups=unprivileged_groups,
                            privileged_groups=privileged_groups)
            transformed_dataset = rw.fit_transform(aif_data)
            X_aif = transformed_dataset.features
            row_id_col = -1
            row_ids = X_aif[:, row_id_col].astype(int)

            raw_weights = transformed_dataset.instance_weights.astype(float)
            raw_weights = np.clip(raw_weights, 0.25, 4.0)
            raw_weights /= raw_weights.mean()

            sample_weights = np.ones(len(X), dtype=float)
            sample_weights[row_ids] = raw_weights

            self.reweight_factors = {
                "mean": float(sample_weights.mean()),
                "std": float(sample_weights.std()),
                "min": float(sample_weights.min()),
                "max": float(sample_weights.max()),
                "original_size": len(X),
                "cleaned_size": int(len(row_ids)),
                "dropped_rows": int(len(X) - len(row_ids)),
                "used": "aif360_reweighing"
            }

            print("[Preprocess-Train] Using original feature matrix with reweighting (no AIF360 feature transform).")
            self._use_reweight_only = True
            return X, y_flat, sensitive_array, sample_weights

        except Exception as e:
            print(f"AIF360 reweighting failed: {e}")
            print("Falling back to simple reweighting...")
            return self._simple_reweight(X, y, sensitive_attr)
    
    def _simple_reweight(self, X: np.ndarray, y: np.ndarray, sensitive_attr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sample_weights = np.ones(len(X))
        groups = np.unique(sensitive_attr)
        overall_pos_rate = np.mean(y)
        print(f"Target positive rate for all groups: {overall_pos_rate:.3f}")
        for group in groups:
            group_mask = sensitive_attr == group
            group_pos_rate = np.mean(y[group_mask])
            group_size = np.sum(group_mask)
            if group_pos_rate > 0:
                pos_weight = overall_pos_rate / group_pos_rate
                neg_weight = (1 - overall_pos_rate) / (1 - group_pos_rate) if group_pos_rate < 1 else 1
                group_pos_mask = group_mask & (y == 1)
                group_neg_mask = group_mask & (y == 0)
                sample_weights[group_pos_mask] *= pos_weight
                sample_weights[group_neg_mask] *= neg_weight
                print(f"{group}: size={group_size}, pos_rate={group_pos_rate:.3f}, pos_weight={pos_weight:.3f}, neg_weight={neg_weight:.3f}")
            else:
                print(f"{group}: size={group_size}, pos_rate={group_pos_rate:.3f} (no positive samples)")
        sample_weights = sample_weights / np.mean(sample_weights)
        self.reweight_factors = {
            "mean": float(sample_weights.mean()),
            "std": float(sample_weights.std()),
            "min": float(sample_weights.min()),
            "max": float(sample_weights.max()),
            "original_size": len(X),
            "cleaned_size": len(X)
        }
        print(f"Reweighting complete. Weight stats: mean={self.reweight_factors['mean']:.3f}, std={self.reweight_factors['std']:.3f}")
        self._use_reweight_only = True
        return X, y, sensitive_attr, sample_weights
    
    def transform_test_data(self, X_test: np.ndarray, y_test: np.ndarray,
                            sensitive_test: np.ndarray, return_mask: bool=False) -> Tuple:
        if (not _AIF360_AVAILABLE) or getattr(self, "_use_reweight_only", False):
            return (X_test, y_test, sensitive_test, np.ones(len(X_test), dtype=bool)) if return_mask else (X_test, y_test, sensitive_test)
        try:
            df = pd.DataFrame(X_test, columns=[f"f{i}" for i in range(X_test.shape[1])])
            df["label"] = y_test
            if hasattr(self, "_prot_mapping"):
                st = np.asarray(sensitive_test).ravel()
                df["prot"] = (st == self._prot_mapping["privileged"]).astype(int)
            else:
                df["prot"] = sensitive_test
            mask_notna = ~df.isna().any(axis=1).to_numpy()
            mask_prot  = df['prot'].isin([0, 1]).to_numpy()
            retain_mask = mask_notna & mask_prot
            df = df.loc[retain_mask]
            feature_cols = [c for c in df.columns if c not in ["label", "prot"]]
            aif_data = StandardDataset(
                df,
                label_name="label",
                favorable_classes=[1],
                protected_attribute_names=["prot"],
                privileged_classes=[[1]],
                features_to_keep=feature_cols
            )
            X_test_clean = aif_data.features
            y_test_clean = aif_data.labels.ravel()
            sensitive_test_clean = aif_data.protected_attributes[:, 0]
            try:
                print(f"[Preprocess-Test] Features: raw={X_test.shape[1]} → cleaned={X_test_clean.shape[1]}")
            except Exception:
                pass
            if return_mask:
                return X_test_clean, y_test_clean, sensitive_test_clean, retain_mask
            else:
                return X_test_clean, y_test_clean, sensitive_test_clean
        except Exception as e:
            print(f"Test data transformation failed: {e}")
            print("Using original test data...")
            if return_mask:
                return X_test, y_test, sensitive_test, np.ones(len(X_test), dtype=bool)
            else:
                return X_test, y_test, sensitive_test
    
    def transform_test_data_multi(self, X_test: np.ndarray, y_test: np.ndarray,
                                  sensitive_test_dict: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
        try:
            key = None
            if getattr(self, 'train_attr', None) and self.train_attr in sensitive_test_dict:
                key = self.train_attr
            elif 'sex' in sensitive_test_dict:
                key = 'sex'
            else:
                key = next(iter(sensitive_test_dict.keys()))
            X_clean, y_clean, st_clean, mask = self.transform_test_data(
                X_test, y_test, sensitive_test_dict[key], return_mask=True
            )
            if len(X_clean) != len(X_test):
                sensitive_clean_dict = {k: np.asarray(v)[mask] for k, v in sensitive_test_dict.items()}
            else:
                sensitive_clean_dict = {k: np.asarray(v) for k, v in sensitive_test_dict.items()}
            sensitive_clean_dict[key] = st_clean
            try:
                if key == self.train_attr and hasattr(self, "_prot_mapping") and "decode" in self._prot_mapping:
                    dec = self._prot_mapping["decode"]
                    st_clean_int = np.asarray(st_clean).astype(int)
                    sensitive_clean_dict[key] = np.array([dec.get(int(v), str(v)) for v in st_clean_int])
            except Exception:
                pass
            return X_clean, y_clean, sensitive_clean_dict
        except Exception as e:
            print(f"Multi-attribute test transformation failed: {e}")
            return X_test, y_test, sensitive_test_dict

    def resample_data(self, X: np.ndarray, y: np.ndarray, 
                     sensitive_attr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        df = pd.DataFrame(X)
        df['target'] = y
        df['sensitive'] = sensitive_attr
        min_size = float('inf')
        for group in np.unique(sensitive_attr):
            for target in [0, 1]:
                group_class_size = len(df[(df['sensitive'] == group) & (df['target'] == target)])
                if group_class_size > 0:
                    min_size = min(min_size, group_class_size)
        from sklearn.utils import resample
        resampled_dfs = []
        for group in np.unique(sensitive_attr):
            for target in [0, 1]:
                group_class_df = df[(df['sensitive'] == group) & (df['target'] == target)]
                if len(group_class_df) > 0:
                    if len(group_class_df) >= min_size:
                        resampled_df = resample(group_class_df, n_samples=min_size, random_state=self.random_state)
                    else:
                        resampled_df = resample(group_class_df, n_samples=min_size, replace=True, random_state=self.random_state)
                    resampled_dfs.append(resampled_df)
        final_df = pd.concat(resampled_dfs, ignore_index=True)
        X_resampled = final_df.drop(['target', 'sensitive'], axis=1).values
        y_resampled = final_df['target'].values
        sensitive_resampled = final_df['sensitive'].values
        return X_resampled, y_resampled, sensitive_resampled


