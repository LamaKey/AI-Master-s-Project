import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

class AdultDataLoader:
    
    def __init__(self, test_size=0.2, random_state=42, drop_columns=None,
                 category_min_count: int = 100,
                 native_country_mode: str = 'us_other',
                 native_country_top_k: int = 5,
                 keep_columns=None):
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.sensitive_features = ['sex', 'race']
        self.drop_columns = set(drop_columns or ['fnlwgt'])
        self.category_min_count = int(category_min_count)
        self.native_country_mode = native_country_mode
        self.native_country_top_k = int(native_country_top_k)
        self.keep_columns = keep_columns
        
    def load_data(self):
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        
        train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
        
        try:
            os.makedirs('data', exist_ok=True)
            cache_file = os.path.join('data', 'adult_dataset_cache.csv')
            if os.path.exists(cache_file):
                print("Loading cached dataset...")
                data = pd.read_csv(cache_file)
                print(f"Dataset loaded from cache. Shape: {data.shape}")
                return data
            
            print("Downloading dataset from UCI repository...")
            import socket
            socket.setdefaulttimeout(30)
            
            train_data = pd.read_csv(train_url, names=column_names, skipinitialspace=True)
            test_data = pd.read_csv(test_url, names=column_names, skipinitialspace=True, skiprows=1)
            
            data = pd.concat([train_data, test_data], ignore_index=True)
            
            data.to_csv(cache_file, index=False)
            print(f"Dataset loaded successfully and cached. Shape: {data.shape}")
            return data
            
        except Exception as e:
            print(f"Error loading data from URL: {e}")
            print("Using synthetic data for demonstration...")
            logging.getLogger(__name__).warning(
                "Adult dataset download failed – falling back to synthetic data; "
                "fairness numbers will NOT match the paper."
            )
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        print("Creating synthetic data for demonstration...")
        np.random.seed(self.random_state)
        
        n_samples = 10000
        data = {
            'age': np.random.normal(39, 13, n_samples).astype(int).clip(17, 90),
            'workclass': np.random.choice(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 
                                         'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay'], n_samples),
            'fnlwgt': np.random.normal(189778, 105549, n_samples).astype(int).clip(12285, 1484705),
            'education': np.random.choice(['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
                                         'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
                                         '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'], n_samples),
            'education-num': np.random.randint(1, 17, n_samples),
            'marital-status': np.random.choice(['Married-civ-spouse', 'Divorced', 'Never-married',
                                              'Separated', 'Widowed', 'Married-spouse-absent'], n_samples),
            'occupation': np.random.choice(['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                                          'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                                          'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                                          'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                                          'Armed-Forces'], n_samples),
            'relationship': np.random.choice(['Wife', 'Own-child', 'Husband', 'Not-in-family',
                                            'Other-relative', 'Unmarried'], n_samples),
            'race': np.random.choice(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
                                    'Other', 'Black'], n_samples, p=[0.85, 0.05, 0.03, 0.02, 0.05]),
            'sex': np.random.choice(['Female', 'Male'], n_samples, p=[0.33, 0.67]),
            'capital-gain': np.random.exponential(1077, n_samples).astype(int).clip(0, 99999),
            'capital-loss': np.random.exponential(87, n_samples).astype(int).clip(0, 4356),
            'hours-per-week': np.random.normal(40, 12, n_samples).astype(int).clip(1, 99),
            'native-country': np.random.choice(['United-States', 'Cambodia', 'England', 'Puerto-Rico',
                                              'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)',
                                              'India', 'Japan', 'Greece', 'South', 'China'], n_samples,
                                             p=[0.891, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.015, 0.01, 0.005, 0.01, 0.014])
        }
        
        income_score = (
            (data['age'] > 35).astype(int) * 0.3 +
            (data['education-num'] > 12).astype(int) * 0.4 +
            (data['hours-per-week'] > 40).astype(int) * 0.2 +
            (np.array(data['sex']) == 'Male').astype(int) * 0.15 +
            (np.array(data['race']) == 'White').astype(int) * 0.1
        )
        
        income_prob = np.clip(income_score / income_score.max(), 0.1, 0.9)
        
        data['income'] = np.where(np.random.random(n_samples) < income_prob, '>50K', '<=50K')
        
        return pd.DataFrame(data)
    
    def remove_outliers(self, df, z_thresh=3.5):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) /
                        df[numeric_cols].std(ddof=0))
        mask = (z_scores < z_thresh).all(axis=1)
        dropped = len(df) - mask.sum()
        if dropped:
            print(f"Removed {dropped} rows as numeric outliers (z>{z_thresh}).")
        return df[mask]

    def clean_data(self, data):
        data = data.replace('?', np.nan)

        data = self.remove_outliers(data)
        
        required_cols = ['income']
        if self.keep_columns:
            required_cols += [c for c in self.keep_columns if c in data.columns]
        else:
            required_cols += ['native-country'] if 'native-country' in data.columns else []
        data = data.dropna(subset=list(dict.fromkeys(required_cols)))
        
        data['income'] = data['income'].str.replace('.', '', regex=False)
        
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].str.strip()

        data = data.drop_duplicates().reset_index(drop=True)

        if self.category_min_count and self.category_min_count > 0:
            for col in data.select_dtypes(include=['object']).columns:
                if col in self.sensitive_features or (self.keep_columns and col in self.keep_columns):
                    continue
                vc = data[col].value_counts(dropna=False)
                rare = vc[vc < self.category_min_count].index
                if len(rare) > 0:
                    data[col] = data[col].apply(lambda x: 'Other' if x in rare else x)
        
        print(f"Data cleaned. Shape after cleaning: {data.shape}")
        return data
    
    def encode_features(self, data):
        X = data.drop('income', axis=1)
        y = data['income']
        
        y_encoded = (y == '>50K').astype(int)
        
        sensitive_attrs = {}
        for attr in self.sensitive_features:
            if attr in X.columns:
                sensitive_attrs[attr] = X[attr].copy()
        drop_cols = [c for c in self.sensitive_features if c in X.columns]
        drop_cols.extend([c for c in self.drop_columns if c in X.columns])
        if drop_cols:
            X = X.drop(columns=drop_cols)
        
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        self.feature_names = X.columns.tolist()
        
        return X, y_encoded, sensitive_attrs, categorical_columns
    
    def encode_features_post_split(self, X_train, X_test, categorical_columns):
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in categorical_columns if c in X_train.columns]

        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', ohe, cat_cols),
            ],
            remainder='drop'
        )

        X_train_enc = self.preprocessor.fit_transform(X_train)
        X_test_enc  = self.preprocessor.transform(X_test)

        def _clean(name: str) -> str:
            return (
                name.replace('__', '_')
                    .replace(' ', '_')
                    .replace('-', '_')
                    .replace('/', '_')
            )
        num_names = [_clean(n) for n in num_cols]
        cat_raw = self.preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
        cat_names = [_clean(n.split('__', 1)[-1]) for n in cat_raw]
        self.feature_names = num_names + cat_names

        return X_train_enc, X_test_enc
    
    def prepare_data(self):
        raw_data = self.load_data()
        clean_data = self.clean_data(raw_data)

        if self.keep_columns:
            keep = [c for c in self.keep_columns if c in clean_data.columns]
            keep = list(dict.fromkeys(keep + ['income']))
            clean_data = clean_data[keep]
        
        X, y, sensitive_attrs, categorical_columns = self.encode_features(clean_data)
        
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, np.arange(len(X)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        sensitive_train, sensitive_test = {}, {}
        for attr_name, attr_values in sensitive_attrs.items():
            s = attr_values.reset_index(drop=True) if hasattr(attr_values, "reset_index") else pd.Series(attr_values)
            sensitive_train[attr_name] = (s.iloc[idx_train]).reset_index(drop=True).values
            sensitive_test[attr_name]  = (s.iloc[idx_test]).reset_index(drop=True).values

        X_train_encoded, X_test_encoded = self.encode_features_post_split(
            X_train, X_test, categorical_columns
        )

        X_train_scaled = X_train_encoded
        X_test_scaled  = X_test_encoded
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_df': pd.DataFrame(X_train_scaled, columns=self.feature_names),
            'X_test_df': pd.DataFrame(X_test_scaled, columns=self.feature_names),
            'sensitive_train': sensitive_train,
            'sensitive_test': sensitive_test,
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders,
            'scaler': None,
            'train_idx': idx_train,
            'test_idx':  idx_test,
            'raw_data': clean_data
        }
    
    def get_dataset_info(self, data_dict):
        print("="*50)
        print("DATASET INFORMATION")
        print("="*50)
        
        print(f"Training samples: {len(data_dict['y_train'])}")
        print(f"Test samples: {len(data_dict['y_test'])}")
        print(f"Number of features: {len(data_dict['feature_names'])}")
        
        print(f"\nTarget distribution (training):")
        print(f"  <=50K: {(data_dict['y_train'] == 0).sum()} ({(data_dict['y_train'] == 0).mean():.2%})")
        print(f"  >50K:  {(data_dict['y_train'] == 1).sum()} ({(data_dict['y_train'] == 1).mean():.2%})")
        
        for attr_name in self.sensitive_features:
            if attr_name in data_dict['sensitive_train']:
                print(f"\n{attr_name.title()} distribution (training):")
                dist = pd.Series(data_dict['sensitive_train'][attr_name]).value_counts()
                for value, count in dist.items():
                    print(f"  {value}: {count} ({count/len(data_dict['y_train']):.2%})")
        
        print("="*50) 