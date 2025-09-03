import numpy as np
from sklearn.linear_model import LogisticRegression
from .preprocessor import BiasPreprocessor
from .inprocessor import BiasInprocessor
from .postprocessor import BiasPostprocessor


class BiasMitigationPipeline:
    

    def __init__(self, random_state: int = 42):
        
        self.random_state = random_state
        self.preprocessor = BiasPreprocessor(random_state)
        self.inprocessor = BiasInprocessor(random_state)
        self.postprocessor = BiasPostprocessor()
        self.models = {}

    def train_baseline_model(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        model.fit(X_train, y_train)
        self.models['baseline'] = model
        return model

    def train_preprocessing_model(self, X_train, y_train, sensitive_train, method='reweight', attr_name=None):
        
        if method == 'reweight':
            X_clean, y_clean, sensitive_clean, sample_weights = self.preprocessor.reweight_samples(X_train, y_train, sensitive_train, attr_name=attr_name)
            model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            model.fit(X_clean, y_clean, sample_weight=sample_weights)
        elif method == 'resample':
            X_resampled, y_resampled, _ = self.preprocessor.resample_data(X_train, y_train, sensitive_train)
            model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            model.fit(X_resampled, y_resampled)
        else:
            raise ValueError("Method must be 'reweight' or 'resample'")
        self.models[f'preprocessing_{method}'] = model
        return model

    def train_inprocessing_model(self, X_train: np.ndarray, y_train: np.ndarray,
                                 sensitive_train: np.ndarray, method: str = 'fairlearn_eg'):
        
        if method == 'adversarial':
            model = self.inprocessor.adversarial_debiasing(X_train, y_train, sensitive_train)
        elif method == 'fairlearn_eg':
            model = self.inprocessor.fairlearn_exponentiated_gradient(
                X_train, y_train, sensitive_train, constraint='equalized_odds'
            )
            model.fit(X_train, y_train, sensitive_features=sensitive_train)
        else:
            raise ValueError("Method must be 'adversarial', or 'fairlearn_eg'")
        self.models[f'inprocessing_{method}'] = model
        return model


