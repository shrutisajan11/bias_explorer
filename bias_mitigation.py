import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class BiasMitigator:
    """
    Comprehensive bias mitigation toolkit with multiple techniques
    """
    
    def __init__(self):
        self.mitigation_techniques = {
            'resampling': 'Data Resampling (Pre-processing)',
            'reweighting': 'Instance Reweighting (Pre-processing)', 
            'adversarial': 'Adversarial Debiasing (In-processing)',
            'fairness_constraints': 'Fairness Constraints (In-processing)',
            'threshold_optimization': 'Threshold Optimization (Post-processing)',
            'calibration': 'Prediction Calibration (Post-processing)'
        }
    
    def demographic_parity_resampling(self, X, y, sensitive_attr, strategy='undersample'):
        """
        Balance dataset to achieve demographic parity
        """
        # Input validation
        if len(X) == 0:
            raise ValueError("Input dataset is empty")
        
        df = pd.DataFrame(X)
        df['target'] = y
        df['sensitive'] = sensitive_attr
        
        # Calculate current distribution
        group_counts = df.groupby(['sensitive', 'target']).size().unstack(fill_value=0)
        
        # Check if we have data for all groups
        if group_counts.empty:
            raise ValueError("No valid groups found in the data")
        
        if strategy == 'undersample':
            # Find minimum group size, but ensure it's at least 1
            min_size = max(1, group_counts[group_counts > 0].min().min())
            
            balanced_dfs = []
            for sensitive_val in df['sensitive'].unique():
                for target_val in df['target'].unique():
                    subset = df[(df['sensitive'] == sensitive_val) & (df['target'] == target_val)]
                    if len(subset) > 0:
                        # Ensure we sample at least 1 but not more than available
                        sample_size = min(len(subset), max(1, min_size))
                        sampled = subset.sample(n=sample_size, random_state=42)
                        balanced_dfs.append(sampled)
            
            if not balanced_dfs:
                raise ValueError("No valid subsets found for undersampling")
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            
        elif strategy == 'oversample':
            # Find maximum group size, but handle edge cases
            max_size = group_counts.max().max()
            if max_size == 0:
                raise ValueError("All groups have zero samples")
            
            # Set minimum oversample size
            min_oversample_size = max(10, max_size // 2)  # At least 10 samples or half of max
            
            balanced_dfs = []
            for sensitive_val in df['sensitive'].unique():
                for target_val in df['target'].unique():
                    subset = df[(df['sensitive'] == sensitive_val) & (df['target'] == target_val)]
                    if len(subset) > 0:
                        # Oversample with replacement, but use reasonable size
                        n_samples = max(min_oversample_size, max_size)
                        sampled = subset.sample(n=n_samples, replace=True, random_state=42)
                        balanced_dfs.append(sampled)
            
            if not balanced_dfs:
                raise ValueError("No valid subsets found for oversampling")
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'undersample' or 'oversample'")
        
        # Final validation
        if balanced_df.empty:
            raise ValueError("Resampling resulted in empty dataset")
        
        # Shuffle the balanced dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        X_balanced = balanced_df.drop(['target', 'sensitive'], axis=1).values
        y_balanced = balanced_df['target'].values
        sensitive_balanced = balanced_df['sensitive'].values
        
        print(f"Resampling complete: {len(X)} -> {len(X_balanced)} samples")
        
        return X_balanced, y_balanced, sensitive_balanced
    
    def calculate_fairness_weights(self, y, sensitive_attr):
        """
        Calculate instance weights to promote fairness
        """
        if len(y) == 0:
            raise ValueError("Cannot calculate weights for empty dataset")
        
        df = pd.DataFrame({'target': y, 'sensitive': sensitive_attr})
        
        # Calculate group proportions
        group_props = df.groupby(['sensitive', 'target']).size()
        total_props = df.groupby('target').size()
        
        weights = np.ones(len(y))
        
        for i, (target_val, sensitive_val) in enumerate(zip(y, sensitive_attr)):
            group_size = group_props.get((sensitive_val, target_val), 1)
            total_size = total_props.get(target_val, 1)
            
            # Inverse weighting - smaller groups get higher weights
            if group_size > 0:
                weights[i] = total_size / (group_size * len(df['sensitive'].unique()))
            else:
                weights[i] = 1.0  # Default weight for edge cases
        
        # Normalize weights to prevent extreme values
        weights = np.clip(weights, 0.1, 10.0)  # Clip to reasonable range
        weights = weights / np.mean(weights)
        
        return weights
    
    def adversarial_debiasing_loss(self, y_true, y_pred, sensitive_attr, lambda_fairness=1.0):
        """
        Custom loss function for adversarial debiasing
        """
        if len(y_true) == 0:
            return 0.0
        
        # Standard classification loss
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Prevent log(0)
        classification_loss = -np.mean(y_true * np.log(y_pred_clipped) + 
                                     (1 - y_true) * np.log(1 - y_pred_clipped))
        
        # Fairness loss - minimize correlation between predictions and sensitive attribute
        sensitive_encoded = pd.get_dummies(sensitive_attr).values
        fairness_loss = 0
        
        for i in range(sensitive_encoded.shape[1]):
            correlation = np.corrcoef(y_pred, sensitive_encoded[:, i])[0, 1]
            if not np.isnan(correlation):
                fairness_loss += abs(correlation)
        
        total_loss = classification_loss + lambda_fairness * fairness_loss
        return total_loss

class FairClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper classifier with built-in fairness constraints
    """
    
    def __init__(self, base_estimator=None, fairness_constraint='demographic_parity', 
                 lambda_fairness=1.0, sensitive_features=None):
        self.base_estimator = base_estimator or LogisticRegression(random_state=42)
        self.fairness_constraint = fairness_constraint
        self.lambda_fairness = lambda_fairness
        self.sensitive_features = sensitive_features
        self.weights_ = None
        
    def fit(self, X, y, sensitive_attr=None):
        """
        Fit classifier with fairness constraints
        """
        if len(X) == 0:
            raise ValueError("Cannot fit on empty dataset")
        
        if sensitive_attr is None and self.sensitive_features is not None:
            sensitive_attr = X[:, self.sensitive_features]
        
        # Calculate fairness weights
        if sensitive_attr is not None:
            self.weights_ = self._calculate_fairness_weights(y, sensitive_attr)
        else:
            self.weights_ = np.ones(len(y))
        
        # Fit base estimator with weights
        if hasattr(self.base_estimator, 'fit') and 'sample_weight' in self.base_estimator.fit.__code__.co_varnames:
            self.base_estimator.fit(X, y, sample_weight=self.weights_)
        else:
            self.base_estimator.fit(X, y)
        
        return self
    
    def predict(self, X):
        return self.base_estimator.predict(X)
    
    def predict_proba(self, X):
        if hasattr(self.base_estimator, 'predict_proba'):
            return self.base_estimator.predict_proba(X)
        else:
            # For SVM, use decision function
            scores = self.base_estimator.decision_function(X)
            # Convert to probabilities using sigmoid
            probs = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - probs, probs])
    
    def _calculate_fairness_weights(self, y, sensitive_attr):
        """Calculate instance weights for fairness"""
        if len(y) == 0:
            return np.array([])
        
        df = pd.DataFrame({'target': y, 'sensitive': sensitive_attr})
        group_counts = df.groupby(['sensitive', 'target']).size()
        
        weights = np.ones(len(y))
        for i, (target_val, sensitive_val) in enumerate(zip(y, sensitive_attr)):
            group_count = group_counts.get((sensitive_val, target_val), 1)
            weights[i] = 1.0 / max(1, group_count)  # Ensure non-zero denominator
        
        # Normalize weights and clip extreme values
        weights = np.clip(weights, 0.1, 10.0)
        weights = weights / np.mean(weights)
        return weights

class PostProcessingMitigator:
    """
    Post-processing bias mitigation techniques
    """
    
    def __init__(self):
        self.thresholds_ = {}
        self.calibration_params_ = {}
    
    def optimize_thresholds(self, y_true, y_prob, sensitive_attr, metric='equal_opportunity'):
        """
        Optimize classification thresholds for fairness
        """
        if len(y_true) == 0:
            raise ValueError("Cannot optimize thresholds on empty dataset")
        
        unique_groups = np.unique(sensitive_attr)
        self.thresholds_ = {}
        
        if metric == 'equal_opportunity':
            # Equalize True Positive Rates
            target_tpr = None
            
            # First pass: find optimal global TPR
            for group in unique_groups:
                mask = sensitive_attr == group
                if np.sum(mask) == 0:
                    continue
                    
                group_y_true = y_true[mask]
                group_y_prob = y_prob[mask]
                
                # Check if group has positive samples
                if np.sum(group_y_true == 1) == 0:
                    continue
                
                # Find threshold that maximizes TPR for this group
                thresholds = np.linspace(0.01, 0.99, 100)  # Avoid extreme values
                best_tpr = 0
                best_threshold = 0.5
                
                for threshold in thresholds:
                    y_pred = (group_y_prob >= threshold).astype(int)
                    tp = np.sum((group_y_true == 1) & (y_pred == 1))
                    fn = np.sum((group_y_true == 1) & (y_pred == 0))
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                    if tpr > best_tpr:
                        best_tpr = tpr
                        best_threshold = threshold
                
                if target_tpr is None:
                    target_tpr = best_tpr
                else:
                    target_tpr = min(target_tpr, best_tpr)
            
            # Handle case where no valid TPR found
            if target_tpr is None:
                target_tpr = 0.5
            
            # Second pass: find thresholds that achieve target TPR
            for group in unique_groups:
                mask = sensitive_attr == group
                if np.sum(mask) == 0:
                    self.thresholds_[group] = 0.5
                    continue
                    
                group_y_true = y_true[mask]
                group_y_prob = y_prob[mask]
                
                if np.sum(group_y_true == 1) == 0:
                    self.thresholds_[group] = 0.5
                    continue
                
                thresholds = np.linspace(0.01, 0.99, 100)
                best_threshold = 0.5
                best_diff = float('inf')
                
                for threshold in thresholds:
                    y_pred = (group_y_prob >= threshold).astype(int)
                    tp = np.sum((group_y_true == 1) & (y_pred == 1))
                    fn = np.sum((group_y_true == 1) & (y_pred == 0))
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                    diff = abs(tpr - target_tpr)
                    if diff < best_diff:
                        best_diff = diff
                        best_threshold = threshold
                
                self.thresholds_[group] = best_threshold
        
        elif metric == 'demographic_parity':
            # Equalize Positive Prediction Rates
            target_ppr = np.mean(y_prob >= 0.5)  # Global positive prediction rate
            
            for group in unique_groups:
                mask = sensitive_attr == group
                if np.sum(mask) == 0:
                    self.thresholds_[group] = 0.5
                    continue
                    
                group_y_prob = y_prob[mask]
                
                thresholds = np.linspace(0.01, 0.99, 100)
                best_threshold = 0.5
                best_diff = float('inf')
                
                for threshold in thresholds:
                    ppr = np.mean(group_y_prob >= threshold)
                    diff = abs(ppr - target_ppr)
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_threshold = threshold
                
                self.thresholds_[group] = best_threshold
        
        return self.thresholds_
    
    def predict_fair(self, y_prob, sensitive_attr):
        """
        Make fair predictions using optimized thresholds
        """
        if len(y_prob) == 0:
            return np.array([])
        
        y_pred = np.zeros(len(y_prob))
        
        for group in self.thresholds_:
            mask = sensitive_attr == group
            threshold = self.thresholds_[group]
            y_pred[mask] = (y_prob[mask] >= threshold).astype(int)
        
        # Handle groups not in thresholds (use default threshold)
        for group in np.unique(sensitive_attr):
            if group not in self.thresholds_:
                mask = sensitive_attr == group
                y_pred[mask] = (y_prob[mask] >= 0.5).astype(int)
        
        return y_pred
    
    def calibrate_predictions(self, y_true, y_prob, sensitive_attr, method='platt'):
        """
        Calibrate predictions to ensure fairness across groups
        """
        if len(y_true) == 0:
            return {}
        
        from sklearn.isotonic import IsotonicRegression
        
        unique_groups = np.unique(sensitive_attr)
        self.calibration_params_ = {}
        
        for group in unique_groups:
            mask = sensitive_attr == group
            if np.sum(mask) < 5:  # Need minimum samples for calibration
                continue
                
            group_y_true = y_true[mask]
            group_y_prob = y_prob[mask]
            
            # Check if we have both classes
            if len(np.unique(group_y_true)) < 2:
                continue
            
            try:
                if method == 'platt':
                    # Platt scaling (sigmoid)
                    from sklearn.linear_model import LogisticRegression
                    calibrator = LogisticRegression(random_state=42)
                    calibrator.fit(group_y_prob.reshape(-1, 1), group_y_true)
                    self.calibration_params_[group] = calibrator
                    
                elif method == 'isotonic':
                    # Isotonic regression
                    calibrator = IsotonicRegression(out_of_bounds='clip')
                    calibrator.fit(group_y_prob, group_y_true)
                    self.calibration_params_[group] = calibrator
            except Exception as e:
                print(f"Warning: Calibration failed for group {group}: {e}")
                continue
        
        return self.calibration_params_
    
    def apply_calibration(self, y_prob, sensitive_attr):
        """
        Apply calibration to predictions
        """
        if len(y_prob) == 0:
            return np.array([])
        
        calibrated_prob = np.copy(y_prob)
        
        for group in self.calibration_params_:
            mask = sensitive_attr == group
            if np.sum(mask) == 0:
                continue
                
            calibrator = self.calibration_params_[group]
            
            try:
                if hasattr(calibrator, 'predict_proba'):
                    # Platt scaling
                    calibrated_prob[mask] = calibrator.predict_proba(y_prob[mask].reshape(-1, 1))[:, 1]
                else:
                    # Isotonic regression
                    calibrated_prob[mask] = calibrator.predict(y_prob[mask])
            except Exception as e:
                print(f"Warning: Calibration application failed for group {group}: {e}")
                continue
        
        return calibrated_prob

def evaluate_mitigation_effectiveness(y_true, y_pred_original, y_pred_mitigated, 
                                    sensitive_attr, technique_name):
    """
    Compare bias metrics before and after mitigation
    """
    def calculate_bias_metrics(y_true, y_pred, sensitive_attr):
        if len(y_true) == 0:
            return {}
        
        metrics = {}
        
        # Demographic Parity
        groups = np.unique(sensitive_attr)
        pos_rates = {}
        for group in groups:
            mask = sensitive_attr == group
            if np.sum(mask) > 0:
                pos_rates[group] = np.mean(y_pred[mask])
        
        if len(pos_rates) > 1:
            metrics['demographic_parity_diff'] = max(pos_rates.values()) - min(pos_rates.values())
        
        # Equal Opportunity (for binary classification)
        if len(np.unique(y_true)) == 2:
            tpr_by_group = {}
            for group in groups:
                mask = sensitive_attr == group
                if np.sum(mask) == 0:
                    continue
                yt, yp = y_true[mask], y_pred[mask]
                tp = np.sum((yt == 1) & (yp == 1))
                fn = np.sum((yt == 1) & (yp == 0))
                tpr_by_group[group] = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            if len(tpr_by_group) > 1:
                metrics['equal_opportunity_diff'] = max(tpr_by_group.values()) - min(tpr_by_group.values())
        
        # Overall accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        return metrics
    
    original_metrics = calculate_bias_metrics(y_true, y_pred_original, sensitive_attr)
    mitigated_metrics = calculate_bias_metrics(y_true, y_pred_mitigated, sensitive_attr)
    
    comparison = {
        'technique': technique_name,
        'original_bias': original_metrics,
        'mitigated_bias': mitigated_metrics,
        'improvement': {}
    }
    
    # Calculate improvements
    for metric in ['demographic_parity_diff', 'equal_opportunity_diff']:
        if metric in original_metrics and metric in mitigated_metrics:
            original_val = original_metrics[metric]
            mitigated_val = mitigated_metrics[metric]
            improvement = ((original_val - mitigated_val) / original_val * 100) if original_val > 0 else 0
            comparison['improvement'][metric] = improvement
    
    # Calculate accuracy trade-off
    if 'accuracy' in original_metrics and 'accuracy' in mitigated_metrics:
        accuracy_change = mitigated_metrics['accuracy'] - original_metrics['accuracy']
        comparison['accuracy_tradeoff'] = accuracy_change
    
    return comparison

# Example usage and integration functions
def get_available_techniques():
    """Return available mitigation techniques"""
    mitigator = BiasMitigator()
    return mitigator.mitigation_techniques

def apply_mitigation_technique(technique, X_train, y_train, sensitive_train, 
                             X_test, y_test, sensitive_test, model, **kwargs):
    """
    Apply specified mitigation technique and return results
    """
    # Input validation
    if len(X_train) == 0:
        raise ValueError("Training data is empty")
    if len(X_test) == 0:
        raise ValueError("Test data is empty")
    
    mitigator = BiasMitigator()
    post_processor = PostProcessingMitigator()
    
    try:
        if technique == 'resampling':
            strategy = kwargs.get('strategy', 'undersample')
            X_train_new, y_train_new, sensitive_train_new = mitigator.demographic_parity_resampling(
                X_train, y_train, sensitive_train, strategy=strategy
            )
            
            # Additional validation
            if len(X_train_new) == 0:
                raise ValueError("Resampling resulted in empty training set")
            
            # Train model on resampled data
            model.fit(X_train_new, y_train_new)
            y_pred = model.predict(X_test)
            
            return y_pred, {'resampled_data_size': len(X_train_new)}
        
        elif technique == 'reweighting':
            # Calculate fairness weights
            weights = mitigator.calculate_fairness_weights(y_train, sensitive_train)
            
            # Train model with weights
            if hasattr(model, 'fit') and 'sample_weight' in model.fit.__code__.co_varnames:
                model.fit(X_train, y_train, sample_weight=weights)
            else:
                # For models that don't support sample weights, use FairClassifier
                fair_model = FairClassifier(base_estimator=model)
                fair_model.fit(X_train, y_train, sensitive_attr=sensitive_train)
                model = fair_model
            
            y_pred = model.predict(X_test)
            return y_pred, {'weights_applied': True}
        
        elif technique == 'fairness_constraints':
            # Use FairClassifier wrapper
            fair_model = FairClassifier(
                base_estimator=model,
                fairness_constraint='demographic_parity',
                lambda_fairness=kwargs.get('lambda_fairness', 1.0)
            )
            fair_model.fit(X_train, y_train, sensitive_attr=sensitive_train)
            y_pred = fair_model.predict(X_test)
            
            return y_pred, {'fairness_constraint_applied': True}
        
        elif technique == 'threshold_optimization':
            # Train original model first
            model.fit(X_train, y_train)
            
            # Get predictions
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                scores = model.decision_function(X_test)
                y_prob = 1 / (1 + np.exp(-scores))  # Convert to probabilities
            else:
                raise ValueError("Model must have predict_proba or decision_function method")
            
            # Optimize thresholds
            metric = kwargs.get('fairness_metric', 'equal_opportunity')
            thresholds = post_processor.optimize_thresholds(y_test, y_prob, sensitive_test, metric=metric)
            
            # Apply optimized thresholds
            y_pred = post_processor.predict_fair(y_prob, sensitive_test)
            
            return y_pred, {'optimized_thresholds': thresholds}
        
        elif technique == 'calibration':
            # Train original model first
            model.fit(X_train, y_train)
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                y_prob_train = model.predict_proba(X_train)[:, 1]
                y_prob_test = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                scores_train = model.decision_function(X_train)
                scores_test = model.decision_function(X_test)
                y_prob_train = 1 / (1 + np.exp(-scores_train))
                y_prob_test = 1 / (1 + np.exp(-scores_test))
            else:
                raise ValueError("Model must have predict_proba or decision_function method")
            
            # Calibrate predictions
            method = kwargs.get('calibration_method', 'platt')
            post_processor.calibrate_predictions(y_train, y_prob_train, sensitive_train, method=method)
            
            # Apply calibration
            y_prob_calibrated = post_processor.apply_calibration(y_prob_test, sensitive_test)
            y_pred = (y_prob_calibrated >= 0.5).astype(int)
            
            return y_pred, {'calibration_applied': True}
        
        else:
            raise ValueError(f"Unknown mitigation technique: {technique}")
    
    except Exception as e:
        print(f"Error in mitigation technique '{technique}': {str(e)}")
        # Return original model predictions as fallback
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred, {'error': str(e), 'fallback_used': True}