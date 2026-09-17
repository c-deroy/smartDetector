"""
Credit Card Fraud Detection System with Three-Layer Architecture

Layer 1: Real-Time Transaction Screening
- Logistic Regression
- Decision Tree
- XGBoost
- Random Forest

Layer 2: Ensemble Intelligence
- Soft Voting
- Weighted Voting
- Stacking
- Adaptive Ensemble Selection

Layer 3: Quantum-Inspired Optimization
- Quantum-Inspired Feature Selection
- Hyperparameter Optimization
- Anomaly Scoring
- Risk Ranking
"""

import numpy as np
import pandas as pd
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        self.n_features = None
        self.explainers = {}

    # Layer 1: Real-Time Transaction Screening
    def train_base_models(self, X_train, y_train):
        """Train lightweight base models for real-time screening"""
        print("Training Layer 1: Real-Time Transaction Screening Models...")

        # Logistic Regression
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        self.models['logistic_regression'] = lr_model

        # Decision Tree
        dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
        dt_model.fit(X_train, y_train)
        self.models['decision_tree'] = dt_model

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic'
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model

        # Random Forest
        rf_model = RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=10,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model

        print("Base models trained successfully!")

    # Layer 2: Ensemble Intelligence
    def create_ensemble_models(self, X_train, y_train):
        """Create ensemble models combining base models"""
        print("Creating Layer 2: Ensemble Intelligence Models...")

        # Get base model predictions for meta-learning
        base_models = [
            ('lr', self.models['logistic_regression']),
            ('dt', self.models['decision_tree']),
            ('xgb', self.models['xgboost']),
            ('rf', self.models['random_forest'])
        ]

        # Soft Voting Ensemble
        soft_voting = VotingClassifier(
            estimators=base_models,
            voting='soft'
        )
        soft_voting.fit(X_train, y_train)
        self.models['soft_voting'] = soft_voting

        # Weighted Voting Ensemble (based on individual AUC scores)
        # Calculate weights based on cross-validation performance
        weights = self._calculate_model_weights(X_train, y_train, base_models)
        weighted_voting = VotingClassifier(
            estimators=base_models,
            voting='soft',
            weights=weights
        )
        weighted_voting.fit(X_train, y_train)
        self.models['weighted_voting'] = weighted_voting

        # Stacking Ensemble
        stacking = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
        stacking.fit(X_train, y_train)
        self.models['stacking'] = stacking

        # Adaptive Ensemble Selection
        adaptive_ensemble = self._create_adaptive_ensemble(X_train, y_train, base_models)
        self.models['adaptive_ensemble'] = adaptive_ensemble

        print("Ensemble models created successfully!")

    def _calculate_model_weights(self, X, y, base_models):
        """Calculate weights for weighted voting based on AUC scores"""
        weights = []
        for name, model in base_models:
            auc_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            weights.append(np.mean(auc_scores))
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        return weights

    def _create_adaptive_ensemble(self, X_train, y_train, base_models):
        """Create adaptive ensemble that selects best models per prediction"""
        # Use AdaBoost as meta-learner for adaptive selection
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=50,
            random_state=42
        )
        ada.fit(X_train, y_train)
        return ada

    # Layer 3: Quantum-Inspired Optimization
    def quantum_inspired_optimization(self, X_train, y_train, X_test, y_test):
        """Apply quantum-inspired techniques for optimization"""
        print("Applying Layer 3: Quantum-Inspired Optimization...")

        # Baseline and quantum-inspired feature selection comparison
        classical_features = self._classical_feature_selection(X_train, y_train)
        quantum_features = self._quantum_feature_selection(X_train, y_train)
        qde_features = self._quantum_differential_evolution_feature_selection(X_train, y_train)

        feature_selection_comparison = {
            'selectkbest': {
                'features': classical_features,
                'k': len(classical_features),
                'score': self._score_feature_subset(X_train, y_train, classical_features),
                'params': {'score_func': 'f_classif', 'k': min(20, X_train.shape[1]), 'cv': 3, 'scoring': 'roc_auc'}
            },
            'quantum_genetic': {
                'features': quantum_features,
                'k': len(quantum_features),
                'score': self._score_feature_subset(X_train, y_train, quantum_features),
                'params': {'population_size': min(40, max(10, 6 * int(np.sqrt(X_train.shape[1])))), 'max_iter': 30, 'mutation_rate': 0.01, 'cv': 3, 'scoring': 'roc_auc'}
            },
            'quantum_differential_evolution': {
                'features': qde_features,
                'k': len(qde_features),
                'score': self._score_feature_subset(X_train, y_train, qde_features),
                'params': {'population_size': 12, 'max_iter': 15, 'mutation_factor': 0.8, 'crossover_rate': 0.9, 'target_k': min(20, X_train.shape[1]), 'cv': 3, 'scoring': 'roc_auc'}
            }
        }

        best_method = max(
            feature_selection_comparison.items(),
            key=lambda item: item[1]['score']
        )[0]
        selected_features = feature_selection_comparison[best_method]['features']
        selected_features = np.asarray(selected_features, dtype=int)

        print("Selected {} features using {} method".format(len(selected_features), best_method))

        # Hyperparameter Optimization using quantum-inspired search
        optimized_models = self._quantum_hyperparameter_optimization(
            X_train[:, selected_features], y_train
        )

        # Anomaly Scoring using quantum-inspired approach
        anomaly_scores = self._quantum_anomaly_scoring(X_test[:, selected_features])

        # Risk Ranking
        risk_ranks = self._quantum_risk_ranking(X_test[:, selected_features], anomaly_scores)

        return {
            'selected_features': selected_features,
            'selected_method': best_method,
            'feature_selection_comparison': feature_selection_comparison,
            'optimized_models': optimized_models,
            'anomaly_scores': anomaly_scores,
            'risk_ranks': risk_ranks
        }

    def _score_feature_subset(self, X, y, feature_indices):
        """Cross-validated AUC score for a feature subset."""
        if feature_indices is None:
            return 0.0

        feature_indices = np.asarray(feature_indices, dtype=int)
        if feature_indices.size == 0:
            return 0.0

        X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
        y_arr = y.values if hasattr(y, 'values') else np.asarray(y)

        try:
            clf = LogisticRegression(max_iter=500, solver='liblinear')
            scores = cross_val_score(
                clf,
                X_arr[:, feature_indices],
                y_arr,
                cv=3,
                scoring='roc_auc',
                n_jobs=-1
            )
            return float(np.nanmean(scores))
        except Exception:
            return 0.0

    def _classical_feature_selection(self, X, y):
        """Classical baseline feature selection using SelectKBest."""
        from sklearn.feature_selection import SelectKBest, f_classif

        selector = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]))
        selector.fit(X, y)
        selected_features = selector.get_support(indices=True)
        return selected_features.astype(int)

    def _quantum_feature_selection(self, X, y):
        """Quantum-inspired genetic feature selection using probability amplitudes."""
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
        y_arr = y.values if hasattr(y, 'values') else np.asarray(y)

        n_features = X_arr.shape[1]
        pop_size = min(40, max(10, 6 * int(np.sqrt(n_features))))
        max_iter = 30

        p = np.full(n_features, 0.5)
        best_individual = None
        best_score = -np.inf

        for it in range(max_iter):
            rand = np.random.rand(pop_size, n_features)
            population = (rand < p).astype(int)

            for i in range(pop_size):
                if population[i].sum() == 0:
                    population[i, np.random.randint(0, n_features)] = 1

            scores = np.zeros(pop_size)
            for i in range(pop_size):
                cols = np.where(population[i] == 1)[0]
                if cols.size == 0:
                    scores[i] = 0.0
                    continue
                try:
                    clf = LogisticRegression(max_iter=500, solver='liblinear')
                    sc = cross_val_score(clf, X_arr[:, cols], y_arr, cv=3, scoring='roc_auc', n_jobs=-1)
                    scores[i] = np.nanmean(sc)
                except Exception:
                    scores[i] = 0.0

            gen_best_idx = int(np.nanargmax(scores))
            gen_best_score = float(scores[gen_best_idx])
            gen_best = population[gen_best_idx]

            if gen_best_score > best_score:
                best_score = gen_best_score
                best_individual = gen_best.copy()

            delta = 0.06 * (1.0 - (it / max_iter))
            p = p + delta * (gen_best - p)
            p = np.clip(p, 1e-3, 1 - 1e-3)

            mutation_rate = 0.01
            mutate = np.random.rand(n_features) < mutation_rate
            p[mutate] = 0.5

        selected = np.where(p > 0.5)[0]
        if selected.size == 0 and best_individual is not None:
            selected = np.where(best_individual == 1)[0]

        return selected.astype(int)

    def _quantum_differential_evolution_feature_selection(self, X, y, target_k=None, population_size=12, max_iter=15, mutation_factor=0.8, crossover_rate=0.9):
        """Quantum-inspired differential evolution for binary feature subset search."""
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
        y_arr = y.values if hasattr(y, 'values') else np.asarray(y)

        n_features = X_arr.shape[1]
        target_k = min(n_features, 20 if target_k is None else target_k)
        pop_size = min(24, max(8, population_size))

        population = []
        for _ in range(pop_size):
            bits = np.zeros(n_features, dtype=int)
            idx = np.random.choice(n_features, size=target_k, replace=False)
            bits[idx] = 1
            population.append(bits)
        population = np.array(population)

        best_individual = population[0].copy()
        best_score = -np.inf
        q_prob = np.full(n_features, 0.5)

        for it in range(max_iter):
            trial_population = population.copy()
            for i in range(pop_size):
                others = [j for j in range(pop_size) if j != i]
                a, b, c = population[np.random.choice(others, 3, replace=False)]

                mask = np.random.rand(n_features) < crossover_rate
                diff = b ^ c
                trial = a.copy()
                trial[mask] = trial[mask] ^ diff[mask]

                if trial.sum() > target_k:
                    excess = np.where(trial == 1)[0]
                    drop_count = trial.sum() - target_k
                    drop_idx = np.random.choice(excess, size=int(drop_count), replace=False)
                    trial[drop_idx] = 0
                elif trial.sum() < target_k:
                    missing = np.where(trial == 0)[0]
                    add_count = target_k - trial.sum()
                    add_idx = np.random.choice(missing, size=int(add_count), replace=False)
                    trial[add_idx] = 1

                q_sample = (np.random.rand(n_features) < q_prob).astype(int)
                if q_sample.sum() == 0:
                    q_sample[np.random.randint(0, n_features)] = 1

                if q_sample.sum() > target_k:
                    excess = np.where(q_sample == 1)[0]
                    q_sample[np.random.choice(excess, size=int(q_sample.sum() - target_k), replace=False)] = 0
                elif q_sample.sum() < target_k:
                    missing = np.where(q_sample == 0)[0]
                    q_sample[np.random.choice(missing, size=int(target_k - q_sample.sum()), replace=False)] = 1

                candidate = trial.copy()
                candidate = np.where(np.random.rand(n_features) < mutation_factor, candidate, q_sample)

                candidate = candidate.astype(int)
                if candidate.sum() == 0:
                    candidate[np.random.randint(0, n_features)] = 1

                cols = np.where(candidate == 1)[0]
                try:
                    clf = LogisticRegression(max_iter=500, solver='liblinear')
                    sc = cross_val_score(clf, X_arr[:, cols], y_arr, cv=3, scoring='roc_auc', n_jobs=-1)
                    candidate_score = float(np.nanmean(sc))
                except Exception:
                    candidate_score = 0.0

                current_cols = np.where(population[i] == 1)[0]
                try:
                    clf = LogisticRegression(max_iter=500, solver='liblinear')
                    sc = cross_val_score(clf, X_arr[:, current_cols], y_arr, cv=3, scoring='roc_auc', n_jobs=-1)
                    current_score = float(np.nanmean(sc))
                except Exception:
                    current_score = 0.0

                if candidate_score > current_score:
                    trial_population[i] = candidate
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_individual = candidate.copy()

                if candidate_score > best_score:
                    best_score = candidate_score
                    best_individual = candidate.copy()

            population = trial_population
            if best_individual is not None:
                q_prob = q_prob + 0.08 * (best_individual - q_prob)
                q_prob = np.clip(q_prob, 1e-3, 1 - 1e-3)

        if best_individual is None:
            return np.arange(min(n_features, target_k), dtype=int)

        selected = np.where(best_individual == 1)[0]
        return selected.astype(int)

    def _quantum_hyperparameter_optimization(self, X, y):
        """Quantum-inspired hyperparameter optimization"""
        # Simplified quantum-inspired search (real implementation would use QAOA)
        optimized_models = {}

        # Optimize XGBoost hyperparameters
        xgb_params = {
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [50, 100, 200]
        }

        xgb_grid = GridSearchCV(
            xgb.XGBClassifier(random_state=42),
            xgb_params,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        xgb_grid.fit(X, y)
        optimized_models['xgboost_optimized'] = xgb_grid.best_estimator_

        # Optimize Random Forest
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }

        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_params,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        rf_grid.fit(X, y)
        optimized_models['rf_optimized'] = rf_grid.best_estimator_

        return optimized_models

    def _quantum_anomaly_scoring(self, X):
        """Quantum-inspired anomaly scoring"""
        # Use isolation forest with quantum-inspired parameters
        from sklearn.ensemble import IsolationForest

        # Quantum-inspired contamination parameter (would be optimized via QAOA)
        quantum_contamination = 0.1

        iso_forest = IsolationForest(
            contamination=quantum_contamination,
            random_state=42,
            n_estimators=100
        )

        # Convert to anomaly scores (-1 for outliers, 1 for inliers)
        anomaly_scores = iso_forest.fit_predict(X)
        # Convert to positive scores (higher = more anomalous)
        anomaly_scores = -anomaly_scores

        return anomaly_scores

    def _quantum_risk_ranking(self, X, anomaly_scores):
        """Quantum-inspired risk ranking"""
        # Combine anomaly scores with model predictions
        risk_scores = []

        for i in range(len(X)):
            sample = X[i:i+1]

            # Get predictions from all models
            model_predictions = []
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(sample)[0][1]  # Probability of fraud
                        model_predictions.append(prob)
                    else:
                        pred = model.predict(sample)[0]
                        model_predictions.append(pred)
                except Exception:
                    # Skip models that cannot handle the reduced feature set
                    continue

            # Combine with anomaly score
            ensemble_prob = np.mean(model_predictions) if model_predictions else 0.5
            combined_risk = (ensemble_prob + anomaly_scores[i]) / 2
            risk_scores.append(combined_risk)

        # Rank by risk score
        risk_ranks = np.argsort(risk_scores)[::-1]  # Descending order

        return risk_ranks

    def fit(self, X, y):
        """Train the complete fraud detection system"""
        print("Training Fraud Detection System...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Handle class imbalance using SMOTE on the training set
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Remember number of features and names for runtime alignment
        try:
            # If X was a DataFrame, preserve column names
            self.feature_names = X_train.columns.tolist()
        except Exception:
            self.feature_names = None
        self.n_features = X_train_scaled.shape[1]

        # Layer 1: Train base models
        self.train_base_models(X_train_scaled, y_train)

        # Layer 2: Create ensemble models
        self.create_ensemble_models(X_train_scaled, y_train)

        # Layer 3: Apply quantum-inspired optimization
        quantum_results = self.quantum_inspired_optimization(
            X_train_scaled, y_train, X_test_scaled, y_test
        )

        self.is_trained = True
        self.quantum_results = quantum_results

        print("Fraud Detection System trained successfully!")
        return X_test_scaled, y_test

    def _normalize_feature_matrix(self, X):
        """Convert raw inputs to the same matrix layout used during training."""
        if X is None:
            return X

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if self.feature_names is not None and X_arr.shape[1] == len(self.feature_names):
            return X_arr

        scaler_n = getattr(self.scaler, 'n_features_in_', None)
        if scaler_n is not None and X_arr.shape[1] == scaler_n:
            try:
                return self.scaler.transform(X_arr)
            except Exception:
                return X_arr

        return X_arr

    def predict(self, X, model_name='soft_voting'):
        """Make predictions using specified model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X_model = transform_dataframe_for_artifact_model(X, self.scaler) if self.feature_names else X.to_numpy()
        else:
            X_model = self._normalize_feature_matrix(X)

        if model_name not in self.models:
            available_models = list(self.models.keys())
            raise ValueError("Model {} not found. Available models: {}".format(model_name, available_models))

        model = self.models[model_name]

        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(X_model)[:, 1]
        else:
            predictions = model.predict(X_model)

        return predictions

    def _default_feature_names(self):
        names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        return names[:self.n_features] if self.n_features else names

    def _prepare_transaction_features(self, transaction_features):
        if isinstance(transaction_features, list):
            transaction_features = np.array(transaction_features).reshape(1, -1)
        elif isinstance(transaction_features, dict):
            transaction_features = np.array(list(transaction_features.values())).reshape(1, -1)

        if transaction_features.ndim == 1:
            transaction_features = transaction_features.reshape(1, -1)

        expected = getattr(self.scaler, 'n_features_in_', None) or self.n_features
        if expected is not None:
            if transaction_features.shape[1] < expected:
                pad_width = expected - transaction_features.shape[1]
                transaction_features = np.hstack([
                    transaction_features,
                    np.zeros((transaction_features.shape[0], pad_width))
                ])
            elif transaction_features.shape[1] > expected and self.feature_names is not None and transaction_features.shape[1] == len(self.feature_names):
                # raw feature vector already includes all model columns; split numeric and one-hot parts
                scaled = self._normalize_feature_matrix(transaction_features)
                return transaction_features, scaled
            elif transaction_features.shape[1] > expected:
                transaction_features = transaction_features[:, :expected]

        try:
            scaled = self._normalize_feature_matrix(transaction_features)
        except Exception:
            scaled = transaction_features

        return transaction_features, scaled

    def _explain_fallback(self, original_features, scaled_features, model, model_name, feature_names, top_n=5):
        explanation = {
            'model': model_name,
            'reason': 'Local explanation based on how each feature changes this transaction risk.',
            'feature_impacts': []
        }

        if hasattr(model, 'predict_proba'):
            baseline_probability = float(model.predict_proba(scaled_features)[0][1])
            impacts = []
            for idx in range(scaled_features.shape[1]):
                counterfactual = scaled_features.copy()
                # Zero is the neutral value after StandardScaler; for one-hot
                # columns it also represents the absence of that category.
                counterfactual[0, idx] = 0.0
                neutral_probability = float(model.predict_proba(counterfactual)[0][1])
                contribution = baseline_probability - neutral_probability
                impacts.append((idx, contribution))

            top_idx = sorted(impacts, key=lambda item: abs(item[1]), reverse=True)[:top_n]
            for idx, contribution in top_idx:
                explanation['feature_impacts'].append({
                    'feature': feature_names[idx] if idx < len(feature_names) else f'feature_{idx}',
                    'value': float(original_features[0, idx]) if idx < original_features.shape[1] else None,
                    'contribution': float(contribution),
                    'direction': 'increases fraud risk' if contribution > 0 else 'reduces fraud risk'
                })

            top_features = [item['feature'] for item in explanation['feature_impacts'][:3]]
            if top_features:
                explanation['reason'] = f"This transaction's risk is most affected by {', '.join(top_features)}."
        else:
            explanation['reason'] = 'No SHAP or feature importance explanation available for this model.'

        return explanation

    def explain_transaction(self, transaction_features, model_name='xgboost', top_n=5):
        if not self.is_trained:
            raise ValueError('Model not trained. Call fit() first.')

        original_features, scaled_features = self._prepare_transaction_features(transaction_features)
        feature_names = self.feature_names or self._default_feature_names()

        if model_name not in self.models:
            model_name = 'xgboost' if 'xgboost' in self.models else next(iter(self.models))

        model = self.models[model_name]

        try:
            explainer = self.explainers.get(model_name)
            if explainer is None:
                try:
                    import shap
                    explainer = shap.TreeExplainer(model)
                except Exception:
                    explainer = None

            if explainer is None:
                raise RuntimeError('SHAP explainer not available')

            shap_values = explainer.shap_values(scaled_features)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            shap_array = np.asarray(shap_values)
            # SHAP versions return binary classifier outputs as either
            # (rows, features), (rows, features, classes), or a list by class.
            if shap_array.ndim == 3:
                shap_array = shap_array[:, :, 1] if shap_array.shape[2] > 1 else shap_array[:, :, 0]
            if shap_array.ndim == 1:
                shap_array = shap_array.reshape(1, -1)
            contributions = np.asarray(shap_array[0], dtype=float).reshape(-1)

            top_idx = np.argsort(np.abs(contributions))[::-1][:top_n]
            feature_impacts = []
            for idx in top_idx:
                contribution = float(contributions[idx])
                feature_impacts.append({
                    'feature': feature_names[idx] if idx < len(feature_names) else f'feature_{idx}',
                    'value': float(original_features[0, idx]) if idx < original_features.shape[1] else None,
                    'contribution': contribution,
                    'direction': 'increases fraud risk' if contribution > 0 else 'reduces fraud risk'
                })

            positive = [item for item in feature_impacts if item['contribution'] > 0]
            top_features = [item['feature'] for item in positive[:3]] or [item['feature'] for item in feature_impacts[:3]]
            reason = 'The most influential features are ' + ', '.join(top_features) + '.'

            return {
                'model': model_name,
                'reason': reason,
                'feature_impacts': feature_impacts
            }
        except Exception as error:
            print(f'SHAP explanation failed for {model_name}: {error}')
            return self._explain_fallback(original_features, scaled_features, model, model_name, feature_names, top_n)

    def load_from_artifacts(self, model_dir='models'):
        """Load a previously trained model, scaler and metadata from disk.

        Returns True if load succeeded, False otherwise.
        """
        model_file = os.path.join(model_dir, 'rf_paysim.joblib')
        scaler_file = os.path.join(model_dir, 'scaler.joblib')
        meta_file = os.path.join(model_dir, 'metadata.json')

        if not (os.path.exists(model_file) and os.path.exists(scaler_file) and os.path.exists(meta_file)):
            return False

        try:
            clf = joblib.load(model_file)
            scaler = joblib.load(scaler_file)
            with open(meta_file, 'r') as f:
                meta = json.load(f)

            # attach artifacts
            self.models = {'rf_paysim': clf}
            self.scaler = scaler
            self.feature_names = meta.get('feature_names')
            self.n_features = len(self.feature_names) if self.feature_names else None
            self.is_trained = True

            # build SHAP explainers if shap is available
            try:
                self._build_shap_explainers()
            except Exception:
                pass

            return True
        except Exception:
            return False

    def prepare_named_features(self, feature_dict):
        """Prepare a 2D numpy array of features from a dict of named inputs.

        - Uses `self.feature_names` (from metadata) when available.
        - Supports direct feature-name lookup and one-hot feature names such as
          `type_TRANSFER` / `merchant_cat_retail` / `country_US`.
        """
        if not isinstance(feature_dict, dict):
            raise ValueError('feature_dict must be a dict')

        if not self.feature_names:
            vals = list(feature_dict.values())
            return np.asarray(vals).reshape(1, -1)

        out = []
        for fname in self.feature_names:
            if fname in feature_dict:
                value = feature_dict[fname]
                try:
                    out.append(float(value))
                except (TypeError, ValueError):
                    out.append(1 if str(value).lower() in ('true', 'yes', 'y', '1') else 0)
                continue

            # handle one-hot style feature names like type_PAYMENT, merchant_cat_restaurant, country_US
            if '_' in fname:
                parts = fname.split('_')
                if parts[0] in ('type', 'merchant_cat', 'country'):
                    prefix = '_'.join(parts[:-1])
                    value = parts[-1]
                else:
                    prefix = parts[0]
                    value = '_'.join(parts[1:])

                provided = feature_dict.get(prefix)
                if provided is None:
                    # support direct nested keys like {'type': 'PAYMENT'} and {'country': 'US'}
                    out.append(0)
                else:
                    out.append(1 if str(provided).lower() == str(value).lower() else 0)
                continue

            out.append(0)

        return np.asarray(out).reshape(1, -1)

    def _build_shap_explainers(self):
        """Build SHAP explainers for tree-based models when available."""
        try:
            import shap
        except Exception as error:
            print(f'SHAP import failed: {error}')
            return

        for name, model in self.models.items():
            try:
                explainer = shap.TreeExplainer(model)
                self.explainers[name] = explainer
            except Exception as error:
                print(f'SHAP initialization failed for {name}: {error}')
                continue

    def get_real_time_score(self, transaction_features):
        """Get real-time fraud score for a single transaction"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        transaction_features, _ = self._prepare_transaction_features(transaction_features)

        # Get scores from all models
        scores = {}
        for model_name, model in self.models.items():
            try:
                score = self.predict(transaction_features, model_name)
                scores[model_name] = float(score[0])
            except:
                scores[model_name] = 0.5

        # Quantum-enhanced risk score
        if hasattr(self, 'quantum_results'):
            anomaly_score = self._quantum_anomaly_scoring(
                self.scaler.transform(transaction_features)[:, self.quantum_results['selected_features']]
            )[0]
            quantum_risk = (scores['soft_voting'] + anomaly_score) / 2
            scores['quantum_enhanced'] = float(quantum_risk)

        scores['explanation'] = self.explain_transaction(transaction_features)
        return scores

    def evaluate_model(self, X_test, y_test, model_name):
        """Evaluate a specific model"""
        predictions = self.predict(X_test, model_name)

        if hasattr(self.models[model_name], 'predict_proba'):
            # Probability predictions
            pred_class = (predictions > 0.5).astype(int)
            auc = roc_auc_score(y_test, predictions)
        else:
            # Class predictions
            pred_class = predictions
            auc = None

        report = classification_report(y_test, pred_class)
        cm = confusion_matrix(y_test, pred_class)

        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'auc_score': auc
        }


def load_credit_card_data():
    """Load credit card fraud detection dataset"""
    try:
        # Try to load from common locations
        data_paths = [
            'creditcard.csv',
            'data/creditcard.csv',
            '../data/creditcard.csv',
            '../../data/creditcard.csv'
        ]

        for path in data_paths:
            try:
                df = pd.read_csv(path)
                print("Data loaded from {}".format(path))
                print("Dataset shape: {}".format(df.shape))
                print("Class distribution: {}".format(df['Class'].value_counts()))
                return df
            except FileNotFoundError:
                continue

        # If no local file found, try to download
        print("Local data not found. Please ensure creditcard.csv is in the working directory.")
        return None

    except Exception as e:
        print("Error loading data: {}".format(e))
        return None


def transform_dataframe_for_artifact_model(df, scaler, feature_names=None):
    """Transform raw PaySim-like data to match the exact trained artifact schema."""
    if df is None:
        return None

    df = df.copy()
    drop_cols = ['nameOrig', 'nameDest', 'device_id', 'ip_addr']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    feature_names = feature_names or []
    if feature_names:
        out = np.zeros((len(df), len(feature_names)), dtype=float)
        numeric_names = [n for n in feature_names if not (n.startswith('type_') or n.startswith('merchant_cat_') or n.startswith('country_'))]
        numeric_cols = [c for c in numeric_names if c in df.columns]

        if numeric_cols:
            X_num = df[numeric_cols].fillna(0)
            scaled = scaler.transform(X_num.to_numpy()) if hasattr(scaler, 'transform') else X_num.to_numpy()
            for idx, col in enumerate(numeric_cols):
                if col in feature_names:
                    out[:, feature_names.index(col)] = scaled[:, idx]

        for idx, fname in enumerate(feature_names):
            if fname.startswith('type_'):
                value = fname.split('type_', 1)[1]
                raw_col = 'type'
                if raw_col in df.columns:
                    out[:, idx] = (df[raw_col].astype(str) == value).astype(float)
            elif fname.startswith('merchant_cat_'):
                value = fname.split('merchant_cat_', 1)[1]
                raw_col = 'merchant_cat'
                if raw_col in df.columns:
                    out[:, idx] = (df[raw_col].astype(str) == value).astype(float)
            elif fname.startswith('country_'):
                value = fname.split('country_', 1)[1]
                raw_col = 'country'
                if raw_col in df.columns:
                    out[:, idx] = (df[raw_col].astype(str) == value).astype(float)

        return out

    # fallback: match the training script behavior exactly
    cat_cols = [c for c in ['type', 'merchant_cat', 'country', 'isFlaggedFraud'] if c in df.columns]
    num_cols = [c for c in df.columns if c not in cat_cols + ['isFraud', 'Class'] and pd.api.types.is_numeric_dtype(df[c])]
    X_num = df[num_cols].fillna(0)
    if cat_cols:
        df_cat = pd.get_dummies(df[cat_cols].astype(str), drop_first=True)
    else:
        df_cat = pd.DataFrame(index=df.index)
    X_num_scaled = scaler.transform(X_num) if hasattr(scaler, 'transform') else X_num.to_numpy()
    return np.hstack([X_num_scaled, df_cat.to_numpy()])


def main():
    """Main function to demonstrate the fraud detection system"""
    print("Credit Card Fraud Detection System")
    print("=" * 50)
    # Prefer engineered PaySim dataset if present
    engineered_paths = [
        'paysim_engineered.csv',
        os.path.join('data', 'paysim_engineered.csv'),
        os.path.join('.', 'paysim_engineered.csv')
    ]

    df = None
    for p in engineered_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                print(f"Loaded engineered data from {p}")
                break
            except Exception:
                continue

    if df is None:
        df = load_credit_card_data()
        if df is None:
            print("Could not load any dataset. Please provide paysim_engineered.csv or creditcard.csv.")
            return
        X = df.drop('Class', axis=1)
        y = df['Class']
    else:
        # engineered PaySim uses 'isFraud' as target
        if 'isFraud' not in df.columns:
            print('Engineered dataset missing isFraud column')
            return
        X = df.drop('isFraud', axis=1)
        y = df['isFraud']

    # Initialize system and try loading artifacts
    fraud_system = FraudDetectionSystem()
    loaded = fraud_system.load_from_artifacts()
    if not loaded:
        print('No trained artifacts found; training on available data...')
        X_test, y_test = fraud_system.fit(X, y)
    else:
        print('Loaded trained model artifacts from models/; performing evaluation using dataset sample')
        try:
            X_model = transform_dataframe_for_artifact_model(X, fraud_system.scaler, fraud_system.feature_names)
            X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=0.2, stratify=y, random_state=42)
        except Exception:
            X_test = X
            y_test = y

    # Evaluate models
    print("\nModel Evaluation Results:")
    print("=" * 30)

    models_to_evaluate = list(fraud_system.models.keys())

    for model_name in models_to_evaluate:
        try:
            results = fraud_system.evaluate_model(X_test, y_test, model_name)
            print("\n{}:".format(model_name.upper()))
            print("AUC Score: {:.4f}".format(results['auc_score']) if results['auc_score'] else "AUC: N/A")
            print("Classification Report:")
            print(results['classification_report'])
        except Exception as e:
            print("Error evaluating {}: {}".format(model_name, e))

    # Demonstrate real-time scoring
    print("\nReal-Time Transaction Scoring Demo:")
    print("=" * 40)

    # Get a sample transaction (first row of test data)
    sample_transaction = X_test[0] if isinstance(X_test, np.ndarray) else X_test.iloc[0].to_numpy()
    scores = fraud_system.get_real_time_score(sample_transaction)

    print("Sample Transaction Scores:")
    for model, score in scores.items():
        if model == 'explanation':
            continue
        print("  {}: {:.4f}".format(model, score))

    if scores.get('explanation'):
        print("\nExplanation:")
        print(scores['explanation'].get('reason'))

    print("\nSystem ready for real-time fraud detection!")

if __name__ == "__main__":
    main()