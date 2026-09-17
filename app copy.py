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

        # Quantum-Inspired Feature Selection
        selected_features = self._quantum_feature_selection(X_train, y_train)
        print("Selected {} features using quantum-inspired method".format(len(selected_features)))

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
            'optimized_models': optimized_models,
            'anomaly_scores': anomaly_scores,
            'risk_ranks': risk_ranks
        }
    

    def _classical_feature_selection(self, X, y):
        """Quantum-inspired feature selection using amplitude amplification"""
        from sklearn.feature_selection import SelectKBest, f_classif

        # Used Classical approach: select features with highest importance
        selector = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]))
        X_selected = selector.fit_transform(X, y)

        # Get selected feature indices
        selected_features = selector.get_support(indices=True)
        return selected_features



    def _quantum_feature_selection(self, X, y):
        """Quantum-inspired feature selection using amplitude amplification"""
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        # Quantum-Inspired Genetic Algorithm (QIGA) for feature selection
        # Representation: a probability vector `p` of length n_features where
        # each entry is the chance of including that feature (a "Q-bit").
        # The algorithm repeatedly samples classical bitstrings from `p`,
        # evaluates them using a lightweight classifier, and updates `p`
        # towards the best-performing samples using a small rotation (delta).

        # Prepare arrays
        X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
        y_arr = y.values if hasattr(y, 'values') else np.asarray(y)

        n_features = X_arr.shape[1]
        pop_size = min(40, max(10, 6 * int(np.sqrt(n_features))))
        max_iter = 30

        # Initialize Q-bit probability vector (uniform superposition)
        p = np.full(n_features, 0.5)

        best_individual = None
        best_score = -np.inf

        for it in range(max_iter):
            # Sample population of classical individuals from Q-bits
            rand = np.random.rand(pop_size, n_features)
            population = (rand < p).astype(int)

            # Ensure each individual selects at least one feature
            for i in range(pop_size):
                if population[i].sum() == 0:
                    population[i, np.random.randint(0, n_features)] = 1

            scores = np.zeros(pop_size)

            # Evaluate fitness using cross-validated AUC with a cheap classifier
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

            # Find best in this generation
            gen_best_idx = int(np.nanargmax(scores))
            gen_best_score = float(scores[gen_best_idx])
            gen_best = population[gen_best_idx]

            if gen_best_score > best_score:
                best_score = gen_best_score
                best_individual = gen_best.copy()

            # Update Q-bits (probabilities) toward the generation best
            delta = 0.06 * (1.0 - (it / max_iter))
            p = p + delta * (gen_best - p)
            p = np.clip(p, 1e-3, 1 - 1e-3)

            # Small mutation to escape local optima
            mutation_rate = 0.01
            mutate = np.random.rand(n_features) < mutation_rate
            p[mutate] = 0.5

        # Final selection: features with probability > 0.5, fallback to best individual
        selected = np.where(p > 0.5)[0]
        if selected.size == 0 and best_individual is not None:
            selected = np.where(best_individual == 1)[0]

        return selected

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

    def predict(self, X, model_name='soft_voting'):
        """Make predictions using specified model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        if model_name not in self.models:
            available_models = list(self.models.keys())
            raise ValueError("Model {} not found. Available models: {}".format(model_name, available_models))

        model = self.models[model_name]

        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(X_scaled)[:, 1]
        else:
            predictions = model.predict(X_scaled)

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
            elif transaction_features.shape[1] > expected:
                transaction_features = transaction_features[:, :expected]

        try:
            scaled = self.scaler.transform(transaction_features)
        except Exception:
            scaled = transaction_features

        return transaction_features, scaled

    def _explain_fallback(self, original_features, model, model_name, feature_names, top_n=5):
        explanation = {
            'model': model_name,
            'reason': 'Fallback explanation generated from model feature importance.',
            'feature_impacts': []
        }

        if hasattr(model, 'feature_importances_'):
            importances = np.array(model.feature_importances_)
            top_idx = np.argsort(importances)[::-1][:top_n]

            for idx in top_idx:
                contribution = float(importances[idx])
                explanation['feature_impacts'].append({
                    'feature': feature_names[idx] if idx < len(feature_names) else f'feature_{idx}',
                    'value': float(original_features[0, idx]) if idx < original_features.shape[1] else None,
                    'contribution': contribution,
                    'direction': 'higher values increase risk' if contribution > 0 else 'lower values increase risk'
                })

            top_features = [item['feature'] for item in explanation['feature_impacts'][:3]]
            if top_features:
                explanation['reason'] = f"Important risk factors are {', '.join(top_features)}."
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
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(scaled_features)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            contributions = np.array(shap_values)[0]

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
        except Exception:
            return self._explain_fallback(original_features, model, model_name, feature_names, top_n)

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


def main():
    """Main function to demonstrate the fraud detection system"""
    print("Credit Card Fraud Detection System")
    print("=" * 50)

    # Load data
    df = load_credit_card_data()
    if df is None:
        print("Could not load data. Please provide creditcard.csv file.")
        return

    # Prepare data
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Initialize and train the system
    fraud_system = FraudDetectionSystem()
    X_test, y_test = fraud_system.fit(X, y)

    # Evaluate models
    print("\nModel Evaluation Results:")
    print("=" * 30)

    models_to_evaluate = ['logistic_regression', 'decision_tree', 'xgboost',
                         'random_forest', 'soft_voting', 'weighted_voting',
                         'stacking', 'adaptive_ensemble']

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
    sample_transaction = X_test[0]
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