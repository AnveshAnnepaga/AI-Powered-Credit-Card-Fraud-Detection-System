import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, average_precision_score, auc

class FraudMLPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.lr_model = None
        self.rf_model = None
        self.if_model = None
        
        # Performance metrics cache
        self.lr_metrics = {}
        self.rf_metrics = {}
        self.if_metrics = {}
        
        # Test split for plotting curves
        self.X_test = None
        self.y_test = None
        
    def fit(self, df):
        """
        Preprocesses and trains Logistic Regression, Random Forest, and Isolation Forest.
        Optimized to handle extreme class imbalance efficiently.
        """
        # 1. Separate features and target
        X = df.drop(columns=['Class', 'Time']) # Exclude Time from modeling, standard practice
        y = df['Class']
        
        # 2. Train-Test Split (stratified to maintain fraud proportion in test set)
        X_train, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=self.random_state
        )
        
        # Save raw test targets
        self.y_test = y_test
        
        # 3. Fit scaler on training amount (and fit transform on all features for modeling)
        # Note: V1-V28 are already PCA features, but scaling the entire matrix is safe and standard
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test_raw)
        
        # Save scaled test features
        self.X_test = X_test_scaled
        
        # 4. Train Logistic Regression (Balanced class weight to adjust for skew)
        self.lr_model = LogisticRegression(
            max_iter=1000, 
            class_weight='balanced', 
            random_state=self.random_state,
            C=0.1
        )
        self.lr_model.fit(X_train_scaled, y_train)
        
        # 5. Train Random Forest (Downsampled depth & class weights for fast, highly accurate fit)
        # Using 50 trees and max_depth=10 to keep training under 2 seconds on large datasets
        self.rf_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            class_weight='balanced_subsample',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.rf_model.fit(X_train_scaled, y_train)
        
        # 6. Train Isolation Forest (Unsupervised - trained on all training features)
        # Contamination is set to 0.01 (1%), which is standard for anomaly detection baselines
        self.if_model = IsolationForest(
            contamination=0.01,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.if_model.fit(X_train_scaled)
        
        # 7. Compute curves and caching
        self._compute_model_metrics(X_test_scaled, y_test)
        
        return X_train, X_test_raw
        
    def _compute_model_metrics(self, X_test, y_test):
        """Precomputes ROC and PR curves for the dashboard visualizations."""
        # --- Logistic Regression ---
        lr_probs = self.lr_model.predict_proba(X_test)[:, 1]
        lr_precision, lr_recall, lr_pr_thresholds = precision_recall_curve(y_test, lr_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
        
        self.lr_metrics = {
            'probs': lr_probs,
            'pr_curve': (lr_precision, lr_recall),
            'roc_curve': (lr_fpr, lr_tpr),
            'pr_auc': average_precision_score(y_test, lr_probs),
            'roc_auc': auc(lr_fpr, lr_tpr)
        }
        
        # --- Random Forest ---
        rf_probs = self.rf_model.predict_proba(X_test)[:, 1]
        rf_precision, rf_recall, rf_pr_thresholds = precision_recall_curve(y_test, rf_probs)
        rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
        
        self.rf_metrics = {
            'probs': rf_probs,
            'pr_curve': (rf_precision, rf_recall),
            'roc_curve': (rf_fpr, rf_tpr),
            'pr_auc': average_precision_score(y_test, rf_probs),
            'roc_auc': auc(rf_fpr, rf_tpr)
        }
        
        # --- Isolation Forest ---
        # Isolation forest decision function outputs negative for anomalies.
        # We transform this output so that more negative scores yield higher anomaly probability metrics.
        if_scores = self.if_model.decision_function(X_test)
        
        # Normalize anomaly scores to a 0 to 1 range
        min_score = np.min(if_scores)
        max_score = np.max(if_scores)
        # High score means normal, low score means anomaly. Let's invert it:
        if_anomaly_probs = 1.0 - (if_scores - min_score) / (max_score - min_score + 1e-9)
        
        if_precision, if_recall, _ = precision_recall_curve(y_test, if_anomaly_probs)
        if_fpr, if_tpr, _ = roc_curve(y_test, if_anomaly_probs)
        
        self.if_metrics = {
            'scores': if_scores,
            'probs': if_anomaly_probs,
            'pr_curve': (if_precision, if_recall),
            'roc_curve': (if_fpr, if_tpr),
            'pr_auc': average_precision_score(y_test, if_anomaly_probs),
            'roc_auc': auc(if_fpr, if_tpr)
        }
        
    def evaluate_threshold(self, model_name, threshold):
        """
        Simulates model performance dynamically at a specific decision threshold.
        Returns metrics (F1, Precision, Recall) and the confusion matrix.
        """
        y_test = self.y_test
        
        if model_name == "Logistic Regression":
            probs = self.lr_metrics['probs']
        elif model_name == "Random Forest":
            probs = self.rf_metrics['probs']
        else: # Isolation Forest
            probs = self.if_metrics['probs']
            
        preds = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
        
    def predict_transaction(self, raw_features):
        """
        Takes a raw transaction (dict of V1-V28 and Amount),
        and runs live inference through all three models.
        Extracts feature contribution maps (risk drivers) from Logistic Regression coefficients.
        """
        # Ensure correct column order matching scaler (excluding Class and Time)
        columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
        df_single = pd.DataFrame([raw_features])[columns]
        
        # Scale input
        scaled_single = self.scaler.transform(df_single)
        
        # 1. Logistic Regression Prediction
        lr_prob = self.lr_model.predict_proba(scaled_single)[0, 1]
        
        # Calculate feature contributions for Logistic Regression
        # Contribution = Scaled value * Coefficient
        coefs = self.lr_model.coef_[0]
        contributions = scaled_single[0] * coefs
        
        # Sort contributions to find top risk and protective factors
        feature_impacts = list(zip(columns, contributions))
        feature_impacts = sorted(feature_impacts, key=lambda x: x[1], reverse=True)
        
        # Top 5 pushing towards fraud (positive log-odds)
        top_risk_factors = [item for item in feature_impacts if item[1] > 0][:5]
        # Top 5 pushing towards normal (negative log-odds)
        top_protective_factors = sorted([item for item in feature_impacts if item[1] < 0], key=lambda x: x[1])[:5]
        
        # 2. Random Forest Prediction
        rf_prob = self.rf_model.predict_proba(scaled_single)[0, 1]
        
        # 3. Isolation Forest Prediction
        if_score = self.if_model.decision_function(scaled_single)[0]
        
        # Map score to the same range as training cache
        train_scores = self.if_metrics['scores']
        min_train = np.min(train_scores)
        max_train = np.max(train_scores)
        if_anomaly_prob = 1.0 - (if_score - min_train) / (max_train - min_train + 1e-9)
        if_anomaly_prob = np.clip(if_anomaly_prob, 0.0, 1.0)
        
        return {
            'lr_prob': lr_prob,
            'rf_prob': rf_prob,
            'if_score': if_score,
            'if_prob': if_anomaly_prob,
            'top_risk_factors': top_risk_factors,
            'top_protective_factors': top_protective_factors
        }
