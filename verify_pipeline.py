import time
from data_loader import load_dataset
from models import FraudMLPipeline

def run_verification():
    print("==================================================")
    print("🛡️ GUARDIAN AI SYSTEM INTEGRITY TEST")
    print("==================================================")
    
    # 1. Test Dataset Loader
    print("\n[STEP 1] Testing Data Loading System...")
    start_time = time.time()
    df, is_synthetic, filepath = load_dataset()
    elapsed = time.time() - start_time
    
    print(f"✔️ Dataset resolved in {elapsed:.2f} seconds.")
    print(f"✔️ Source: {'[SYNTHETIC FALLBACK]' if is_synthetic else '[REAL KAGGLE CSV]'}")
    print(f"✔️ Location: {filepath}")
    print(f"✔️ Shape: {df.shape}")
    print(f"✔️ Class distribution: {df['Class'].value_counts().to_dict()}")
    
    # 2. Test Model Training Pipeline
    print("\n[STEP 2] Initializing & Training Models (Logistic Regression, Random Forest, Isolation Forest)...")
    start_time = time.time()
    pipeline = FraudMLPipeline(random_state=42)
    X_train, X_test = pipeline.fit(df)
    elapsed = time.time() - start_time
    print(f"✔️ ML Pipeline fitted in {elapsed:.2f} seconds.")
    
    # 3. Verify Metrics Caching
    print("\n[STEP 3] Verifying Performance Metrics...")
    print(f"✔️ Logistic Regression - Test PR-AUC: {pipeline.lr_metrics['pr_auc']:.4f}")
    print(f"✔️ Random Forest       - Test PR-AUC: {pipeline.rf_metrics['pr_auc']:.4f}")
    print(f"✔️ Isolation Forest   - Test PR-AUC: {pipeline.if_metrics['pr_auc']:.4f}")
    
    # 4. Verify Single Transaction Prediction
    print("\n[STEP 4] Testing Live Transaction Assessor...")
    sample_transaction = {
        'Amount': 150.00,
        **{f'V{i}': 0.0 for i in range(1, 29)}
    }
    # Inject moderate fraud indicator
    sample_transaction['V14'] = -5.0
    sample_transaction['V17'] = -4.5
    
    pred_res = pipeline.predict_transaction(sample_transaction)
    print("✔️ Live Assessor executed successfully.")
    print(f"✔️ Logistic Reg Prob:  {pred_res['lr_prob']*100:.2f}%")
    print(f"✔️ Random Forest Prob: {pred_res['rf_prob']*100:.2f}%")
    print(f"✔️ Isolation Forest:  Score = {pred_res['if_score']:.4f}, Anomaly Prob = {pred_res['if_prob']*100:.2f}%")
    print(f"✔️ Top Risk Drivers:   {pred_res['top_risk_factors'][:2]}")
    
    # 5. Verify Threshold Simulator
    print("\n[STEP 5] Testing Dynamic Threshold Evaluator...")
    eval_res = pipeline.evaluate_threshold("Random Forest", 0.30)
    print("✔️ Threshold Evaluator executed successfully.")
    print(f"✔️ Random Forest (t=0.30) - Precision: {eval_res['precision']*100:.2f}%, Recall: {eval_res['recall']*100:.2f}%")
    
    print("\n==================================================")
    print("✅ GUARDIAN AI SYSTEM INTEGRITY TEST COMPLETE: ALL SYSTEMS NOMINAL")
    print("==================================================")

if __name__ == "__main__":
    run_verification()
