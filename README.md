# 🛡️ Guardian: AI-Powered Credit Card Fraud Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/AnveshAnnepaga/AI-Powered-Credit-Card-Fraud-Detection-System/main/app.py)

An interactive, premium executive dashboard engineered to teach students, data scientists, and risk underwriters the real-world operational challenges of **Banking AI**. Built with Streamlit and Scikit-Learn, this system models, evaluates, and simulates automated credit card transaction approvals and anomalies.

---

## 🎯 Educational Takeaways

### 1. The Challenge of Extreme Class Imbalance
In real-world credit card transaction logs, fraud is extremely rare (typically **0.17%** of all volume). 
- **The Accuracy Trap**: If a machine learning model simply predicts "Safe/Normal" for every single transaction, it achieves a deceptively high **99.83% accuracy**. However, this model is **100% useless** because it stops $0 of fraud!
- **Imbalance Mitigation**: This system demonstrates how classifiers adjust to skewness using **class weighting (`class_weight='balanced'`)** to penalize missed fraud heavily during optimization.

### 2. Why Precision-Recall Curves Beat ROC-AUC
- **ROC Curves** plot True Positive Rate vs. False Positive Rate. Because the number of True Negatives (actual normal transactions) is massive, the False Positive Rate stays tiny, resulting in a deceptively optimistic Area Under the Curve (ROC-AUC > 0.99) even for mediocre models.
- **Precision-Recall Curves** plot Precision (how many of our fraud alarms were correct) vs. Recall (how much of the total fraud we caught). They ignore True Negatives entirely, exposing how well the model actually isolates fraud amongst false alarms. This is the gold standard for imbalanced data.

### 3. Banking Economics: Cost-Sensitive Decision Tuning
In corporate finance, machine learning thresholds are never tuned using mathematical accuracy alone. Decisions are driven by **financial profit and loss**:
- **Missed Fraud (False Negative) Cost**: Paying for the transaction refund, handling chargeback fees, and facing regulatory compliance fines (e.g., **$1,200** per case).
- **False Alarm (False Positive) Cost**: Blocking an honest customer's card, generating card replacement costs, call center volume, and triggering customer churn (e.g., **$45** per case).
- **The Optimizer**: Changing the decision threshold simulates how a bank finds the exact threshold that minimizes the **Total Net Financial Loss**!

---

## ⚙️ Implemented Algorithms

The pipeline leverages multiple analytical paradigms to demonstrate supervised vs. unsupervised models:

### 1. Logistic Regression (Supervised Baseline)
- **Concept**: Fits a linear decision boundary using log-odds.
- **Role**: Supervised classification with class weights. Offers highly interpretable **log-odds coefficients** showing exactly which transaction features drive the risk score.

### 2. Random Forest (Supervised Ensemble)
- **Concept**: An ensemble of decision trees fitting non-linear interactions.
- **Role**: Supervised ensemble modeling, capturing subtle correlation patterns across anonymized latent components with high precision.

### 3. Isolation Forest (Unsupervised Anomaly Detection)
- **Concept**: Isolates anomalies in high-dimensional feature space by randomly partitioning data splits.
- **Role**: Unsupervised anomaly detection. Unlike standard classifiers, it is trained **without** using transaction labels, isolating novel or zero-day fraud patterns purely because they look statistically "out-of-place".

### 4. Principal Component Analysis (PCA)
- **Concept**: Dimensionality reduction projecting high-dimensional raw attributes to 28 principal components.
- **Role**: Features `V1` to `V28` in the dataset are the result of PCA. We render an interactive **3D PCA Space Map** allowing users to select components and visually isolate fraudulent clusters in a three-dimensional field.

---

## 📂 System Architecture & Files

```
├── data_loader.py    # Auto-scans, downloads real Kaggle CSV from Hugging Face,
│                     # or falls back to a high-fidelity synthetic fraud simulator.
├── models.py        # ML Pipeline: scaling, training models, threshold evaluation,
│                     # and live coefficient contribution extraction.
├── app.py           # Gorgeous dark-themed Streamlit dashboard & user interface.
├── requirements.txt # Project library dependencies.
└── README.md        # Technical and educational documentation (This file).
```

---

## 🚀 Installation & Quick Start

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your local environment.

### 2. Clone and Setup Environment
Navigate into the repository directory and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Running the Dashboard
Launch the Streamlit portal:
```bash
streamlit run app.py
```

### 4. Data Connection Telemetry
Upon startup, the pipeline executes a multi-stage data check:
- **Local Find**: Scans for a local `creditcard.csv`.
- **Cloud Download**: If missing, downloads the real dataset mirror from Hugging Face (~150MB).
- **Graceful Fallback**: If offline, generates a highly realistic synthetic dataset containing **25,000 transactions** (with a calibrated ~0.35% class balance) so the portal runs instantly and seamlessly!