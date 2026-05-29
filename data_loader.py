import os
import pandas as pd
import numpy as np
import urllib.request
import streamlit as st
def check_network_connection(test_url="http://www.google.com", timeout=5):
    """
    Quick network connectivity check.
    Returns True if reachable, False otherwise.
    """
    try:
        urllib.request.urlopen(test_url, timeout=timeout)
        return True
    except Exception:
        return False


def generate_synthetic_data(num_samples=25000, seed=42):
    """
    Generates a highly realistic synthetic credit card fraud dataset.
    Mimics the Kaggle 'creditcardfraud' schema: Time, V1-V28, Amount, Class.
    Calibrates fraud distributions on key PCA components to simulate realistic model training
    and non-perfect precision/recall curves.
    """
    np.random.seed(seed)
    
    # 1. Generate Class (0 = Normal, 1 = Fraud)
    # Calibrated to ~0.35% fraud rate to ensure we get enough fraud instances (~88 in 25,000)
    # for stable model fitting and visualization, while preserving the extreme skew.
    fraud_prob = 0.0035
    class_labels = np.random.binomial(1, fraud_prob, size=num_samples)
    num_fraud = int(np.sum(class_labels))
    num_normal = num_samples - num_fraud
    
    # 2. Time (seconds elapsed from start, mimicking 48-hour window from 0 to 172792)
    time = np.random.randint(0, 172793, size=num_samples)
    
    # 3. V1 to V28 (PCA-reduced features)
    # Normal transactions: Mean 0, Std 1 for standard features.
    # Fraud transactions: Distinct means on predictive components (V3, V4, V10, V11, V12, V14, V17, V18).
    v_data = np.zeros((num_samples, 28))
    
    # Fill normal features
    v_data[class_labels == 0] = np.random.normal(0, 1.0, size=(num_normal, 28))
    
    # Fill fraud features with realistic mean shifts and variances
    fraud_means = {
        2: -3.5,   # V3 (indices are 0-indexed, so V3 is index 2)
        3: 2.8,    # V4
        9: -2.6,   # V10
        10: 2.2,   # V11
        11: -3.2,  # V12
        13: -4.2,  # V14
        16: -4.8,  # V17
        17: -2.5   # V18
    }
    
    # Populate normal features for fraud first, then inject specific shifts
    v_data[class_labels == 1] = np.random.normal(0, 1.2, size=(num_fraud, 28))
    for v_idx, mean_shift in fraud_means.items():
        v_data[class_labels == 1, v_idx] = np.random.normal(mean_shift, 1.5, size=num_fraud)
        
    # 4. Amount (Log-normally distributed transaction amounts)
    # Normal: Median ~$20, range up to a few thousands
    # Fraud: Median ~$70, distinct tail showing higher values on average
    amount = np.zeros(num_samples)
    amount[class_labels == 0] = np.random.lognormal(mean=2.9, sigma=0.9, size=num_normal)
    amount[class_labels == 1] = np.random.lognormal(mean=4.1, sigma=1.1, size=num_fraud)
    amount = np.round(amount, 2)
    
    # Create DataFrame
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    df = pd.DataFrame(np.column_stack([time, v_data, amount, class_labels]), columns=columns)
    
    # Ensure Class is integer
    df['Class'] = df['Class'].astype(int)
    df['Time'] = df['Time'].astype(int)
    
    return df

def download_dataset(url, dest_path):
    """
    Downloads dataset from HF mirror in chunks to show visual status/safety.
    """
    try:
        # User Agent override to bypass rigid firewalls
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
        urllib.request.install_opener(opener)
        
        # Start download
        with urllib.request.urlopen(url) as response, open(dest_path, 'wb') as out_file:
            meta = response.info()
            file_size = int(meta.get("Content-Length", 0))
            
            chunk_size = 1024 * 1024  # 1MB chunks
            downloaded = 0
            
            # Simple terminal/Streamlit logging
            while True:
                buffer = response.read(chunk_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def load_dataset(base_dir=None):
    """
    Checks for a local creditcard.csv in the directory.
    If missing, attempts to download it from Hugging Face.
    If the download fails or is offline, falls back to generating high-fidelity synthetic data.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        
    filename = 'creditcard.csv'
    local_path = os.path.join(base_dir, filename)
    
    # 1. Search locally
    if os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path)
            if 'Class' in df.columns and 'V1' in df.columns:
                return df, False, local_path  # False = Not synthetic
        except Exception as e:
            print(f"Local file load failed: {e}")
            
    # 2. Try to download from Hugging Face dataset mirror
    hf_url = "https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection/resolve/main/creditcard.csv"
    
    # Early network check before attempting download
    if check_network_connection():
        print(f"Dataset not found at {local_path}. Attempting to download from Hugging Face...")
        success = download_dataset(hf_url, local_path)
        if success:
            try:
                df = pd.read_csv(local_path)
                if 'Class' in df.columns and 'V1' in df.columns:
                    return df, False, local_path
            except Exception as e:
                print(f"Failed to read downloaded file: {e}")
    else:
        print("No network connection detected. Skipping download attempt.")
            
    # 3. Fallback to High-Fidelity Synthetic Generator
    print("Falling back to high-fidelity synthetic Credit Card Fraud simulator...")
    df = generate_synthetic_data(num_samples=25000, seed=42)
    
    synthetic_path = os.path.join(base_dir, 'creditcard_synthetic.csv')
    df.to_csv(synthetic_path, index=False)
    
    return df, True, synthetic_path
