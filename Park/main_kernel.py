#!/usr/bin/env python3

import numpy as np
import pandas as pd
from datetime import datetime
import os

SEED = 42
np.random.seed(SEED)

from SpADG import SpADG_Kernel_Regression


def load_parkinsons_data(data_path):
    df = pd.read_csv(data_path)
    features = df.drop(['subject#', 'motor_UPDRS', 'total_UPDRS'], axis=1).values
    motor_UPDRS = df['motor_UPDRS'].values
    total_UPDRS = df['total_UPDRS'].values
    return features, motor_UPDRS, total_UPDRS, df['subject#'].values


def preprocess_parkinsons_data(X_all, y_all, subjects, n_samples):
    unique_subjects = np.unique(subjects)
    domains = []
    for s in unique_subjects:
        mask = (subjects == s)
        X_s = X_all[mask]
        y_s = y_all[mask]
        if len(X_s) < n_samples:
            continue
        idx = np.random.choice(len(X_s), n_samples, replace=False)
        domains.append({'X': X_s[idx], 'y': y_s[idx]})
    return domains


def split_domains(domains, n_test=10, n_val=5, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    shuffled = domains.copy()
    np.random.shuffle(shuffled)
    return shuffled[n_test + n_val:], shuffled[n_test:n_test + n_val], shuffled[:n_test]


def train_spadg_kernel(train_domains, test_domains):
    model = SpADG_Kernel_Regression(lambda_=0.001, gamma_kx=0.1, sigma_p=0.5, random_state=SEED)
    model.fit(train_domains)
    rmses, maes = [], []
    for t in test_domains:
        rmse, mae = model.evaluate(t)
        rmses.append(rmse)
        maes.append(mae)
    return np.mean(rmses), np.mean(maes)


def save_results(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")


def run_kernel_experiment(data_path, target='motor'):
    print(f"\n=== Kernel Park Experiment ({target} UPDRS) ===")
    features, motor_UPDRS, total_UPDRS, subjects = load_parkinsons_data(data_path)
    
    if target == 'motor':
        y_all = motor_UPDRS
    else:
        y_all = total_UPDRS
    
    domains = preprocess_parkinsons_data(features, y_all, subjects, n_samples=10)
    train_domains, val_domains, test_domains = split_domains(domains, n_test=10, n_val=5, random_state=SEED)
    if len(train_domains) > 25:
        train_domains = train_domains[:25]
    print(f"Train: {len(train_domains)}, Val: {len(val_domains)}, Test: {len(test_domains)}")
    results = {'target': [], 'algorithm': [], 'rmse': [], 'mae': []}
    print("Training SpADG...")
    rmse, mae = train_spadg_kernel(train_domains, test_domains)
    print(f"  SpADG: RMSE={rmse:.4f}, MAE={mae:.4f}")
    results['target'].append(target)
    results['algorithm'].append('SpADG')
    results['rmse'].append(rmse)
    results['mae'].append(mae)
    return results


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), 'parkinsons_updrs.data')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    motor_results = run_kernel_experiment(data_path, target='motor')
    save_results(motor_results, f"Park_kernel_motor_results_{timestamp}.csv")
    
    total_results = run_kernel_experiment(data_path, target='total')
    save_results(total_results, f"Park_kernel_total_results_{timestamp}.csv")
    
    print("\n=== Kernel Park Experiment Completed ===")
