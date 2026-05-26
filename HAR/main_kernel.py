#!/usr/bin/env python3

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, confusion_matrix

SEED = 0
np.random.seed(SEED)
import os

from SpADG import SpADG_Kernel


def load_har_data(data_path):
    X_train = np.loadtxt(os.path.join(data_path, 'train', 'X_train.txt'))
    X_test = np.loadtxt(os.path.join(data_path, 'test', 'X_test.txt'))
    y_train = np.loadtxt(os.path.join(data_path, 'train', 'y_train.txt'))
    y_test = np.loadtxt(os.path.join(data_path, 'test', 'y_test.txt'))
    subject_train = np.loadtxt(os.path.join(data_path, 'train', 'subject_train.txt'))
    subject_test = np.loadtxt(os.path.join(data_path, 'test', 'subject_test.txt'))
    X_all = np.vstack((X_train, X_test))
    y_all = np.hstack((y_train, y_test))
    subject_all = np.hstack((subject_train, subject_test))
    mask = (y_all == 2) | (y_all == 3)
    X_binary = X_all[mask]
    y_binary = y_all[mask]
    subject_binary = subject_all[mask]
    y_binary = np.where(y_binary == 2, -1, 1)
    return X_binary, y_binary, subject_binary


def preprocess_har_data(X_all, y_all, subject_all, n_samples, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    unique_subjects = np.unique(subject_all)
    domains = []
    for s in unique_subjects:
        mask = (subject_all == s)
        X_s = X_all[mask]
        y_s = y_all[mask]
        if len(X_s) < n_samples:
            continue
        idx = np.random.choice(len(X_s), n_samples, replace=False)
        domains.append({'X': X_s[idx], 'y': y_s[idx]})
    return domains


def split_domains(domains, n_test=5, n_val=5, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    shuffled = domains.copy()
    np.random.shuffle(shuffled)
    return shuffled[n_test + n_val:], shuffled[n_test:n_test + n_val], shuffled[:n_test]


def train_spadg_kernel(train_domains, test_domains):
    model = SpADG_Kernel(lambda_=0.001, gamma_kx=1.0, sigma_p=1.0, random_state=SEED)
    model.fit(train_domains)
    accs, preds_all, targets_all = [], [], []
    for t in test_domains:
        acc, pred = model.evaluate(t)
        accs.append(acc)
        preds_all.extend(pred)
        targets_all.extend(t['y'])
    preds_np = np.array(preds_all)
    targets_np = np.array(targets_all)
    f1 = f1_score(targets_np, preds_np)
    tn, fp, fn, tp = confusion_matrix(targets_np, preds_np).ravel()
    return np.mean(accs) * 100, f1, tp / (tp + fn) if (tp + fn) > 0 else 0, fp / (fp + tn) if (fp + tn) > 0 else 0


def save_results(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")


def run_kernel_experiment(data_path):
    print("\n=== Kernel HAR Experiment ===")
    X_all, y_all, subject_all = load_har_data(data_path)
    domains = preprocess_har_data(X_all, y_all, subject_all, n_samples=50, random_state=SEED)
    train_domains, val_domains, test_domains = split_domains(domains, n_test=5, n_val=5, random_state=SEED)
    np.random.seed(SEED)
    train_domains = np.random.choice(train_domains, 20, replace=False).tolist()
    print(f"Train: {len(train_domains)}, Val: {len(val_domains)}, Test: {len(test_domains)}")
    results = {'algorithm': [], 'accuracy': [], 'f1': [], 'tpr': [], 'fpr': []}
    print("Training SpADG...")
    acc, f1, tpr, fpr = train_spadg_kernel(train_domains, test_domains)
    print(f"  SpADG: Acc={acc:.4f}%, F1={f1:.4f}")
    results['algorithm'].append('SpADG')
    results['accuracy'].append(acc)
    results['f1'].append(f1)
    results['tpr'].append(tpr)
    results['fpr'].append(fpr)
    return results


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), 'Dataset')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kernel_results = run_kernel_experiment(data_path)
    save_results(kernel_results, f"HAR_kernel_results_{timestamp}.csv")
    print("\n=== Kernel HAR Experiment Completed ===")
