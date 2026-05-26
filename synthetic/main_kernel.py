#!/usr/bin/env python3

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, confusion_matrix

SEED = 42
np.random.seed(SEED)

from SpADG import SpADG_Kernel


def generate_domain_data(n_domains=40, n_samples=32, n_features=50, random_state=42):
    np.random.seed(random_state)
    domains = []
    for d in range(n_domains):
        alpha = np.random.uniform(np.pi / 6, np.pi / 3)
        a, b = 1.0, 0.5
        theta = np.random.uniform(0, 2 * np.pi, n_samples)
        r = np.sqrt(np.random.uniform(0, 1, n_samples))
        x_orig = a * r * np.cos(theta)
        y_orig = b * r * np.sin(theta)
        rot = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        points = np.vstack([x_orig, y_orig])
        rot_pts = np.dot(rot, points)
        x_rot, y_rot = rot_pts[0, :], rot_pts[1, :]
        short_axis = np.array([-np.sin(alpha), np.cos(alpha)])
        proj = x_rot * short_axis[0] + y_rot * short_axis[1]
        y = np.sign(proj)
        X = np.zeros((n_samples, n_features))
        X[:, 0] = x_rot
        X[:, 1] = y_rot
        X[:, 2:] = np.random.uniform(-1.0, 1.0, (n_samples, n_features - 2))
        domains.append({'X': X, 'y': y})
    return domains


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
    print(f"Results saved to {filename}")


def run_kernel_experiment():
    print("\n=== Kernel Synthetic Experiment ===")
    domains = generate_domain_data(n_domains=40, n_samples=32, n_features=50, random_state=SEED)
    all_idx = list(range(40))
    np.random.shuffle(all_idx)
    train_idx, test_idx = all_idx[:30], all_idx[30:]
    train_domains = [domains[i] for i in train_idx]
    test_domains = [domains[i] for i in test_idx]
    results = {'algorithm': [], 'accuracy': [], 'f1': [], 'tpr': [], 'fpr': []}
    print("Training SpADG...")
    acc, f1, tpr, fpr = train_spadg_kernel(train_domains, test_domains)
    print(f"  SpADG: Acc={acc:.2f}%, F1={f1:.4f}, TPR={tpr:.4f}, FPR={fpr:.4f}")
    results['algorithm'].append('SpADG')
    results['accuracy'].append(acc)
    results['f1'].append(f1)
    results['tpr'].append(tpr)
    results['fpr'].append(fpr)
    return results


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kernel_results = run_kernel_experiment()
    save_results(kernel_results, f"synthetic_kernel_results_{timestamp}.csv")
    print("\n=== Kernel Synthetic Experiment Completed ===")
