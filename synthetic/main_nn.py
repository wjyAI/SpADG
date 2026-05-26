#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, confusion_matrix

device = torch.device("cpu")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

from SpADG import SpADG_NN, calculate_kme_values


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


def prepare_nn_data(domains, domain_indices):
    X = np.vstack([domains[i]['X'] for i in domain_indices])
    y = np.hstack([domains[i]['y'] for i in domain_indices])
    ids = np.hstack([[i] * len(domains[i]['y']) for i in domain_indices])
    return X, (y + 1) / 2, ids


def create_dataloader(X, y, ids, batch_size=32, shuffle=True):
    return DataLoader(TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y).long(), torch.LongTensor(ids)), 
                     batch_size=batch_size, shuffle=shuffle)


def custom_hinge_loss(outputs, targets):
    return torch.mean(torch.max(torch.zeros_like(outputs), 1 - targets * outputs))


def train_spadg_nn(train_data, test_data, n_features, lr=5e-3, max_epoch=20, lbd=0.1, m=100, sigma=1.0):
    X_train, y_train, ids_train = train_data
    X_test, y_test, ids_test = test_data
    trainloader = create_dataloader(X_train, y_train, ids_train, batch_size=32, shuffle=True)
    testloader = create_dataloader(X_test, y_test, ids_test, batch_size=32, shuffle=False)
    domain_dict = {i: X_train[ids_train == i] for i in np.unique(ids_train)}
    model = SpADG_NN(n_features, m=m).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(max_epoch):
        model.train()
        for inputs, targets, ids in trainloader:
            optimizer.zero_grad()
            targets_hinge = targets.float() * 2 - 1
            kme = calculate_kme_values(inputs, ids, domain_dict, sigma=sigma)
            outputs = model(inputs, kme)
            loss = custom_hinge_loss(outputs, targets_hinge)
            reg = sum(torch.norm(p) for n, p in model.named_parameters() if 'fc3' in n and 'weight' in n)
            loss += lbd * torch.sqrt(reg)
            loss.backward()
            optimizer.step()
    model.eval()
    preds, targets_list = [], []
    with torch.no_grad():
        for inputs, targets, ids in testloader:
            kme = calculate_kme_values(inputs, ids, {i: X_test[ids_test == i] for i in np.unique(ids_test)}, sigma=sigma)
            outputs = model(inputs, kme)
            preds.extend((outputs > 0).cpu().numpy())
            targets_list.extend(targets.numpy())
    preds = np.array(preds) * 2 - 1
    targets_np = np.array(targets_list) * 2 - 1
    acc = 100 * np.mean(preds == targets_np)
    f1 = f1_score(targets_np, preds)
    tn, fp, fn, tp = confusion_matrix(targets_np, preds).ravel()
    return acc, f1, tp / (tp + fn) if (tp + fn) > 0 else 0, fp / (fp + tn) if (fp + tn) > 0 else 0


def save_results(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def run_nn_experiment():
    print("\n=== NN Synthetic Experiment ===")
    domains = generate_domain_data(n_domains=40, n_samples=32, n_features=50, random_state=SEED)
    all_idx = list(range(40))
    np.random.shuffle(all_idx)
    train_idx, test_idx = all_idx[:30], all_idx[30:]
    train_data = prepare_nn_data(domains, train_idx)
    test_data = prepare_nn_data(domains, test_idx)
    results = {'algorithm': [], 'accuracy': [], 'f1': [], 'tpr': [], 'fpr': []}
    print("Training SpADG...")
    acc, f1, tpr, fpr = train_spadg_nn(train_data, test_data, n_features=50)
    print(f"  SpADG: Acc={acc:.2f}%, F1={f1:.4f}, TPR={tpr:.4f}, FPR={fpr:.4f}")
    results['algorithm'].append('SpADG')
    results['accuracy'].append(acc)
    results['f1'].append(f1)
    results['tpr'].append(tpr)
    results['fpr'].append(fpr)
    return results


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nn_results = run_nn_experiment()
    save_results(nn_results, f"synthetic_nn_results_{timestamp}.csv")
    print("\n=== NN Synthetic Experiment Completed ===")
