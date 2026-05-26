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
import os

from SpADG import SpADG_NN, calculate_kme_values


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
    y_binary = np.where(y_binary == 2, 0, 1)
    return X_binary, y_binary, subject_binary


def preprocess_har_data(X_all, y_all, subject_all, n_samples):
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


def prepare_nn_data(domains, domain_indices):
    X = np.vstack([domains[i]['X'] for i in domain_indices])
    y = np.hstack([domains[i]['y'] for i in domain_indices])
    ids = np.hstack([[i] * len(domains[i]['y']) for i in domain_indices])
    return X, y, ids


def create_dataloader(X, y, ids, batch_size=32, shuffle=True):
    return DataLoader(TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y).long(), torch.LongTensor(ids)), 
                     batch_size=batch_size, shuffle=shuffle)


def custom_hinge_loss(outputs, targets):
    return torch.mean(torch.max(torch.zeros_like(outputs), 1 - targets * outputs))


def train_spadg_nn(train_domains, val_domains, test_domains, n_features, lr=5e-3, max_epoch=20, lbd=0.1, m=100, sigma=1.0):
    train_data = prepare_nn_data(train_domains, range(len(train_domains)))
    val_data = prepare_nn_data(val_domains, range(len(val_domains)))
    test_data = prepare_nn_data(test_domains, range(len(test_domains)))
    trainloader = create_dataloader(*train_data, batch_size=32, shuffle=True)
    valloader = create_dataloader(*val_data, batch_size=32, shuffle=False)
    testloader = create_dataloader(*test_data, batch_size=32, shuffle=False)
    train_dict = {i: train_domains[i]['X'] for i in range(len(train_domains))}
    val_dict = {i: val_domains[i]['X'] for i in range(len(val_domains))}
    test_dict = {i: test_domains[i]['X'] for i in range(len(test_domains))}
    model = SpADG_NN(n_features, m=m).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    best_state = None
    for epoch in range(max_epoch):
        model.train()
        for inputs, targets, ids in trainloader:
            optimizer.zero_grad()
            targets_hinge = targets.float() * 2 - 1
            kme = calculate_kme_values(inputs, ids, train_dict, sigma=sigma)
            outputs = model(inputs, kme)
            loss = custom_hinge_loss(outputs, targets_hinge)
            reg = sum(torch.norm(p) for n, p in model.named_parameters() if 'fc3' in n and 'weight' in n)
            loss += lbd * torch.sqrt(reg)
            loss.backward()
            optimizer.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets, ids in valloader:
                kme = calculate_kme_values(inputs, ids, val_dict, sigma=sigma)
                outputs = model(inputs, kme)
                correct += ((outputs > 0).float() * 2 - 1 == targets.float() * 2 - 1).sum().item()
                total += len(targets)
        val_acc = 100 * correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
    model.load_state_dict(best_state)
    model.eval()
    preds, targets_list = [], []
    with torch.no_grad():
        for inputs, targets, ids in testloader:
            kme = calculate_kme_values(inputs, ids, test_dict, sigma=sigma)
            outputs = model(inputs, kme)
            preds.extend(((outputs > 0).float().cpu().numpy()))
            targets_list.extend(targets.numpy())
    preds_np = np.array(preds) * 2 - 1
    targets_np = np.array(targets_list) * 2 - 1
    acc = 100 * np.mean(preds_np == targets_np)
    f1 = f1_score(targets_np, preds_np)
    tn, fp, fn, tp = confusion_matrix(targets_np, preds_np).ravel()
    return acc, f1, tp / (tp + fn) if (tp + fn) > 0 else 0, fp / (fp + tn) if (fp + tn) > 0 else 0


def save_results(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")


def run_nn_experiment(data_path):
    print("\n=== NN HAR Experiment ===")
    X_all, y_all, subject_all = load_har_data(data_path)
    domains = preprocess_har_data(X_all, y_all, subject_all, n_samples=50)
    train_domains, val_domains, test_domains = split_domains(domains, n_test=5, n_val=5, random_state=SEED)
    train_domains = train_domains[:20]
    n_features = train_domains[0]['X'].shape[1]
    print(f"Train: {len(train_domains)}, Val: {len(val_domains)}, Test: {len(test_domains)}")
    results = {'algorithm': [], 'accuracy': [], 'f1': [], 'tpr': [], 'fpr': []}
    print("Training SpADG...")
    acc, f1, tpr, fpr = train_spadg_nn(train_domains, val_domains, test_domains, n_features)
    print(f"  SpADG: Acc={acc:.2f}%, F1={f1:.4f}")
    results['algorithm'].append('SpADG')
    results['accuracy'].append(acc)
    results['f1'].append(f1)
    results['tpr'].append(tpr)
    results['fpr'].append(fpr)
    return results


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), 'Dataset')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nn_results = run_nn_experiment(data_path)
    save_results(nn_results, f"HAR_nn_results_{timestamp}.csv")
    print("\n=== NN HAR Experiment Completed ===")
