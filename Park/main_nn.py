#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from datetime import datetime

device = torch.device("cpu")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

from SpADG import SpADG_NN, calculate_kme_values


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


def prepare_nn_data(domains, domain_indices):
    X = np.vstack([domains[i]['X'] for i in domain_indices])
    y = np.hstack([domains[i]['y'] for i in domain_indices])
    ids = np.hstack([[i] * len(domains[i]['y']) for i in domain_indices])
    return X, y, ids


def create_dataloader(X, y, ids, batch_size=32, shuffle=True):
    return DataLoader(TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y), torch.LongTensor(ids)), 
                     batch_size=batch_size, shuffle=shuffle)


def train_spadg_nn(train_domains, val_domains, test_domains, n_features, lr=5e-3, max_epoch=50, lbd=1.0, m=50, sigma=1.0):
    train_data = prepare_nn_data(train_domains, range(len(train_domains)))
    val_data = prepare_nn_data(val_domains, range(len(val_domains)))
    test_data = prepare_nn_data(test_domains, range(len(test_domains)))
    all_train = np.vstack([d['X'] for d in train_domains])
    all_train_y = np.hstack([d['y'] for d in train_domains])
    g_mean, g_std = np.mean(all_train, axis=0), np.std(all_train, axis=0) + 1e-8
    y_mean, y_std = np.mean(all_train_y), np.std(all_train_y) + 1e-8
    X_tr = (train_data[0] - g_mean) / g_std
    y_tr = (train_data[1] - y_mean) / y_std
    X_val = (val_data[0] - g_mean) / g_std
    y_val = (val_data[1] - y_mean) / y_std
    X_te = (test_data[0] - g_mean) / g_std
    y_te = (test_data[1] - y_mean) / y_std
    train_dict = {i: (train_domains[i]['X'] - g_mean) / g_std for i in range(len(train_domains))}
    val_dict = {i: (val_domains[i]['X'] - g_mean) / g_std for i in range(len(val_domains))}
    test_dict = {i: (test_domains[i]['X'] - g_mean) / g_std for i in range(len(test_domains))}
    trainloader = create_dataloader(X_tr, y_tr, train_data[2], batch_size=32, shuffle=True)
    valloader = create_dataloader(X_val, y_val, val_data[2], batch_size=32, shuffle=False)
    testloader = create_dataloader(X_te, y_te, test_data[2], batch_size=32, shuffle=False)
    model = SpADG_NN(n_features, m=m).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_rmse = float('inf')
    best_state = None
    for epoch in range(max_epoch):
        model.train()
        for inputs, targets, ids in trainloader:
            optimizer.zero_grad()
            kme = calculate_kme_values(inputs, ids, train_dict, sigma=sigma)
            outputs = model(inputs, kme)
            loss = nn.functional.mse_loss(outputs, targets)
            reg = sum(torch.norm(p) for n, p in model.named_parameters() if 'weight' in n)
            loss += lbd * reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for inputs, targets, ids in valloader:
                kme = calculate_kme_values(inputs, ids, val_dict, sigma=sigma)
                outputs = model(inputs, kme)
                preds.extend(outputs.cpu().numpy())
        val_rmse = np.sqrt(np.mean((np.array(preds) - y_val) ** 2))
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict().copy()
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for inputs, targets, ids in testloader:
            kme = calculate_kme_values(inputs, ids, test_dict, sigma=sigma)
            outputs = model(inputs, kme)
            preds.extend((outputs * y_std + y_mean).cpu().numpy())
    preds = np.array(preds)
    targets = test_data[1]
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = np.mean(np.abs(preds - targets))
    return rmse, mae


def save_results(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")


def run_nn_experiment(data_path, target='motor'):
    print(f"\n=== NN Park Experiment ({target} UPDRS) ===")
    features, motor_UPDRS, total_UPDRS, subjects = load_parkinsons_data(data_path)
    
    if target == 'motor':
        y_all = motor_UPDRS
    else:
        y_all = total_UPDRS
    
    domains = preprocess_parkinsons_data(features, y_all, subjects, n_samples=10)
    train_domains, val_domains, test_domains = split_domains(domains, n_test=10, n_val=5, random_state=SEED)
    if len(train_domains) > 25:
        train_domains = train_domains[:25]
    n_features = train_domains[0]['X'].shape[1]
    print(f"Train: {len(train_domains)}, Val: {len(val_domains)}, Test: {len(test_domains)}")
    results = {'target': [], 'algorithm': [], 'rmse': [], 'mae': []}
    print("Training SpADG...")
    rmse, mae = train_spadg_nn(train_domains, val_domains, test_domains, n_features)
    print(f"  SpADG: RMSE={rmse:.4f}, MAE={mae:.4f}")
    results['target'].append(target)
    results['algorithm'].append('SpADG')
    results['rmse'].append(rmse)
    results['mae'].append(mae)
    return results


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), 'parkinsons_updrs.data')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    motor_results = run_nn_experiment(data_path, target='motor')
    save_results(motor_results, f"Park_nn_motor_results_{timestamp}.csv")
    
    total_results = run_nn_experiment(data_path, target='total')
    save_results(total_results, f"Park_nn_total_results_{timestamp}.csv")
    
    print("\n=== NN Park Experiment Completed ===")
