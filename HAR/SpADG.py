#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
from scipy.spatial.distance import cdist

class SpADG_Kernel:
    def __init__(self, kernel_type='rbf', kappa_type='rbf', lambda_=0.001,
                 max_iter=1000, tol=1e-3, gamma_kx=1.0, gamma_kx_prime=1.0, 
                 sigma_p=1.0, degree=2, coef0=1.0, random_state=None):
        self.kernel_type = kernel_type
        self.kappa_type = kappa_type
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.gamma_kx = gamma_kx
        self.gamma_kx_prime = gamma_kx_prime
        self.sigma_p = sigma_p
        self.degree = degree
        self.coef0 = coef0
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def _compute_dimension_wise_embedding(self, X, dim):
        X_dim = X[:, dim].reshape(-1, 1)
        if self.kernel_type == 'rbf' or self.kappa_type == 'rbf':
            K_internal = rbf_kernel(X_dim, gamma=self.gamma_kx_prime)
            return K_internal.mean(axis=0)
        elif self.kernel_type == 'linear' or self.kappa_type == 'linear':
            return X_dim.mean(axis=0)
        elif self.kernel_type == 'poly' or self.kappa_type == 'poly':
            K_internal = polynomial_kernel(X_dim, degree=self.degree, gamma=self.gamma_kx_prime, coef0=self.coef0)
            return K_internal.mean(axis=0)

    def _compute_kernel(self, X1, X2=None, kernel_type='rbf'):
        if X2 is None:
            X2 = X1
        if kernel_type == 'rbf':
            return rbf_kernel(X1, X2, gamma=self.gamma_kx)
        elif kernel_type == 'linear':
            return linear_kernel(X1, X2)
        elif kernel_type == 'poly':
            return polynomial_kernel(X1, X2, degree=self.degree, gamma=self.gamma_kx, coef0=self.coef0)

    def fit(self, domains):
        domains_X = [task['X'] for task in domains]
        domains_y = [task['y'] for task in domains]
        n_domains = len(domains_X)

        self.scalers = {}
        self.domain_embeddings = {}
        self.domains_info = {}

        all_train_data = np.vstack(domains_X)
        global_mean = np.mean(all_train_data, axis=0)
        global_std = np.std(all_train_data, axis=0) + 1e-8

        for domain_idx, (X, y) in enumerate(zip(domains_X, domains_y)):
            X_scaled = (X - global_mean) / global_std
            self.scalers[domain_idx] = {'mean': global_mean, 'std': global_std}
            n_dims = X_scaled.shape[1]
            dim_embeddings = []
            for dim in range(n_dims):
                emb = self._compute_dimension_wise_embedding(X_scaled, dim)
                dim_embeddings.append(emb)
            self.domain_embeddings[domain_idx] = dim_embeddings
            self.domains_info[domain_idx] = {'n_samples': X_scaled.shape[0]}

        X_all = np.vstack([(X - global_mean) / global_std for X in domains_X])
        y_all = np.concatenate(domains_y)
        domain_ids = np.concatenate([[i] * len(y) for i, y in enumerate(domains_y)])
        n_samples, n_dims = X_all.shape

        self.X_train = X_all
        self.y_train = y_all
        self.domain_ids_train = domain_ids
        self.n_dims = n_dims
        self.n_domains = n_domains
        self.tau = np.ones(n_dims)

        self.K_components = []
        for dim in range(n_dims):
            K_dim = np.zeros((n_samples, n_samples))
            X_dim = X_all[:, dim].reshape(-1, 1)
            K_sample = self._compute_kernel(X_dim, kernel_type=self.kernel_type)

            for i in range(n_samples):
                dom_i = domain_ids[i]
                emb_i = self.domain_embeddings[dom_i][dim]
                for j in range(n_samples):
                    dom_j = domain_ids[j]
                    emb_j = self.domain_embeddings[dom_j][dim]
                    if self.kappa_type == 'rbf':
                        dist_sq = np.sum((emb_i - emb_j) ** 2)
                        k_domain = np.exp(-dist_sq / (2 * self.sigma_p ** 2))
                    elif self.kappa_type == 'linear':
                        k_domain = np.dot(emb_i, emb_j)
                    elif self.kappa_type == 'poly':
                        k_domain = (self.coef0 + np.dot(emb_i, emb_j)) ** self.degree
                    K_dim[i, j] = k_domain * K_sample[i, j]
            self.K_components.append(K_dim)

        self.alpha = np.zeros((len(self.K_components), n_samples))
        self._fit_l2_regularization(y_all)
        return self

    def _fit_l2_regularization(self, y):
        n_samples = len(y)
        eta = 0.01
        alpha_old = self.alpha.copy()

        for epoch in range(self.max_iter):
            pred = np.zeros(n_samples)
            for t in range(len(self.K_components)):
                pred += np.dot(self.K_components[t], self.alpha[t])
            margins = y * pred
            hinge_grad = -np.where(margins < 1, y, 0)
            for t in range(len(self.K_components)):
                grad = np.dot(self.K_components[t], hinge_grad) / n_samples
                alpha_update = self.alpha[t] - eta * grad
                norm = np.linalg.norm(alpha_update, 2)
                scale = max(0, 1 - eta * self.lambda_ * self.tau[t] / (norm + 1e-8))
                self.alpha[t] = scale * alpha_update
            diff = np.linalg.norm(self.alpha - alpha_old)
            if diff < self.tol:
                break
            alpha_old = self.alpha.copy()

    def predict(self, X, domain_idx=0):
        decisions = self.decision_function(X, domain_idx)
        return np.sign(decisions)

    def decision_function(self, X, domain_idx):
        scaler_params = self.scalers.get(domain_idx, self.scalers[0])
        X_scaled = (X - scaler_params['mean']) / (scaler_params['std'] + 1e-8)
        n_test_samples, n_dims = X_scaled.shape
        domain_embeddings = []
        for dim in range(n_dims):
            emb = self._compute_dimension_wise_embedding(X_scaled, dim)
            domain_embeddings.append(emb)

        decisions = np.zeros(n_test_samples)
        for i in range(n_test_samples):
            for dim in range(n_dims):
                k_val = 0
                for train_dom_idx in range(self.n_domains):
                    train_emb = self.domain_embeddings[train_dom_idx][dim]
                    if self.kappa_type == 'rbf':
                        dist_sq = np.sum((train_emb - domain_embeddings[dim]) ** 2)
                        k_domain = np.exp(-dist_sq / (2 * self.sigma_p ** 2))
                    elif self.kappa_type == 'linear':
                        k_domain = np.dot(train_emb, domain_embeddings[dim])
                    elif self.kappa_type == 'poly':
                        k_domain = (self.coef0 + np.dot(train_emb, domain_embeddings[dim])) ** self.degree
                    if self.kernel_type == 'rbf':
                        dists = (self.X_train[:, dim] - X_scaled[i, dim]) ** 2
                        k_sample = np.exp(-self.gamma_kx * dists)
                    elif self.kernel_type == 'linear':
                        k_sample = self.X_train[:, dim] * X_scaled[i, dim]
                    k_val += np.sum(k_domain * k_sample * self.alpha[dim])
                decisions[i] += k_val
        return decisions

    def evaluate(self, test_task):
        X_test = test_task['X']
        y_test = test_task['y']
        decisions = self.decision_function(X_test, domain_idx=0)
        y_pred = np.sign(decisions)
        accuracy = np.mean(y_pred == y_test)
        return accuracy, y_pred


class SpADG_Kernel_Regression:
    def __init__(self, kernel_type='rbf', kappa_type='rbf', lambda_=0.001,
                 max_iter=500, tol=1e-3, gamma_kx=0.1, gamma_kx_prime=0.1,
                 sigma_p=0.5, epsilon=0.1, random_state=None):
        self.kernel_type = kernel_type
        self.kappa_type = kappa_type
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.gamma_kx = gamma_kx
        self.gamma_kx_prime = gamma_kx_prime
        self.sigma_p = sigma_p
        self.epsilon = epsilon
        self.random_state = random_state

    def _compute_dimension_wise_embedding(self, X, dim):
        X_dim = X[:, dim].reshape(-1, 1)
        if self.kernel_type == 'rbf':
            K_internal = rbf_kernel(X_dim, gamma=self.gamma_kx_prime)
            return K_internal.mean()
        return X_dim.mean()

    def fit(self, tasks):
        domains_X = [task['X'] for task in tasks]
        domains_y = [task['y'] for task in tasks]
        n_domains = len(domains_X)

        all_train_data = np.vstack(domains_X)
        global_mean = np.mean(all_train_data, axis=0)
        global_std = np.std(all_train_data, axis=0) + 1e-8

        self.domain_embeddings = {}
        for domain_idx, (X, y) in enumerate(zip(domains_X, domains_y)):
            X_scaled = (X - global_mean) / global_std
            dim_embeddings = []
            for dim in range(X_scaled.shape[1]):
                emb = self._compute_dimension_wise_embedding(X_scaled, dim)
                dim_embeddings.append(emb)
            self.domain_embeddings[domain_idx] = dim_embeddings

        X_all = np.vstack([(X - global_mean) / global_std for X in domains_X])
        y_all = np.concatenate(domains_y)
        domain_ids = np.concatenate([[i] * len(y) for i, y in enumerate(domains_y)])
        n_samples, n_dims = X_all.shape

        self.X_train = X_all
        self.y_train = y_all
        self.domain_ids_train = domain_ids
        self.n_dims = n_dims
        self.n_domains = n_domains
        self.y_mean = np.mean(y_all)
        self.y_std = np.std(y_all) + 1e-8
        self.y_train_scaled = (y_all - self.y_mean) / self.y_std
        self.X_mean = global_mean
        self.X_std = global_std
        self.tau = np.ones(n_dims)

        self.K_components = []
        for dim in range(n_dims):
            K_dim = np.zeros((n_samples, n_samples))
            X_dim = X_all[:, dim].reshape(-1, 1)
            K_sample = rbf_kernel(X_dim, gamma=self.gamma_kx)

            for i in range(n_samples):
                dom_i = domain_ids[i]
                emb_i = self.domain_embeddings[dom_i][dim]
                for j in range(n_samples):
                    dom_j = domain_ids[j]
                    emb_j = self.domain_embeddings[dom_j][dim]
                    dist_sq = np.sum((emb_i - emb_j) ** 2)
                    k_domain = np.exp(-dist_sq / (2 * self.sigma_p ** 2))
                    K_dim[i, j] = k_domain * K_sample[i, j]
            self.K_components.append(K_dim)

        self.alpha = np.zeros((n_dims, n_samples))
        self._fit_epsilon_insensitive_loss()
        return self

    def _fit_epsilon_insensitive_loss(self):
        n_samples = len(self.y_train_scaled)
        eta = 0.001
        alpha_old = self.alpha.copy()

        for epoch in range(2000):
            if epoch % 30 == 0 and epoch > 0:
                eta *= 0.95

            pred = np.zeros(n_samples)
            for t in range(self.n_dims):
                pred += np.dot(self.K_components[t], self.alpha[t])

            diff = self.y_train_scaled - pred
            loss_grad = -2 * diff / n_samples

            for t in range(self.n_dims):
                loss_grad += 2 * self.lambda_ * np.dot(self.K_components[t], self.alpha[t])

            grad_norm = np.linalg.norm(loss_grad)
            if grad_norm > 5.0:
                loss_grad = loss_grad * 5.0 / grad_norm

            for t in range(self.n_dims):
                grad = np.dot(self.K_components[t], loss_grad)
                alpha_update = self.alpha[t] - eta * grad
                norm = np.linalg.norm(alpha_update, 2)
                if norm > 500:
                    alpha_update = alpha_update * 500 / norm
                self.alpha[t] = alpha_update

            if epoch > 10 and np.linalg.norm(self.alpha - alpha_old) < self.tol:
                break
            alpha_old = self.alpha.copy()

    def predict(self, X, domain_idx=0):
        X_scaled = (X - self.X_mean) / (self.X_std + 1e-8)
        n_test_samples, n_dims = X_scaled.shape
        domain_embeddings = []
        for dim in range(n_dims):
            emb = self._compute_dimension_wise_embedding(X_scaled, dim)
            domain_embeddings.append(emb)

        decisions = np.zeros(n_test_samples)
        for i in range(n_test_samples):
            for dim in range(n_dims):
                alpha_dim = self.alpha[dim]
                dists = (self.X_train[:, dim] - X_scaled[i, dim]) ** 2
                k_sample = np.exp(-self.gamma_kx * dists)
                k_val = 0
                for j in range(len(self.X_train)):
                    train_dom_idx = self.domain_ids_train[j]
                    train_emb = self.domain_embeddings[train_dom_idx][dim]
                    dist_sq = np.sum((train_emb - domain_embeddings[dim]) ** 2)
                    k_domain = np.exp(-dist_sq / (2 * self.sigma_p ** 2))
                    k_val += k_domain * k_sample[j] * alpha_dim[j]
                decisions[i] += k_val

        return decisions * self.y_std + self.y_mean

    def evaluate(self, test_task):
        X_test = test_task['X']
        y_test = test_task['y']
        predictions = self.predict(X_test)
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        mae = np.mean(np.abs(predictions - y_test))
        return rmse, mae


class FeatureNN(nn.Module):
    def __init__(self, output_dim=1, m=100):
        super().__init__()
        self.fc1 = nn.Linear(1, m)
        self.fc2 = nn.Linear(m, 50)
        self.fc3 = nn.Linear(50, output_dim, bias=False)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class SpADG_NN(nn.Module):
    def __init__(self, n_features, m=100):
        super().__init__()
        self.n_features = n_features
        self.feature_nns = nn.ModuleList([FeatureNN(m=m) for _ in range(n_features)])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, kme_values):
        feature_outputs = []
        for i in range(self.n_features):
            sub_out = self.feature_nns[i](x[:, [i]])
            kme_i = kme_values[:, [i]]
            feature_output = sub_out * kme_i
            feature_outputs.append(feature_output)
        output = feature_outputs[0] + self.bias
        for i in range(1, self.n_features):
            output += feature_outputs[i]
        return output.squeeze(1)


def calculate_kme_values(x, domain_ids, domain_data_dict, sigma=1.0):
    batch_size, n_features = x.shape
    kme_values = torch.ones(batch_size, n_features, device=x.device)
    x_np = x.detach().cpu().numpy()
    domain_ids_np = domain_ids.detach().cpu().numpy()

    for i in range(batch_size):
        sample_x = x_np[i]
        domain_id = domain_ids_np[i]
        if domain_id in domain_data_dict:
            domain_X = domain_data_dict[domain_id]
            for j in range(n_features):
                sample_feature = sample_x[j:j+1]
                domain_features = domain_X[:, j:j+1]
                dist_sq = np.sum((sample_feature - domain_features) ** 2, axis=1)
                K = np.exp(-dist_sq / (2 * sigma**2))
                kme = np.mean(K)
                kme_values[i, j] = torch.tensor(kme, device=x.device)
    return kme_values
