import torch
import torch.nn as nn
from torch.optim.lr_scheduler import *
import numpy as np
import scanpy as sc
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.special import expit as sigmoid
import scipy.sparse as sp


class DenseEncoder(nn.Module):
    """
    Multi-layer perceptron encoder that outputs the mean and variance
    of latent variables for the spatial VAE.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, activation="relu", dropout=0, dtype=torch.float32, norm="batchnorm"):
        super(DenseEncoder, self).__init__()
        self.layers = buildNetwork([input_dim]+hidden_dims, network="decoder", activation=activation, dropout=dropout, dtype=dtype, norm=norm)
        self.enc_mu = nn.Linear(hidden_dims[-1], output_dim)
        self.enc_var = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        h = self.layers(x)
        mu = self.enc_mu(h)
        var = torch.exp(self.enc_var(h).clamp(-15, 15))
        return mu, var


def buildNetwork(layers, network="decoder", activation="relu", dropout=0., dtype=torch.float32, norm="batchnorm"):
    """
    Construct a feed-forward neural network given a list of layer sizes.
    """
    
    net = []
    if network == "encoder" and dropout > 0:
        net.append(nn.Dropout(p=dropout))
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if norm == "batchnorm":
            net.append(nn.BatchNorm1d(layers[i]))
        elif norm == "layernorm":
            net.append(nn.LayerNorm(layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="elu":
            net.append(nn.ELU())
        if dropout > 0:
            net.append(nn.Dropout(p=dropout))
    return nn.Sequential(*net)

class spavaeDecoder(nn.Module):

    def __init__(self, dropout):
        super(spavaeDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, spavae, vgae_z):
        spavae = self.dropout(spavae)
        vgae_z = self.dropout(vgae_z)
        features = torch.mm(spavae, vgae_z.t())
        return features


def gauss_cross_entropy(mu1, var1, mu2, var2):
    """
    Computes the element-wise cross entropy
    Given q(z) ~ N(z| mu1, var1)
    returns E_q[ log N(z| mu2, var2) ]
    args:
        mu1:  mean of expectation (batch, tmax, 2) tf variable
        var1: var  of expectation (batch, tmax, 2) tf variable
        mu2:  mean of integrand (batch, tmax, 2) tf variable
        var2: var of integrand (batch, tmax, 2) tf variable
    returns:
        cross_entropy: (batch, tmax, 2) tf variable
    """

    term0 = 1.8378770664093453  # log(2*pi)
    term1 = torch.log(var2)
    term2 = (var1 + mu1 ** 2 - 2 * mu1 * mu2 + mu2 ** 2) / var2

    cross_entropy = -0.5 * (term0 + term1 + term2)

    return cross_entropy


def sparse_to_tuple(sparse_mx):
    """Convert a scipy sparse matrix to tuple representation (coords, values, shape)"""
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj, device = 'cuda'):
    """Normalize adjacency matrix"""
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = (
        adj_.dot(degree_mat_inv_sqrt)
        .transpose()
        .dot(degree_mat_inv_sqrt)
        .tocoo()
    )
    adj_norm = sparse_to_tuple(adj_normalized)
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = (
            adj.shape[0]
            * adj.shape[0]
            / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        )
    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    adj_norm = torch.sparse.FloatTensor(
            torch.LongTensor(adj_norm[0].T),
            torch.DoubleTensor(adj_norm[1]),
            torch.Size(adj_norm[2]),
        ).to(device)
    adj_label = torch.sparse.FloatTensor(
            torch.LongTensor(adj_label[0].T),
            torch.DoubleTensor(adj_label[1]),
            torch.Size(adj_label[2]),
        ).to(device)


    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0), dtype=torch.float64).to(device)
    weight_tensor[weight_mask] = pos_weight
    return norm, adj_norm, adj_label, weight_tensor


def compute_loss_weights(labels):

    batch_size = labels.size(0)
    num_positive = labels.sum().item()
    num_negative = batch_size - num_positive

    if num_positive == 0:
        pos_weight = 1.0
    else:
        pos_weight = float(num_negative) / float(num_positive)

    if num_negative == 0:
        norm = 1.0
    else:
        norm = batch_size / (2.0 * num_negative)

    weight_mask = labels == 1
    weight_tensor = torch.ones(batch_size, dtype=torch.float64)
    weight_tensor[weight_mask] = pos_weight

    return norm, weight_tensor


class ParallelDataLoader:
    """
    Parallel DataLoader to iterate two loaders simultaneously,
    with optional cycling of each loader.

    Parameters
    ----------
    loader1 : DataLoader
        First DataLoader.
    loader2 : DataLoader
        Second DataLoader.
    cycle_flags : tuple of bool
        Whether to cycle each loader when exhausted. (cycle_loader1, cycle_loader2)
    """

    def __init__(self, loader1, loader2, cycle_flags=(False, False)):
        self.loader1 = loader1
        self.loader2 = loader2
        self.cycle1, self.cycle2 = cycle_flags

    def __iter__(self):
        self.iter1 = iter(self.loader1)
        self.iter2 = iter(self.loader2)
        return self

    def __next__(self):
        try:
            batch1 = next(self.iter1)
        except StopIteration:
            if self.cycle1:
                self.iter1 = iter(self.loader1)
                batch1 = next(self.iter1)
            else:
                raise StopIteration
        
        try:
            batch2 = next(self.iter2)
        except StopIteration:
            if self.cycle2:
                self.iter2 = iter(self.loader2)
                batch2 = next(self.iter2)
            else:
                raise StopIteration

        return batch1, batch2

    def __len__(self):
        if self.cycle1 or self.cycle2:
            raise ValueError("ParallelDataLoader with cycling does not have a defined length.")
        return min(len(self.loader1), len(self.loader2))
    
def mask_test_edges(adj, test=False):
    """
    Split the adjacency matrix into training, validation, and test edge sets.
    
    Args:
        test (bool): Whether to create a held-out test set (default=False).
    """

    adj = adj - sp.dia_matrix(
        (adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape
    )
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.0)) if test else 0
    num_val = int(np.floor(edges.shape[0] / 20.0))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val : (num_val + num_test)] if test else []
    train_edge_idx = all_edge_idx[(num_val + num_test):] if test else all_edge_idx[num_val:]
    test_edges = edges[test_edge_idx] if test else np.array([])
    val_edges = edges[val_edge_idx]
    train_edges = edges[train_edge_idx]

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    if test:
        while len(test_edges_false) < len(test_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if ismember([idx_j, idx_i], np.array(test_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(test_edges_false)):
                    continue
            test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], test_edges):
            continue
        if ismember([idx_j, idx_i], test_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
    if test:
        assert ~ismember(test_edges_false, edges_all)
        assert ~ismember(test_edges, train_edges)
        assert ~ismember(val_edges, test_edges)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)


    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix(
        (data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape
    )
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    if test:
        return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
    else:
        return adj_train, train_edges, val_edges, val_edges_false
    

def ismember(a, b, tol=5):
    rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    return np.any(rows_close)

def findByRow(mat, row):
    return np.where((mat == row).all(1))[0]

def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    """
    Normalize AnnData object
    """
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata


def convert_similarity_to_edge_list(prob_matrix_df):
    """
    Convert a dense symmetric similarity (probability) matrix
    into an edge list suitable for evaluation.
    """
    edges = []
    genes = prob_matrix_df.columns
    TF = prob_matrix_df.index


    for i in range(len(TF)):
        for j in range(i + 1, len(genes)):
            gene1 = TF[i]
            gene2 = genes[j]
            edge_weight = prob_matrix_df.iloc[i, j]
            edges.append((gene1, gene2, edge_weight))


    edges_df = pd.DataFrame(edges, columns=['Gene1', 'Gene2', 'EdgeWeight'])


    edges_df_sorted = edges_df.sort_values(by='EdgeWeight', ascending=False).reset_index(drop=True)

    return edges_df_sorted