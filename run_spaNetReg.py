"""
run_spaNetReg.py
-------------------
Main training script for spaNetReg: a dependency–aware graph neural networks
designed to reconstruct transcription factors regulatory networks (TRNs) from spatial ATAC-seq data.

Description:
------------
This script loads the input matrices (TF–TF adjacency, TF–spot RP scores, and spatial coordinates). 
The trained model predicts TRNs, which are saved as adjacency and probability matrices.

Input files (for each sample):
    - {sample}/{sample}.txt                    : Binary adjacency matrix of the TF network
    - {sample}/{sample}_TF_rp_score.txt        : TF-by-spot regulatory potential matrix (features)
    - {sample}/{sample}_pos.csv                : Spatial coordinates of each spot

Output files:
    - {sample}/result/prob_matrix.txt        : Predicted edge probability matrix
    - {sample}/result/adj_matrix_pred.txt    : Binarized predicted adjacency matrix
"""


import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from time import time
from scripts.utils import *
from scripts.spaNetReg import SPANETREG
import argparse


# torch.manual_seed(123)
# np.random.seed(123)

if __name__ == "__main__":
    
    # setting the parameters
    parser = argparse.ArgumentParser(description= 'Dependency-aware graph neural networks')
    parser.add_argument('--sample',type=str, help='Sample name')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--patience', type=int,default=150, help='Early stopping patience')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='Weight decay for optimizer')
    parser.add_argument('--inducing_point_steps', default=15, type=int)
    parser.add_argument('--loc_range', default=20., type=float)
    parser.add_argument('--GP_dim', default=5, type=int, help='dimension of the latent Gaussian process embedding')
    parser.add_argument('--Normal_dim', default=27, type=int, help='dimension of the latent standard Gaussian embedding')
    args = parser.parse_args()

    # ==========================
    #  Data loading
    # ==========================
    work_dir = os.getcwd()
    
    # Load binary adjacency matrix (TF–TF network)
    df_ = pd.read_csv(f"{work_dir}/{args.sample}/{args.sample}.txt", sep="\t")
    gene_names = df_.iloc[:, 0].values.tolist()
    df = df_.values[:,1:]
    adj = sp.csr_matrix(df.astype(int))

    # Load feature matrix (TF regulatory potential scores)
    df = pd.read_csv(f"{work_dir}/{args.sample}/{args.sample}_TF_rp_score.txt", sep="\t")
    features = torch.tensor(np.array(df), dtype=torch.float64)
    tf_num, spot_num = features.shape

    # Load spatial coordinates
    pos_path = f"{work_dir}/{args.sample}/{args.sample}_pos.csv"
    df = pd.read_csv(pos_path, header=None)
    selected_columns = df.iloc[:, 1:3]
    loc = selected_columns.to_numpy()
    scaler = MinMaxScaler()
    loc = scaler.fit_transform(loc) * args.loc_range

    # ==========================
    #  Graph data preparation
    # ==========================
    (
        adj_train,
        train_edges,
        val_edges,
        val_edges_false
    ) = mask_test_edges(adj)
    
    data = np.ones(val_edges.shape[0])
    adj_val = sp.csr_matrix((data, (val_edges[:, 0], val_edges[:, 1])), shape=adj.shape)
    adj_val = adj_val + adj_val.T

    # Generate inducing points for the Gaussian process prior.
    # Controlled by the argument "inducing_point_steps":
    #   - It controls number of grid steps.
    #   - The total number of inducing points is (inducing_point_steps + 1)^2.
    eps = 1e-5
    initial_inducing_points = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * args.loc_range

    # ==========================
    #  Model construction
    # ==========================
    model = SPANETREG(input_dim=tf_num, vgae_input_dim = spot_num, GP_dim=args.GP_dim, Normal_dim=args.Normal_dim, 
                encoder_layers=[128, 64],
                encoder_dropout=0, decoder_dropout=0,
                fixed_inducing_points=True, initial_inducing_points=initial_inducing_points, 
                fixed_gp_params=False, kernel_scale=20, N_train=spot_num, 
                init_beta=10, gamma = 1000,
                dtype=torch.float64, device=args.device,
                vgae_encoder_layers = [64, 32])
    
    # ==========================
    #  Model training
    # ==========================
    t0 = time()
    model.train_model(pos=loc, features=features, adj_train=adj_train, adj_val=adj_val, 
                    lr=1e-5,weight_decay=args.weight_decay, maxiter=5000, 
                        patience=args.patience, batch_size = 256,
                        model_weights=f"{args.sample}_model.pt",
                        pretrain_vgae=True, pretrain_epochs=5000, pretrain_lr=0.001,
                        pretrain_patience=1000)
    print('Training time: %d seconds.' % int(time() - t0))
    
    
    # ==========================
    #  Model output
    # ==========================
    prob_matrix_df, adj_matrix_pred_df, _ = model.evaluate(features=features, adj_full=adj, gene_names=gene_names,
                            ref_file_path=None,cutoff=0.80)

    result_file_path = f'{work_dir}/{args.sample}/result'
    os.makedirs(result_file_path, exist_ok=True)

    prob_matrix_df.to_csv(f"{result_file_path}/prob_matrix.txt", sep='\t')
    adj_matrix_pred_df.to_csv(f"{result_file_path}/adj_matrix_pred.txt", sep='\t')