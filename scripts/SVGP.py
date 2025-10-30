import torch
import torch.nn as nn
from torch.optim.lr_scheduler import *
import numpy as np
from kernel import MultiMaternKernel, CauchyKernel


def _add_diagonal_jitter(matrix, jitter=1e-8):
    Eye = torch.eye(matrix.size(-1), device=matrix.device).expand(matrix.shape)
    return matrix + jitter * Eye


class SVGP(nn.Module):
    def __init__(self, fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, jitter, N_train, dim, Multi ,dtype, device):
        super(SVGP, self).__init__()
        self.N_train = N_train
        self.jitter = jitter
        self.dtype = dtype
        self.device = device
        self.multi = Multi
        # inducing points
        if fixed_inducing_points:
            self.inducing_index_points = torch.tensor(initial_inducing_points, dtype=dtype).to(device)
        else:
            self.inducing_index_points = nn.Parameter(torch.tensor(initial_inducing_points, dtype=dtype).to(device), requires_grad=True)

        # length scale of the kernel
        self.kernel = CauchyKernel(scale=kernel_scale, fixed_scale=fixed_gp_params, dtype=dtype, device=device).to(device)
        self.MultiMaternKernel = MultiMaternKernel(scale=kernel_scale, fixed_scale=fixed_gp_params,dim = dim, dtype=dtype, device=device).to(device)

    def kernel_matrix(self, x, y, l=None, x_inducing=True, y_inducing=True, diag_only=False):
        """
        Computes GP kernel matrix K(x,y).
        :param x:
        :param y:
        :param x_inducing: whether x is a set of inducing points
        :param y_inducing: whether y is a set of inducing points
        :param diag_only: whether or not to only compute diagonal terms of the kernel matrix
        :return:
        """
        if self.multi:
            if diag_only:
                matrix = self.MultiMaternKernel.forward_diag(x, y, l)
            else:
                matrix = self.MultiMaternKernel(x, y, l)
        else:
            if diag_only:
                matrix = self.kernel.forward_diag(x, y)
            else:
                matrix = self.kernel(x, y)
        return matrix
    
    def stable_inv(self, matrix):
        jitter_matrix = self.jitter * torch.eye(matrix.size(-1), device=matrix.device)
        chol = torch.linalg.cholesky(matrix + jitter_matrix)
        return torch.cholesky_inverse(chol)

    def compute_variational_and_posterior(self, x, y, noise, l=None):
        """
        Computes L_H for the data in the current batch.
        :param x: auxiliary data for current batch (batch, 1 + 1 + M)
        :param y: mean vector for current latent channel, output of the encoder network (batch, 1)
        :param noise: variance vector for current latent channel, output of the encoder network (batch, 1)
        :Returns:
        - variational_loss (L_3_sum_term, KL_term)
        - posterior parameters (mean_vector, B)
        """
        b = x.shape[0]
        m = self.inducing_index_points.shape[0]
        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points, l=l) # (m,m)
        K_mm_inv = self.stable_inv(K_mm) # (m,m)
        K_nn = self.kernel_matrix(x, x, l=l, x_inducing=False, y_inducing=False, diag_only=True) # (b)
        K_nm = self.kernel_matrix(x, self.inducing_index_points, l=l, x_inducing=False)  # (b, m)
        K_mn = K_mn = K_nm.T
#        S = A_hat
        scaled_K_nm = K_nm / noise[:, None]
        sigma_l = K_mm + (self.N_train / b) * torch.matmul(K_mn, scaled_K_nm)
        sigma_l_inv = self.stable_inv(sigma_l)
        # Mu_hat
        mu_hat = (self.N_train / b) * torch.matmul(torch.matmul(K_mm, sigma_l_inv),
                                                   torch.matmul(K_mn, y / noise))  # (m, 1)
        # KL term
        mean_vector = torch.matmul(K_nm, torch.matmul(K_mm_inv, mu_hat))
        # Posterior covariance
        K_xm = K_nm  # Reusing K_nm as test and train are the same
        K_mx = K_xm.T  # (m, b)
        K_xx = K_nn  # Reusing K_nn as diagonal-only
        K_xm_Sigma_l_K_mx = torch.matmul(K_xm, torch.matmul(sigma_l_inv, K_mx))
        K_xm_K_mm_inv = torch.matmul(K_xm, K_mm_inv)  # (b, m)
        K_xm_term = torch.matmul(K_xm_K_mm_inv, K_mx)  # (b, b)
        B = K_xx + torch.diagonal(-K_xm_term + K_xm_Sigma_l_K_mx, dim1=0, dim2=1)  # (b,)
        # A_hat
        A_hat = torch.matmul(K_mm, torch.matmul(sigma_l_inv, K_mm))  # (m, m)
        # KL divergence terms
        K_mm_chol = torch.linalg.cholesky(_add_diagonal_jitter(K_mm, self.jitter))
        S_chol = torch.linalg.cholesky(_add_diagonal_jitter(A_hat, self.jitter))
        
        logdet_K_mm = 2 * torch.sum(torch.log(torch.diagonal(K_mm_chol)))  # 使用对角线直接求 log(det)
        logdet_S = 2 * torch.sum(torch.log(torch.diagonal(S_chol)))
        trace_term = torch.einsum('ij,ji->', K_mm_inv, A_hat)  # 利用 einsum 优化 trace 计算
        mu_term = torch.vdot(mu_hat.squeeze(), K_mm_inv @ mu_hat.squeeze())
        KL_term = 0.5 * (logdet_K_mm - logdet_S - m + trace_term + mu_term)

        # Variational loss terms
        K_tilde_terms = (1 / noise) * (
                K_nn - torch.sum(K_nm * (K_mm_inv @ K_nm.T).T, dim=1)  # 直接利用 sum 替代 diagonal
        )
        M = K_mm_inv @ A_hat @ K_mm_inv  # (m, m)
        temp = torch.matmul(K_nm, M)  # (b, m)
        term_per_sample = torch.sum(temp * K_nm, dim=1)  # (b,)
        trace_terms = (1 / noise) * term_per_sample

        L_3_sum_term = -0.5 * (
                torch.sum(K_tilde_terms) + torch.sum(trace_terms) +
                torch.sum(torch.log(noise)) + b * np.log(2 * np.pi) +
                torch.sum((1 / noise) * (y - mean_vector) ** 2)
        )
        return L_3_sum_term, KL_term, mean_vector, B
