from collections import deque
from SVGP import SVGP
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import torch.optim as optim
from torch.optim.lr_scheduler import *
import torch.nn.functional as F
from torch.nn.modules.module import Module
from VGAE import VGAE_encoder
import torch.nn as nn
import torch
from utils import *
import pandas as pd
from itertools import combinations
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import math

class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, modelfile='model.pt', pretrain=False):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.inf
        self.model_file = modelfile
        self.pretrain = pretrain

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if not self.pretrain:
                    model.load_model(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss


class SPANETREG(nn.Module):
    """spaNetReg main module."""
    def __init__(self, input_dim, vgae_input_dim, GP_dim, Normal_dim, encoder_layers, encoder_dropout, decoder_dropout, 
                    fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, N_train, 
                    init_beta, gamma, dtype, device, vgae_encoder_layers = [64, 32]):
        super(SPANETREG, self).__init__()
        torch.set_default_dtype(dtype)
        self.svgp = SVGP(fixed_inducing_points=fixed_inducing_points, initial_inducing_points=initial_inducing_points,
                fixed_gp_params=fixed_gp_params, kernel_scale=kernel_scale, jitter=1e-8, N_train=N_train, dim = GP_dim,
                 Multi = True, dtype=dtype, device=device)
        self.input_dim = input_dim      # number of spots
        self.beta = init_beta           # weight on KL terms (GP + Normal)
        self.gamma = gamma              # weight on VGAE loss
        self.dtype = dtype
        self.GP_dim = GP_dim            # dimension of latent Gaussian process embedding
        self.Normal_dim = Normal_dim    # dimension of latent standard Gaussian embedding
        self.device = device
        self.encoder = DenseEncoder(input_dim=input_dim, hidden_dims=encoder_layers, output_dim=GP_dim+Normal_dim, activation="elu", dropout=encoder_dropout)
        self.decoder = spavaeDecoder(dropout=decoder_dropout)
        self.vgae_encoder = VGAE_encoder(vgae_input_dim, *vgae_encoder_layers, 0)
        self.to(device) 
    

    def save_model(self, path):
        torch.save(self.state_dict(), path)


    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


    def pretrain_vgae(self, features, norm, adj_norm, adj_label, weight_tensor, 
                      norm_val, adj_norm_val, adj_label_val, weight_tensor_val,
                     pretrain_epochs=5000, pretrain_lr=0.001, pretrain_patience = 1000):
        """
        Warm up the VGAE branch before joint training (stabilizes training).

        Minimizes BCE (with class weights) + KL of VGAE posterior.
        Early-stops on validation loss with `pretrain_patience`.
        """
        n_nodes = features.shape[0]
        for param in self.parameters():
            param.requires_grad = False
        for param in self.vgae_encoder.parameters():
            param.requires_grad = True
        vgae_optimizer = torch.optim.Adam(self.vgae_encoder.parameters(), lr=pretrain_lr)
        early_stopping = EarlyStopping(patience=pretrain_patience, modelfile='checkpoint.pt',pretrain=True)
        for epoch in range(pretrain_epochs):
            # ---- Train loss
            logits, mu, logvar, _ = self.vgae_encoder(adj_norm, features)
            loss = norm * F.binary_cross_entropy_with_logits(
                logits.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor,
                reduction = 'sum'
            )
            KLD = -0.5 / n_nodes * torch.sum(
                1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2))
            loss += KLD

            vgae_optimizer.zero_grad()
            loss.backward()
            vgae_optimizer.step()
            
            # ---- Validation loss
            val_logits, val_mu, val_logvar, _ = self.vgae_encoder(adj_norm_val, features)
            val_loss = norm_val * F.binary_cross_entropy_with_logits(
                val_logits.view(-1), adj_label_val.to_dense().view(-1), weight=weight_tensor_val,
                reduction = 'sum'
            )
            val_KLD = -0.5 / n_nodes * torch.sum(
                1 + 2 * val_logvar - val_mu.pow(2) - val_logvar.exp().pow(2))
            val_loss += val_KLD
            
            val_loss = val_loss.item()
            print(f"VGAE pretrain Epoch {epoch+1}/{pretrain_epochs}, Loss: {loss:.4f}")
            early_stopping(val_loss, self.vgae_encoder)
            if early_stopping.early_stop:
                print("EARLY STOPPING")
                break
            
        # Unfreeze for joint training
        for param in self.parameters():
            param.requires_grad = True


    def forward(self, x, y, adj, features, norm, adj_norm, adj_label, weight_tensor, num_samples=1):
        """Forward pass and compute composite ELBO.

        Args:
            x (Tensor): mini-batch of positions.
            y (Tensor): mini-batch of RP score.
            features (Tensor): (TFs x spots) for VGAE
            norm, adj_norm, adj_label, weight_tensor: graph inputs for VGAE loss
            num_samples (int): number of samplings of the posterior distribution of latent embedding.
        """

        self.train()
        b = y.shape[0]
        n_nodes = features.shape[0]
        
        # ---- Encoder: mean/var of Gaussian process and standard Gaussian latents
        qnet_mu, qnet_var = self.encoder(y)

        gp_mu = qnet_mu[:, 0:self.GP_dim]
        gp_var = qnet_var[:, 0:self.GP_dim]

        gaussian_mu = qnet_mu[:, self.GP_dim:]
        gaussian_var = qnet_var[:, self.GP_dim:]

        inside_elbo_recon, inside_elbo_kl = [], []
        gp_p_m, gp_p_v = [], []
         
        if self.GP_dim > 0:
            for l in range(self.GP_dim):
                inside_elbo_recon_l,  inside_elbo_kl_l, gp_p_m_l, gp_p_v_l = self.svgp.compute_variational_and_posterior(x=x,
                                                                        y=gp_mu[:, l], noise=gp_var[:, l], l=l)

                inside_elbo_recon.append(inside_elbo_recon_l)
                inside_elbo_kl.append(inside_elbo_kl_l)
                gp_p_m.append(gp_p_m_l)
                gp_p_v.append(gp_p_v_l)

            
            inside_elbo_recon = torch.stack(inside_elbo_recon, dim=-1)
            inside_elbo_kl = torch.stack(inside_elbo_kl, dim=-1)
            inside_elbo_recon = torch.sum(inside_elbo_recon)
            inside_elbo_kl = torch.sum(inside_elbo_kl)

            inside_elbo = inside_elbo_recon - (b / self.svgp.N_train) * inside_elbo_kl

            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)
            # cross entropy term
            gp_ce_term = gauss_cross_entropy(gp_p_m, gp_p_v, gp_mu, gp_var)
            gp_ce_term = torch.sum(gp_ce_term)
            # KL term of GP prior
            gp_KL_term = gp_ce_term - inside_elbo

        else :
            inside_elbo_recon = torch.tensor(0.0, device=self.device)
            inside_elbo_kl = torch.tensor(0.0, device=self.device)
            inside_elbo = inside_elbo_recon
            gp_p_m = torch.tensor([], device=self.device)
            gp_p_v = torch.tensor([], device=self.device)
            gp_ce_term = torch.tensor(0.0, device=self.device)
            gp_KL_term = torch.tensor(0.0, device=self.device)

        # KL term of Gaussian prior
        gaussian_prior_dist = Normal(torch.zeros_like(gaussian_mu), torch.ones_like(gaussian_var))
        gaussian_post_dist = Normal(gaussian_mu, torch.sqrt(gaussian_var))
        gaussian_KL_term=kl_divergence(gaussian_post_dist, gaussian_prior_dist).sum()

        # SAMPLE
        p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
        p_v = torch.cat((gp_p_v, gaussian_var), dim=1)

        latent_dist = Normal(p_m, torch.sqrt(p_v))
        latent_samples = []
        mean_samples = []
        disp_samples = []
        for _ in range(num_samples):
            latent_samples_ = latent_dist.rsample()
            latent_samples.append(latent_samples_)
        
        recon_loss = 0
        
        # ---- VGAE
        logits ,mu, logvar, vgae_z = self.vgae_encoder(adj_norm, features)
        # compute vgae loss
        vgae_loss = norm * F.binary_cross_entropy_with_logits(
            logits.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor,
            reduction = 'sum'
        )
        # KL term of vgae
        KLD = -0.5 / n_nodes * torch.sum(
                1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2))
        vgae_loss += KLD

        
        for f in latent_samples:
            hidden_samples = self.decoder(f, vgae_z)
            recon_loss += F.mse_loss(hidden_samples, y, reduction = 'none').sum()
            
        recon_loss = recon_loss / num_samples
         
        # ELBO
        elbo = recon_loss + self.beta * gp_KL_term + self.beta * gaussian_KL_term + self.gamma * vgae_loss

        return elbo, recon_loss, gp_KL_term, gaussian_KL_term, inside_elbo, gp_ce_term, gp_p_m, gp_p_v, qnet_mu, qnet_var, \
            mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples, vgae_loss, logits, vgae_z




    def train_model(self, pos, features, adj_train, adj_val,
                    lr=0.001, weight_decay=0.001, batch_size=256,
                    num_samples=1, train_size=0.95, 
                    maxiter=5000, patience=100, save_model=True, model_weights="model.pt", print_kernel_scale=True,
                    pretrain_vgae=True, pretrain_epochs=5000, pretrain_lr=0.001,pretrain_patience=1000):
        """
        Model training.

        Args:
            pos (np.ndarray): Spatial coordinates (N x 2)
            features (Tensor): (spots x TFs) for VGAE
            adj_train/adj_val (sparse): Sparse train/val graphs for VGAE
            lr, weight_decay (float): Optimizer hyperparameter
            batch_size (int): Batch size for mini-batch
            train_size (float): Proportion of training size, the other samples are validations of VAE
            maxiter (int): Maximum number of training epochs
            patience (int): Early stopping patience for ELBO validation
            print_kernel_scale (bool): Whether to print current kernel scale during training steps
            pretrain_vgae (bool): Whether to pre-train the VGAE encoder
            pretrain_epochs (int): Number of epochs for VGAE pre-training.
            pretrain_lr (float): Learning rate during VGAE pre-training.
            pretrain_patience (int): Early stopping patience for VGAE pre-training.
        """
        
        
        self.train()
        features = features.to(self.device)
        ncounts = features.T

        # Preprocess graph adjacency for VGAE
        norm, adj_norm, adj_label, weight_tensor = preprocess_graph(adj_train, device=self.device)
        norm_val, adj_norm_val, adj_label_val, weight_tensor_val = preprocess_graph(adj_val, device=self.device)
        
        # VGAE pretraining
        if pretrain_vgae:
            self.pretrain_vgae(features=features,norm=norm,adj_norm=adj_norm,
                               adj_label=adj_label,weight_tensor=weight_tensor,
                               norm_val=norm_val,adj_norm_val=adj_norm_val,
                               adj_label_val=adj_label_val, weight_tensor_val=weight_tensor_val,
                               pretrain_epochs=pretrain_epochs,pretrain_lr=pretrain_lr,
                               pretrain_patience=pretrain_patience
        )
        
        dataset = TensorDataset(torch.tensor(pos, dtype=self.dtype),ncounts)

        if train_size < 1:
            train_dataset, validate_dataset = random_split(dataset=dataset, lengths=[int(train_size * len(dataset)), len(dataset)-int(train_size * len(dataset))])
            validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        else:
            train_dataset = dataset
        
        if ncounts.shape[0]*train_size > batch_size:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        
        early_stopping = EarlyStopping(patience=patience, modelfile=model_weights)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        print("Training")

        for epoch in range(maxiter):
            elbo_val = 0
            recon_loss_val = 0
            gp_KL_term_val = 0
            gaussian_KL_term_val = 0
            num = 0
            vgae_loss_val = 0
            num2 = 0     
            
            
            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                elbo, recon_loss, gp_KL_term, gaussian_KL_term, inside_elbo, gp_ce_term, p_m, p_v, qnet_mu, qnet_var, \
                    mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples, vgae_loss, logits, vgae_z = \
                    self.forward(x=x_batch, y=y_batch, num_samples=num_samples, adj=adj_train, features=features, 
                                 norm=norm, adj_norm=adj_norm, adj_label=adj_label, weight_tensor=weight_tensor)

                self.zero_grad()
                elbo.backward(retain_graph=True)
                optimizer.step()

                elbo_val += elbo.item()
                recon_loss_val += recon_loss.item()
                gp_KL_term_val += gp_KL_term.item()
                
                gaussian_KL_term_val += gaussian_KL_term.item()
                
                num += x_batch.shape[0]
                vgae_loss_val += vgae_loss.item()
                num2 += 1
            

            elbo_val = elbo_val/num
            recon_loss_val = recon_loss_val/num
            gp_KL_term_val = gp_KL_term_val/num
            gaussian_KL_term_val = gaussian_KL_term_val/num
            vgae_loss_val = vgae_loss_val/num2

            print('Training epoch {}, ELBO:{:.8f}, recon_loss:{:.8f}, GP KLD loss:{:.8f}, Gaussian KLD loss:{:.8f}, VGAE_loss:{:.8f}'.format(epoch+1, elbo_val, recon_loss_val, gp_KL_term_val, gaussian_KL_term_val, vgae_loss_val))
            print('Current beta', self.beta)

            if print_kernel_scale:
                print('Current kernel scale', torch.clamp(F.softplus(self.svgp.kernel.scale), min=1e-10, max=1e4).data)


            if train_size < 1:
                validate_elbo_val = 0
                validate_num = 0                
                
                for _, (validate_x_batch, validate_y_batch) in enumerate(validate_dataloader):
                    validate_x_batch = validate_x_batch.to(self.device)
                    validate_y_batch = validate_y_batch.to(self.device)

                    validate_elbo, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _= \
                        self.forward(x=validate_x_batch, y=validate_y_batch, num_samples=num_samples,  adj=adj_val, features=features,
                                     norm=norm_val, adj_norm=adj_norm_val, adj_label=adj_label_val, weight_tensor=weight_tensor_val)

                    validate_elbo_val += validate_elbo.item()
                    validate_num += validate_x_batch.shape[0]

                validate_elbo_val = validate_elbo_val / validate_num

                print("Training epoch {}, validating ELBO:{:.8f}".format(epoch+1, validate_elbo_val))
                early_stopping(validate_elbo_val, self)
                if early_stopping.early_stop:
                    print('EarlyStopping: run {} iteration'.format(epoch+1))
                    break
            
        if save_model:
            torch.save(self.state_dict(), model_weights)
            
    def evaluate(self, features, adj_full, ref_file_path, gene_names, return_metrics=False, cutoff = 0.8):
        """
        Evaluate spaNetReg on spatial data and compute regulatory network metrics.

        Args:
            adj_full (scipy.sparse): Full adjacency matrix
            cutoff (float): Quantile used as threshold for binarization 
        """
        
        self.eval()
        features = features.to(self.device)
        _, adj_norm, _, _ = preprocess_graph(adj_full, device=self.device)
        logits ,_, _, vgae_z = self.vgae_encoder(adj_norm, features)
        vgae_z = vgae_z.cpu().detach()
        prob_matrix = sigmoid(logits.cpu().detach())
        prob_matrix_df = pd.DataFrame(prob_matrix.cpu().numpy(), index=gene_names, columns=gene_names)


        # Construct null distribution by randomly shuffling embeddings
        null_distributions = []
        for _ in range(100):
            z_flat = vgae_z.flatten()
            z_perm = z_flat[torch.randperm(z_flat.size(0))].view(vgae_z.size())
            null_logits = torch.sigmoid(torch.matmul(z_perm, z_perm.t()))
            null_distributions.append(null_logits)
        null_distributions = torch.stack(null_distributions, dim=0)
        threshold = torch.quantile(null_distributions, cutoff)

        # Binarize network
        adj_matrix_pred = (prob_matrix >= threshold).int()
        adj_matrix_pred_df = pd.DataFrame(adj_matrix_pred, index=gene_names, columns=gene_names)

        
        
        
        # Evaluation
        if ref_file_path is None:
            print("No ground truth provided, returning predictions only.")
            return prob_matrix_df, adj_matrix_pred_df, vgae_z
        
        
        edges_df_sorted = convert_similarity_to_edge_list(prob_matrix_df)
        predEdgeDF = edges_df_sorted

        true_df = pd.read_csv(ref_file_path,sep = ',', header = 0, index_col = None)
        genes = pd.unique(true_df[['Gene1', 'Gene2']].values.ravel())
        edge_combinations = list(combinations(genes, 2))
        true_edge_set = set(tuple(sorted((g1, g2))) for g1, g2 in zip(true_df['Gene1'], true_df['Gene2']))
        
        pred_edge_dict = {tuple(sorted((g1, g2))): w for g1, g2, w in zip(predEdgeDF['Gene1'], predEdgeDF['Gene2'], predEdgeDF['EdgeWeight'])}
        all_edges = []
        for edge in edge_combinations:
            key = tuple(sorted(edge))
            label = 1 if key in true_edge_set else 0
            score = abs(pred_edge_dict.get(key, 0))
            all_edges.append((label, score))

        pos_samples = [x for x in all_edges if x[0] == 1]
        neg_samples = [x for x in all_edges if x[0] == 0]

        if len(pos_samples) == 0:
            raise ValueError("No positive samples found.")
        if len(neg_samples) == 0:
            raise ValueError("No negative samples found.")



        roc_true = [x[0] for x in all_edges]
        roc_scores = [x[1] for x in all_edges]
        fpr, tpr, _ = roc_curve(roc_true, roc_scores)
        prec, recall, _ = precision_recall_curve(roc_true, roc_scores)

        adj_pred = adj_matrix_pred.cpu().numpy().astype(int)
        np.fill_diagonal(adj_pred, 0)
        adj_true_df = pd.DataFrame(0, index=gene_names, columns=gene_names)
        for _, row in true_df.iterrows():
            tf = row["Gene1"]
            gene = row["Gene2"]
            if tf in gene_names and gene in gene_names:
                adj_true_df.loc[tf, gene] = 1
                adj_true_df.loc[gene, tf] = 1 
        adj_true_ref = adj_true_df.values.astype(int)
        pred_flat = adj_pred.flatten()
        true_flat = adj_true_ref.flatten()
        tp = ((pred_flat == 1) & (true_flat == 1)).sum()
        fp = ((pred_flat == 1) & (true_flat == 0)).sum()
        tn = ((pred_flat == 0) & (true_flat == 0)).sum()
        fn = ((pred_flat == 0) & (true_flat == 1)).sum()

        precision = tp / (tp + fp) 
        recall_TPR = tp / (tp + fn) 
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * precision * recall_TPR / (precision + recall_TPR) 
        specificity = tn / (tn + fp) 
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator == 0:
            mcc = 0.0
        else:
            mcc = (tp * tn - fp * fn) / denominator


        print("End of training!")
        print('Allauprc:',auc(recall, prec))
        print('auroc:',auc(fpr, tpr))

        print(f"Precision: {precision:.4f}")
        print(f"Recall_TPR: {recall_TPR:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Specificity_TNR: {specificity:.4f}")
        print('tp:', tp, 'fp:', fp, 'tn:', tn, 'fn:', fn)
        print(f"🧮 MCC: {mcc:.4f}")
        
        if return_metrics:
            return {
                'auprc': auc(recall, prec),
                'auroc': auc(fpr, tpr),
                'precision': precision,
                'recall': recall_TPR,
                'accuracy': accuracy,
                'f1': f1,
                'mcc': mcc
            },  prob_matrix_df, adj_matrix_pred_df, vgae_z