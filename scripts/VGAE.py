import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn as nn

class VGAE_encoder(nn.Module):
    """
    Variational Graph Autoencoder (VGAE) encoder module.

    Structure:
        input features  →  GraphConvolution  →  LayerNorm  →
        Linear layers → (μ, logσ²) → Reparameterization → Decoder

    Args:
        in_dim (int): Number of input features (nodes × feature dim).
        hidden_dim1 (int): Hidden dimension of first GCN layer.
        hidden_dim2 (int): Latent embedding dimension.
        dropout (float): Dropout rate applied to input features.
    """
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, dropout):
        super(VGAE_encoder, self).__init__()
        self.gc1 = GraphConvolution(in_dim, hidden_dim1,dropout, act=F.relu)
        self.ln1 = nn.LayerNorm(hidden_dim1)
        self.gc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)       
        self.gc3 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.dc = InnerProductDecoder(dropout)
    
    def forward(self, g, features):
        hidden1 = self.ln1(self.gc1(adj=g, input=features))
        mu = self.gc2(hidden1)
        logvar = self.gc3(hidden1)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar, z
    
    def reparameterize(self, mu, logvar):
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)


        
class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = torch.mm(z, z.t())
        return adj



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features,  dropout=0.,act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, adj, input):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'