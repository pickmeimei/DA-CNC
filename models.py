import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
from torch.autograd import Function
import math


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grads):
        dx = ctx.lambda_ * grads.neg()

        return dx, None


def uniform_neib_sampler(adj, ids, n_samples, device='cpu'):
    tmp = adj[ids]
    perm = torch.randperm(tmp.shape[1]).to(device)
    tmp = tmp[:, perm]

    return tmp[:, :n_samples]


class GraphSAGE(nn.Module):
    def __init__(self, aggregator_class, input_dim, layer_specs, device='cpu'):
        super(GraphSAGE, self).__init__()
        self.sample_fns = [partial(uniform_neib_sampler, n_samples=s['n_sample'], device=device) for s in layer_specs]
        agg_layers = []
        for spec in layer_specs:
            agg = aggregator_class(input_dim=input_dim, output_dim=spec['output_dim'], activation=spec['activation'])
            agg_layers.append(agg)
            input_dim = agg.output_dim
        self.agg_layers = nn.Sequential(*agg_layers)

    def forward(self, ids, adj, feats):
        tmp_feats = feats[ids]
        all_feats = [tmp_feats]
        for _, sampler_fn in enumerate(self.sample_fns):
            ids = sampler_fn(adj=adj, ids=ids).contiguous().view(-1)
            tmp_feats = feats[ids]
            all_feats.append(tmp_feats)
        for agg_layer in self.agg_layers.children():
            all_feats = [agg_layer(all_feats[k], all_feats[k + 1]) for k in range(len(all_feats) - 1)]
        assert len(all_feats) == 1, "len(all_feats) != 1"
        out = all_feats[0]

        return out


class VGAE_GraphSAGE(nn.Module):
    def __init__(self, aggregator_class, input_dim, layer_specs, latent_dim=64, device='cpu'):
        super(VGAE_GraphSAGE, self).__init__()
        self.sage = GraphSAGE(aggregator_class, input_dim, layer_specs, device)
        self.mu_layer = nn.Linear(2*layer_specs[-1]['output_dim'], 2*layer_specs[-1]['output_dim'])
        self.logvar_layer = nn.Linear(2*layer_specs[-1]['output_dim'], 2*layer_specs[-1]['output_dim'])
        self.reconstruction_layer = nn.Linear(2*layer_specs[-1]['output_dim'], 2*layer_specs[-1]['output_dim'])

    def forward(self, ids, adj_tail, adj, feats):
        sage_output = self.sage(ids, adj_tail, feats)
        x = self.sage(ids, adj, feats)
        mu = self.mu_layer(sage_output)
        logvar = self.logvar_layer(sage_output)
        z = self.reparameterize(mu, logvar)
        adj_recon = self.decode(z)
        return x, adj_recon, mu, logvar, z

    def decode(self, z):
        recon_adj_logits = self.reconstruction_layer(z)
        return torch.sigmoid(recon_adj_logits)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss_function(self, adj_recon, adj_labels, mu, logvar):

        bce_loss = F.binary_cross_entropy(adj_recon, adj_labels)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce_loss + kl_loss


class GraphSAGE1(nn.Module):
    def __init__(self, aggregator_class, input_dim, layer_specs, device='cpu'):
        super(GraphSAGE1, self).__init__()
        self.sample_fns = [partial(uniform_neib_sampler, n_samples=s['n_sample'], device=device) for s in layer_specs]
        agg_layers = []
        for spec in layer_specs:
            agg = aggregator_class(input_dim=input_dim, output_dim=spec['output_dim'], activation=spec['activation'])
            agg_layers.append(agg)
            input_dim = agg.output_dim
        self.agg_layers = nn.Sequential(*agg_layers)

    def forward(self, ids, adj, feats):
        tmp_feats = feats[ids]
        all_feats = [tmp_feats]
        for _, sampler_fn in enumerate(self.sample_fns):
            ids = sampler_fn(adj=adj, ids=ids).contiguous().view(-1)
            tmp_feats = feats[ids]
            all_feats.append(tmp_feats)
        for agg_layer in self.agg_layers.children():
            all_feats = [agg_layer(all_feats[k], all_feats[k + 1]) for k in range(len(all_feats) - 1)]
        assert len(all_feats) == 1, "len(all_feats) != 1"
        out = all_feats[0]

        return out


class Cly_net(nn.Module):
    def __init__(self, ninput, noutput, layers, activation=nn.ReLU(), dropout=nn.Dropout()):
        super(Cly_net, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        layer_sizes = [ninput] + [int(x) for x in layers.split('-') if x != '']
        self.layers = []
        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(layer_sizes)), layer)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

        return x


class Disc(nn.Module):
    def __init__(self, ninput, layers, noutput=1):
        super(Disc, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.model = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(layer_sizes[2], layer_sizes[-1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[-1], noutput),
        )

    def forward(self, x, lambda_):
        x = GradientReversalFunction.apply(x, lambda_)
        x = self.model(x)

        return x


