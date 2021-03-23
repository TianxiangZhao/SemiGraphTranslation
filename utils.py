import argparse
import scipy.sparse as sp
from trainer import Trainer
import numpy as np
import torch
import ipdb
from scipy.io import loadmat
import networkx as nx
import multiprocessing as mp
import torch.nn.functional as F
from functools import partial

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--enable_feat', action='store_true', default=True)
    parser.add_argument('--enable_pos', action='store_true', default=True)
    parser.add_argument('--pos_pre', action='store_true', default=False)
    parser.add_argument('--decode_pre', action='store_true', default=False)
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--npos', type=int, default=8)
    parser.add_argument('--dataset', type=str, default='dblp_split')
    parser.add_argument('--setting', type=str, default='checkpoint')
    parser.add_argument('--size', type=int, default=100)

    if hasattr(Trainer, 'add_args'):
        Trainer.add_args(parser)

    return parser


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def norm_sparse(adj):#normalize a torch dense tensor for GCN, and change it into sparse.
    adj = adj + torch.eye(adj.shape[0]).to(adj)
    rowsum = torch.sum(adj,1)
    r_inv = 1/rowsum
    r_inv[torch.isinf(r_inv)] = 0.
    new_adj = torch.mul(r_inv.reshape(-1,1), adj)

    indices = torch.nonzero(new_adj).t()
    values = new_adj[indices[0], indices[1]] # modify this based on dimensionality

    return torch.sparse.FloatTensor(indices, values, new_adj.size())

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def compute_dist_to_node(node, G):

    dist_dict = dict(nx.single_target_shortest_path_length(G, node))
    for i in list(G.nodes):
        if i not in dist_dict:
            #set 999 as the maximum distance
            dist_dict[i] = 999

    dist_array = np.array([[k,v] for k, v in dist_dict.items()])
    dist = dist_array[dist_array[:,0].argsort(),1]

    return dist


def compute_dist_to_group(adj, anchor_chosen):
    """
    return a normalized dist array, node_num*anchor_num
    """
    assert anchor_chosen.shape[0]%4 ==0, "anchor number should be proportional to 4(number of workers)"
    #ipdb.set_trace()
    #change to a networkx object
    G = nx.Graph()
    G.add_nodes_from(np.arange(adj.shape[0]))

    coo_adj = adj.tocoo()
    edge_list = [(coo_adj.row[i], coo_adj.col[i]) for i in np.arange(coo_adj.nnz)]
    G.add_edges_from(edge_list)

    if not isinstance(anchor_chosen, list):
        anchor_chosen = anchor_chosen.tolist()

    #count the pair-wise distance towards the chosen anchors
    with mp.Pool(4) as p:
        result = p.map(partial(compute_dist_to_node, G=G), anchor_chosen)

    dist_embed = np.vstack(result).T

    dist_embed = 1.0/(1+dist_embed)

    return dist_embed

def calc_gradient_penalty(netD, src_data, real_data, fake_data):
    
    alpha = torch.rand(src_data.shape[0], 1).view(-1, 1) if src_data.dim()==2 else torch.rand(src_data.shape[0], 1).view(-1, 1, 1)
    #move to proper position

    alpha = alpha.to(src_data)

    interpolates = (alpha * real_data + ((1 - alpha) * fake_data))
    interpolates.requires_grad = True

    disc_interpolates = netD(src_data, interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=src_data.new(disc_interpolates.shape),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = (((gradients+0.001).norm(2, dim=1) - 1) ** 2).mean() * 0.01 #add 0.001 for preventing NAN. is is necesary?
    return gradient_penalty

def MI_estimation(paired_batch, discriminator):
    """
    paired_batch: [x, E_(x)].
    discriminator: MI estimator. input the pairs, output scores
    x is taken as fixed, here, only consider optimize E_().
    return the estimated MI score
    estimate it in a batch-wise manner
    """
    assert paired_batch[0].shape[0] >= 1

    #
    genuine_score = 0
    fake_score = 0

    for i in paired_batch[0].shape[0]:
        input_x = x[i].detach().expand(x.shape)
        
        #return a new tensor of scores
        result = discriminator(input_x, paired_batch[1])

        genuine_return = result[np.arange(result.shape[0])==i]
        fake_return = result[np.arange(result.shape[0])==i].mean()

        genuine_score = genuine_score - (F.softplus(-genuine_return)).mean()
        fake_score = fake_score + F.softplus(fake_return).mean()

    return genuine_score - fake_score

def adj_mse_loss(adj_rec, adj_tgt, adj_mask = None):

    neg_candid = adj_rec.new(adj_tgt.shape).fill_(0.0)
    neg_candid[adj_tgt==0] = 1.0

    if adj_mask is not None:
        valid_node_num = adj_mask.nonzero().shape[0]
        neg_candid = torch.mul(neg_candid, adj_mask)
    else:
        valid_node_num = adj_rec.numel()

    pos_number = adj_tgt.nonzero().shape[0]
    neg_number = valid_node_num - pos_number

    adj_rec = adj_rec.flatten()
    adj_tgt = adj_tgt.flatten()
    neg_candid_mask = neg_candid.flatten()
    if neg_number>pos_number*2:
        #masked_out = torch.multinomial(neg_candid_mask, neg_number//2)
        masked_out = neg_candid_mask.nonzero()[pos_number:,:].flatten()
        neg_candid_mask[masked_out] = 0.0
        neg_number = neg_candid_mask.nonzero().shape[0]

    total_mask = neg_candid_mask[adj_tgt!=0] = 1.0

    loss = F.mse_loss(torch.mul(adj_rec, total_mask), adj_tgt)

    return loss

