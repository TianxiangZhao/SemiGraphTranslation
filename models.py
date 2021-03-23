import torch.nn as nn
import torch.nn.functional as F
import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import init
import ipdb

### layers###
#GCN layer
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        #for 3_D batch, need a loop!!!


        if self.bias is not None:
            return output + self.bias
        else:
            return output

#Multihead attention layer
class MultiHead(Module):#currently, allowed for only one sample each time. As no padding mask is required.
    def __init__(
        self,
        input_dim,
        num_heads,
        kdim=None,
        vdim=None,
        embed_dim = 128,#should equal num_heads*head dim
        v_embed_dim = None,
        dropout=0.1,
        bias=True,
    ):
        super(MultiHead, self).__init__()
        self.input_dim = input_dim
        self.kdim = kdim if kdim is not None else input_dim
        self.vdim = vdim if vdim is not None else input_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.v_embed_dim = v_embed_dim if v_embed_dim is not None else embed_dim

        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.bias = bias
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        assert self.v_embed_dim % num_heads ==0, "v_embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5


        self.q_proj = nn.Linear(self.input_dim, self.embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, self.embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, self.v_embed_dim, bias=bias)

        self.out_proj = nn.Linear(self.v_embed_dim, self.v_embed_dim//self.num_heads, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if True:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.normal_(self.k_proj.weight)
            nn.init.normal_(self.v_proj.weight)
            nn.init.normal_(self.q_proj.weight)
        else:
            nn.init.normal_(self.k_proj.weight)
            nn.init.normal_(self.v_proj.weight)
            nn.init.normal_(self.q_proj.weight)

        nn.init.normal_(self.out_proj.weight)

        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

        if self.bias:
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.q_proj.bias, 0.)

    def forward(
        self,
        query,
        key,
        value,
        need_weights: bool = False,
        need_head_weights: bool = False,
    ):
        """Input shape: Time x Batch x Channel
        Args:
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        batch_num, node_num, input_dim = query.size()

        assert key is not None and value is not None

        #project input
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q * self.scaling

        #compute attention
        q = q.view(batch_num, node_num, self.num_heads, self.head_dim).transpose(-2,-3).contiguous().view(batch_num*self.num_heads, node_num, self.head_dim)
        k = k.view(batch_num, node_num, self.num_heads, self.head_dim).transpose(-2,-3).contiguous().view(batch_num*self.num_heads, node_num, self.head_dim)
        v = v.view(batch_num, node_num, self.num_heads, self.vdim).transpose(-2,-3).contiguous().view(batch_num*self.num_heads, node_num, self.vdim)
        attn_output_weights = torch.bmm(q, k.transpose(-1,-2))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        #drop out
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        #collect output
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.view(batch_num, self.num_heads, node_num, self.vdim).transpose(-2,-3).contiguous().view(batch_num, node_num, self.v_embed_dim)
        attn_output = self.out_proj(attn_output)


        if need_weights:
            attn_output_weights = attn_output_weights #view: (batch_num, num_heads, node_num, node_num)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output


#Graphsage layer
class SageConv(Module):
    """
    Simple Graphsage layer
    """

    def __init__(self, in_features, out_features, bias=False):
        super(SageConv, self).__init__()

        self.proj = nn.Linear(in_features*2, out_features, bias=bias)

        self.reset_parameters()

        print("note: for dense graph in graphsage, require it normalized.")

    def reset_parameters(self):

        nn.init.normal_(self.proj.weight)

        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)

    def forward(self, adj, features):
        """
        Args:
            adj: can be sparse or dense matrix.
        """

        #fuse info from neighbors. to be added:
        if not isinstance(adj, torch.sparse.FloatTensor):
            neigh_feature = torch.bmm(adj, features) / (adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1],-1))+1)
        else:
            print("spmm not implemented for batch training. Note!")
            neigh_feature = torch.spmm(adj, features)

        #perform conv
        data = torch.cat([features,neigh_feature], dim=-1)
        combined = self.proj(data)

        return combined


###models###

#GraphSage Like encoder
class Encoder(Module):
    """
    Simple Graphsage-like encoder
    """

    def __init__(self, nfeat, nhid, nembedding, dropout=0.1, 
        layer_num = 2, 
        feature_pre = True,
        feature_dim = None,
        jump = False):

        super(Encoder, self).__init__()
        self.layer_num = layer_num
        self.dropout = dropout
        self.feature_pre = feature_pre

        if feature_pre:
            feature_dim = nfeat//4
            self.linear_pre = nn.Linear(nfeat, feature_dim)
            nn.init.normal_(self.linear_pre.weight)
            self.conv_first = SageConv(feature_dim, nhid)
        else:
            self.conv_first = SageConv(nfeat, nhid)
        self.conv_hidden = nn.ModuleList([SageConv(nhid+nfeat, nhid) for i in range(layer_num - 2)])
        self.conv_out = SageConv(nhid+nfeat, nembedding)
        

    def forward(self, adj, x):
        """
        Args:
            adj: can be sparse or dense matrix.
        """
        feat = x
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(adj, x)
        x = F.relu(x)

        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = torch.cat([x,feat], dim=-1)
            x = self.conv_hidden[i](adj, x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)

        x = torch.cat([x,feat], dim=-1)
        x = self.conv_out(adj, x)

        return x


#attr/attr+pos decoder
class Decoder(Module):
    """
    Simple Graphsage layer
    """

    def __init__(self, npos, nhid, nfeat, args, dropout=0.1, need_x = True):
        super(Decoder, self).__init__()
        self.dropout = dropout

        self.MulHead_pos = MultiHead(npos, 8, embed_dim=8*npos, dropout = self.dropout)
        self.args = args
        self.decode_pre = args.decode_pre
        self.need_x = need_x

        n_combine = npos + nhid

        if need_x:
            self.MulHead_x = MultiHead(npos, 8, vdim = nhid, v_embed_dim = 8*nhid , dropout = self.dropout)

            
            self.MLP_x_1 = nn.Linear(n_combine, n_combine*2)
            self.MLP_x_2 = nn.Linear(n_combine*2, nfeat)


        self.de_weight = Parameter(torch.FloatTensor(n_combine, n_combine))

        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)

        nn.init.normal_(self.MLP_x_1.weight)
        nn.init.normal_(self.MLP_x_2.weight)

    def forward(self, x_embed, pos_embed):
        if self.decode_pre:
            pos = self.MulHead_pos(pos_embed, pos_embed, pos_embed)

            if x_embed is not None and self.need_x:
                x = self.MulHead_x(pos_embed, pos_embed, x_embed)
                combine = torch.cat([x,pos], dim=-1)
                x_out = self.MLP_x_2(F.relu(self.MLP_x_1(combine)))
            else:
                combine = pos
                x_out = None
        
        elif x_embed is not None and self.need_x:
            #x = self.MulHead_x(pos_embed, pos_embed, x_embed)
            combine = torch.cat([x_embed,pos_embed], dim=-1)
            x_out = self.MLP_x_2(F.relu(self.MLP_x_1(combine)))

            #if self.args == 'checkpoint_nopos':
            #    combine = x_embed
                
        else:
            combine = torch.cat([x_embed,pos_embed], dim=-1)
            x_out = None

        #'''
        #predict adj matrix
        #adj_out = F.tanh(torch.bmm(combine, combine.transpose(-1,-2)))
        combine = F.linear(combine, self.de_weight)
        adj_out = torch.sigmoid(torch.bmm(combine, combine.transpose(-1,-2)))

        return adj_out,x_out

        

#global readout
class ReadOut(Module):
    """
    Simple Graphsage layer
    """

    def __init__(self, ):
        super(ReadOut, self).__init__()


    def forward(self, X):


        return X.sum(dim=1)


#Translator
class Translator(Module):
    """
    Simple Graphsage layer
    """

    def __init__(self, nhid, nfeat, bias=False):
        super(Translator, self).__init__()

        self.MLP_1 = nn.Linear(nhid, nhid*3, bias)
        self.MLP_2 = nn.Linear(nhid*3, nfeat, bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.MLP_1.weight)
        nn.init.normal_(self.MLP_2.weight)

    def forward(self, node_feat, glob_feat):
        combined = torch.cat([node_feat,glob_feat.view(glob_feat.shape[0], 1, glob_feat.shape[-1]).expand(glob_feat.shape[0],node_feat.shape[1],-1)], dim=-1)

        out = self.MLP_2(F.relu(self.MLP_1(combined)))


        return out

#discriminator based on global feature
class Glob_Discriminator(Module):
    """
    a simple implementation of discriminator based on the global Readout vector
    """

    def __init__(self, nhid, bias=True):
        super(Glob_Discriminator, self).__init__()

        self.MLP_1 = nn.Linear(nhid+nhid, nhid, bias)
        self.MLP_2 = nn.Linear(nhid, 2, bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.MLP_1.weight)
        nn.init.normal_(self.MLP_2.weight)

    def forward(self, glob_feat_a, glob_feat_b, T=5):
        #ipdb.set_trace()
        out = self.MLP_2(F.relu(self.MLP_1(torch.cat([glob_feat_a, glob_feat_b], dim=-1)))) # get the probability of being real
        #out = F.softmax(self.MLP_2(F.relu(self.MLP_1(torch.cat([glob_feat_a, glob_feat_b], dim=-1))))/T)[:,0] # get the probability of being real

        return out

#MI estimator based on paired global feature

class Glob_MINE(Module):
    """
    a simple implementation of Mutual Information Neural Estimation based on the global Readout vector
    """

    def __init__(self, nhid, bias=True):
        super(Glob_MINE, self).__init__()

        self.MLP_1 = nn.Linear(nhid+nhid, nhid, bias)
        self.MLP_2 = nn.Linear(nhid, 2, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.MLP_1.weight)
        nn.init.normal_(self.MLP_2.weight)

    def forward(self, glob_feat_a, glob_feat_b):
        out = self.MLP_2(F.relu(self.MLP_1(torch.cat([glob_feat_a, glob_feat_b], dim=-1))))

        return out



