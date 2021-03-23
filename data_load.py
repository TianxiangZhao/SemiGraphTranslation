import scipy.sparse as sp
import numpy as np
import torch
import ipdb
from scipy.io import loadmat
import os
import utils
import pickle
import ipdb

FFTWTY_NODES = 6408 

def load_paired_data(path='./dataset/FFTWTY', dataset="FFTWTY", ):#modified from code: pygcn
    print('Loading {} dataset...'.format(dataset))

    #if has features, load them
    if os.path.exists(os.path.join(path,'idx_features_labels_new')):
        idx_features_labels = np.genfromtxt(os.path.join(path,'idx_features_labels'),
                                        dtype=np.dtype(str))
        
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

        labels = idx_features_labels[:, -1]
        set_labels = set(labels)
        classes_dict = {c: np.arange(len(set_labels))[i] for i, c in enumerate(set_labels)}

        labels = np.array(list(map(classes_dict.get, labels)))

        idx = np.array(idx_features_labels[:, 0], dtype=np.float32).astype(int)
        idx_map = {j: i for i, j in enumerate(idx)}
        features = np.array(features.todense())

    else:#assume already add the id. self-mapping
        idx = np.array(range(1,FFTWTY_NODES),dtype=np.float32).astype(int)
        idx_map = {j:i for i,j in enumerate(idx)}
        features = None

    # build idx map

    #build undirected source graph
    edges_view0 = np.genfromtxt(os.path.join(path,'view_0'),
                                    dtype=np.float32).astype(int)
    edges_view0 = edges_view0[:,:2]

    for i in edges_view0.flatten():
        if i not in idx_map:
            ipdb.set_trace()


    edges_0 = np.array(list(map(idx_map.get, edges_view0.flatten())),
                     dtype=np.float32).astype(int).reshape(edges_view0.shape)

    adj = sp.coo_matrix((np.ones(edges_0.shape[0]), (edges_0[:, 0], edges_0[:, 1])),
                        shape=(idx.shape[0], idx.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj_src = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    #build undirected tgt graph
    edges_view1 = np.genfromtxt(os.path.join(path,'view_1'), dtype = np.float32).astype(int)
    edges_view1 = edges_view1[:,:2]
    edges_1 = np.array(list(map(idx_map.get, edges_view1.flatten())), 
                    dtype=np.float32).astype(int).reshape(edges_view1.shape)

    adj_tgt = sp.coo_matrix((np.ones(edges_1.shape[0]), (edges_1[:,0], edges_1[:,1])), 
                        shape=(idx.shape[0], idx.shape[0]), 
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj_tgt = adj_tgt + adj_tgt.T.multiply(adj_tgt.T > adj_tgt) - adj_tgt.multiply(adj_tgt.T > adj_tgt)


    '''#test the code of subgraph obtaining
    ipdb.set_trace()
    chosen_set = set([3])

    for s in range(2):
        enlarge_set = set.union(*[set(adj[node].nonzero()[1]) for node in chosen_set])
        chosen_set = set.union(chosen_set, enlarge_set)
    '''
    #filter out non-connected node
    adj_src_dense = np.clip(adj_src.todense(), 0.0,1.0)
    np.fill_diagonal(adj_src_dense, 0.0)
    idx_chosen = np.array(adj_src_dense.sum(axis=1)).nonzero()[0]
    adj_src_dense = adj_src_dense[idx_chosen, :][:, idx_chosen]
    adj_src = sp.coo_matrix(adj_src_dense)

    adj_tgt_dense = np.clip(adj_tgt.todense(), 0.0,1.0)
    np.fill_diagonal(adj_tgt_dense, 0.0)
    adj_tgt_dense = adj_tgt_dense[idx_chosen, :][:, idx_chosen]
    adj_tgt = sp.coo_matrix(adj_tgt_dense)

    if features is not None:
        features = features[idx_chosen, :]

    adj_src = adj_src.tocsr()
    adj_tgt = adj_tgt.tocsr()

    return adj_src, adj_tgt, features

def load_auth_data(scene = 'train', size = 300, user = None):
    if scene == 'train':
        path = './dataset/authentic/'+str(size)+'/train/'
        adj_src = np.load(path+'reg_data.npy')
        adj_tgt = np.load(path+'mal_data.npy')
        features = None



    return adj_src, adj_tgt, features

def load_traffic_data(path='./dataset/traffic', file='traffic_graph_2015.pkl'):
    data = pickle.load(open(os.path.join(path,file), 'rb'))
    valid_num = len(data.keys())
    valid_idx = np.array(range(valid_num))



    return data, valid_idx


    
#test
if __name__ == '__main__':
    #ipdb.set_trace()
    #adj, adj_tgt, features = load_paired_data()

    #print(adj.shape)
    #print(adj_tgt.shape)
    ipdb.set_trace()

    adj, adj_tgt, features = load_paired_data('./dataset/dblp','dblp')
