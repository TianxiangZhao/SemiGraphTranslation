import numpy as np 
import utils
import ipdb
import scipy.sparse as sp
import random
import time
import copy
import torch
import torch.nn.functional as F
import pickle
from data_load import load_paired_data
import argparse
import os


def uniform_sub_graph(adj, chosen_ind, batch_node_num, node_attr=None, require_valid_ind=False, npos=8):
    cur_node_num = chosen_ind.shape[0]
    add_num = 0

    last_valid_ind = batch_node_num
    if cur_node_num > batch_node_num:
        chosen_ind = np.random.choice(chosen_ind, batch_node_num, replace=False)

    elif cur_node_num < batch_node_num:
        add_num = batch_node_num - cur_node_num
        last_valid_ind = cur_node_num

    adj_extracted = adj[chosen_ind,:].tocsc()[:,chosen_ind]

    valid_adj = adj_extracted#used for getting position embedding
    valid_ind = chosen_ind

    if node_attr is not None:
        node_extracted = node_attr[chosen_ind,:]
        if add_num !=0:
            add_feat = np.zeros((add_num, node_extracted.shape[1]))
            node_extracted = np.concatenate((node_extracted, add_feat), axis=0)
    else:
        node_extracted = None


    pos_extracted = get_pos_with_anchor(valid_adj, anchor_number=npos)
    if add_num !=0:
        add_pos = np.zeros((add_num, pos_extracted.shape[1]))
        pos_extracted = np.concatenate((pos_extracted, add_pos), axis=0)
        
    if add_num != 0:
        adj_extracted = np.zeros((batch_node_num, batch_node_num))
        adj_extracted[:cur_node_num, :cur_node_num] = np.array(valid_adj.todense())
    else:
        adj_extracted = np.array(adj_extracted.todense())

    adj_extracted = np.clip(adj_extracted, 0.0,1.0)


    if require_valid_ind:
        return adj_extracted, node_extracted, pos_extracted, last_valid_ind, valid_ind
    else:
        return adj_extracted, node_extracted, pos_extracted, last_valid_ind    

def get_pos_with_anchor(adj, adj_tgt=None, anchor_number = 8):
    """
    sampling some anchor nodes, use relative closeness to them to generate 
        the position embedding vector for each node
    for paired data, all use the same anchor sets to initialize. 

    """

    assert isinstance(adj, sp.csr_matrix) or isinstance(adj, sp.csc_matrix), "cannot obtain subgraph, wrong data type. expecting csr/csr matrix"

    anchor_chosen = np.random.choice(adj.shape[0], anchor_number, replace=True)

    dist_src = utils.compute_dist_to_group(adj, anchor_chosen)

    if adj_tgt is not None:
        dist_tgt = utils.compute_dist_to_group(adj_tgt, anchor_chosen)
        return dist_src, dist_tgt

    else:
        return dist_src

class GraphDataset(object):
    def __init__(self, args):
        self.args = args
        self.cur_ind = 0

    def return_full_index(self):

        raise NotImplementedError

    def get_sample(self, index):

        raise NotImplementedError

    def get_random_sample(self, valid_index=None, num=1):

        raise NotImplementedError

    def get_ordered_sample(self, num=1):

        raise NotImplementedError


class PairedTransData(GraphDataset):
    """docstring for PairedTransGraph"""
    def __init__(self, args, adj_src, adj_tgt, feature=None):
        '''
        require adj in: csr matirx
        require feature in: np.array
        '''
        super(PairedTransData, self).__init__(args)
        assert isinstance(adj_tgt, sp.csr_matrix), "cannot obtain subgraph, wrong data type. expecting csr matrix"
        assert isinstance(adj_src, sp.csr_matrix), "cannot obtain subgraph, wrong data type. expecting csr matrix"

        self.adj_src = adj_src
        self.adj_tgt = adj_tgt
        self.feature = feature

        self.tot_num = adj_src.shape[0]
        self.cur_ind = 0


    def return_full_index(self):

        return np.arange(self.adj_src.shape[0])


    def obtain_sub_graph_ind(self, valid_ind=None, step=2, given_ind = None, is_src=True):
        if is_src:
            adj = self.adj_src
        else:
            adj = self.adj_tgt

        if given_ind is not None:
            center_ind = given_ind
        elif valid_ind is None:
            center_ind = np.random.randint(0,adj.shape[0])
        else:
            center_ind = np.random.choice(valid_ind, 1)[0]


        chosen_set = set([center_ind])

        for s in range(step):
            #enlarge_set = set.union(*[set(adj_src[node].nonzero()[1]) for node in chosen_set])

            #add a neighbor sampler. need to make sure they should be within a range
            enlarge_set = set()
            for node in chosen_set:
                new_set = set(adj[node].nonzero()[1])
                if len(new_set) > 15 and len(new_set) <=30:
                    new_set = set(random.sample(new_set, len(new_set)//2))
                elif len(new_set) > 30:
                    new_set = set(random.sample(new_set, len(new_set)//3))

                enlarge_set = set.union(enlarge_set, new_set)

            chosen_set = set.union(chosen_set, enlarge_set)

        chosen_ind = np.array(list(chosen_set))

        return chosen_ind

    def get_random_sample(self, valid_index=None, num=1, is_pair=True):
        #for source graph:
        extracted_ind = []
        extracted_adj_src = []
        extracted_node_src = []
        extracted_pos_src = []
        extracted_valid_ind = []

        batch_node_num = 0

        for i in range(num):

            #sample a center node
            chosen_ind = self.obtain_sub_graph_ind(valid_index)

            batch_node_num = batch_node_num + chosen_ind.shape[0]
            extracted_ind.append(chosen_ind)

        batch_node_num = batch_node_num // num


        #adjust the graph size, form a batch
        for i in range(num):
            chosen_ind = extracted_ind[i]

            adj_extracted, node_extracted, pos_extracted, last_valid_ind, valid_ind = uniform_sub_graph(self.adj_src, chosen_ind, batch_node_num ,self.feature, require_valid_ind = True)
            extracted_ind[i] = valid_ind

            extracted_adj_src.append(adj_extracted)
            extracted_node_src.append(node_extracted)
            extracted_pos_src.append(pos_extracted)
            extracted_valid_ind.append(last_valid_ind)

        #change into torch tensor
        if self.args.cuda:
            extracted_adj_src = torch.FloatTensor(extracted_adj_src).cuda()
            if self.args.enable_feat:
                extracted_node_src = torch.FloatTensor(extracted_node_src).cuda()
            else:
                extracted_node_src = None
            extracted_pos_src = torch.FloatTensor(extracted_pos_src).cuda()
            extracted_valid_ind_src = torch.IntTensor(extracted_valid_ind).cuda()
        else:
            extracted_adj_src = torch.FloatTensor(extracted_adj_src)
            if self.args.enable_feat:
                extracted_node_src = torch.FloatTensor(extracted_node_src)
            else:
                extracted_node_src = None
            extracted_pos_src = torch.FloatTensor(extracted_pos_src)
            extracted_valid_ind_src = torch.IntTensor(extracted_valid_ind)

        #get the tgt subgraph
        extracted_adj_tgt = []
        extracted_node_tgt = []
        extracted_pos_tgt = []

        if is_pair is False:
            extracted_valid_ind = []
            extracted_ind = []
            batch_node_num = 0

            for i in range(num):
                #sample a center node
                chosen_ind = self.obtain_sub_graph_ind(valid_index, is_src=False)
                extracted_ind.append(chosen_ind)
                batch_node_num = batch_node_num + chosen_ind.shape[0]

            batch_node_num = batch_node_num // number

        for i in range(num):
            chosen_ind = extracted_ind[i]

            adj_extracted, node_extracted, pos_extracted, last_valid_ind = uniform_sub_graph(self.adj_tgt, chosen_ind, batch_node_num ,self.feature)

            extracted_adj_tgt.append(adj_extracted)
            extracted_node_tgt.append(node_extracted)
            extracted_pos_tgt.append(pos_extracted)
            assert extracted_valid_ind[i] == last_valid_ind, "valid_ind_number in paired sample must be the same"

        #change into torch tensor
        if self.args.cuda:
            extracted_adj_tgt = torch.FloatTensor(extracted_adj_tgt).cuda()
            if self.args.enable_feat:
                extracted_node_tgt = torch.FloatTensor(extracted_node_tgt).cuda()
            else:
                extracted_node_tgt = None
            extracted_pos_tgt = torch.FloatTensor(extracted_pos_tgt).cuda()
            extracted_valid_ind_tgt = torch.IntTensor(extracted_valid_ind).cuda()
        else:
            extracted_adj_tgt = torch.FloatTensor(extracted_adj_tgt)
            if self.args.enable_feat:
                extracted_node_tgt = torch.FloatTensor(extracted_node_tgt)
            else:
                extracted_node_tgt = None
            extracted_pos_tgt = torch.FloatTensor(extracted_pos_tgt)
            extracted_valid_ind_tgt = torch.IntTensor(extracted_valid_ind)
        


        return extracted_adj_src, extracted_node_src, extracted_pos_src, extracted_adj_tgt, extracted_node_tgt, extracted_pos_tgt, extracted_valid_ind_src, extracted_valid_ind_tgt


    def get_ordered_sample(self, valid_index=None, num=1, preset_size=None):
        #########################################
        #used for NEC DGT. 
        #need to be modified if want to be used by Dual-Embedding.
        #########################################

        if valid_index is not None:
            tot_num = len(valid_index)
        else:
            tot_num = self.tot_num

        if self.cur_ind+num >= tot_num:
            self.cur_ind = 0
            print("finish obtain one epoch of data")

        #for source graph:
        extracted_ind = []
        extracted_adj_src = []
        extracted_node_src = []
        extracted_pos_src = []
        extracted_adj_tgt = []
        extracted_node_tgt = []
        extracted_pos_tgt = []
        extracted_valid_ind = []

        batch_node_num = 0

        for i in range(num):

            #sample a center node
            chosen_ind = self.obtain_sub_graph_ind(valid_index, given_ind = self.cur_ind+i)

            batch_node_num = batch_node_num + chosen_ind.shape[0]
            extracted_ind.append(chosen_ind)

        batch_node_num = batch_node_num // num

        if preset_size is not None:
            batch_node_num = preset_size

        #adjust the graph size, form a batch
        for i in range(num):
            chosen_ind = extracted_ind[i]

            adj_extracted, node_extracted, pos_extracted, last_valid_ind, valid_ind = uniform_sub_graph(self.adj_src, chosen_ind, batch_node_num ,self.feature, require_valid_ind = True)
            extracted_ind[i] = valid_ind

            extracted_adj_src.append(adj_extracted)
            extracted_node_src.append(node_extracted)
            extracted_pos_src.append(pos_extracted)
            extracted_valid_ind.append(last_valid_ind)

        #change into torch tensor
        #ipdb.set_trace()

        '''
        if self.args.cuda:
            extracted_adj_src = torch.FloatTensor(extracted_adj_src).cuda()
            if self.args.enable_feat:
                extracted_node_src = torch.FloatTensor(extracted_node_src).cuda()
            else:
                extracted_node_src = None
            extracted_pos_src = torch.FloatTensor(extracted_pos_src).cuda()
            extracted_valid_ind_src = torch.IntTensor(extracted_valid_ind).cuda()
        else:
            extracted_adj_src = torch.FloatTensor(extracted_adj_src)
            if self.args.enable_feat:
                extracted_node_src = torch.FloatTensor(extracted_node_src)
            else:
                extracted_node_src = None
            extracted_pos_src = torch.FloatTensor(extracted_pos_src)
            extracted_valid_ind_src = torch.IntTensor(extracted_valid_ind)
        '''

        #get the tgt subgraph
        for i in range(num):
            chosen_ind = extracted_ind[i]

            adj_extracted, node_extracted, pos_extracted, last_valid_ind = uniform_sub_graph(self.adj_tgt, chosen_ind, batch_node_num ,self.feature)

            extracted_adj_tgt.append(adj_extracted)
            extracted_node_tgt.append(node_extracted)
            extracted_pos_tgt.append(pos_extracted)
            assert extracted_valid_ind[i] == last_valid_ind, "valid_ind_number in paired sample must be the same"


        #change into torch tensor

        '''
        if self.args.cuda:
            extracted_adj_tgt = torch.FloatTensor(extracted_adj_tgt).cuda()
            if self.args.enable_feat:
                extracted_node_tgt = torch.FloatTensor(extracted_node_tgt).cuda()
            else:
                extracted_node_tgt = None
            extracted_pos_tgt = torch.FloatTensor(extracted_pos_tgt).cuda()
            extracted_valid_ind_tgt = torch.IntTensor(extracted_valid_ind).cuda()
        else:
            extracted_adj_tgt = torch.FloatTensor(extracted_adj_tgt)
            if self.args.enable_feat:
                extracted_node_tgt = torch.FloatTensor(extracted_node_tgt)
            else:
                extracted_node_tgt = None
            extracted_pos_tgt = torch.FloatTensor(extracted_pos_tgt)
            extracted_valid_ind_tgt = torch.IntTensor(extracted_valid_ind)
        '''
        extracted_adj_src = np.array(extracted_adj_src)
        extracted_adj_tgt = np.array(extracted_adj_tgt)
        extracted_node_src = np.array(extracted_node_src)



        self.cur_ind = self.cur_ind + num

        #initialize edge info
        edge_src = np.zeros((num, self.args.Dr, self.args.Nr))
        edge_tgt = np.zeros((num, self.args.Dr, self.args.Nr))

        Rr = np.zeros((num, self.args.No, self.args.Nr))
        Rs = np.zeros((num, self.args.No, self.args.Nr))

        for src in range(self.args.No):
            Rr[:,src,src*(self.args.No-1):(src+1)*(self.args.No-1)] = 1
            for tgt in range(self.args.No-1):
                temp = tgt if tgt<src else tgt+1
                Rs[:, temp, src*(self.args.No-1)+tgt]=1


        for g in range(num):
            edge_s = extracted_adj_src[g].nonzero()[0]
            edge_t = extracted_adj_src[g].nonzero()[1]

            for i in range(edge_s.shape[0]):
                src = edge_s[i]
                tgt = edge_t[i]

                y = tgt if tgt<src else tgt-1

                edge_src[g,:, src*(self.args.No-1)+y] = 1

            edge_s = extracted_adj_tgt[g].nonzero()[0]
            edge_t = extracted_adj_tgt[g].nonzero()[1]

            for i in range(edge_t.shape[0]):
                src = edge_s[i]
                tgt = edge_t[i]

                y = tgt if tgt<src else tgt-1

                edge_tgt[g,:, src*(self.args.No-1)+y] = 1

        #initialize node info
        #ipdb.set_trace()
        X_data = extracted_node_src.transpose(0,2,1)

        #initialize not used info
        node_src = np.zeros((num, self.args.Ds, self.args.No))
        node_tgt = np.zeros((num, self.args.Ds, self.args.No))


        return node_src, node_tgt, edge_src, edge_tgt, Rr, Rs, X_data


    def get_tosave_batch(self, valid_index=None, num=1, preset_size=None):
        if valid_index is not None:
            tot_num = len(valid_index)
        else:
            tot_num = self.tot_num

        if self.cur_ind+num >= tot_num:
            self.cur_ind = 0
            print("finish obtain one epoch of data")

        #for source graph:
        extracted_ind = []
        extracted_adj_src = []
        extracted_node_src = []
        extracted_pos_src = []
        extracted_adj_tgt = []
        extracted_node_tgt = []
        extracted_pos_tgt = []
        extracted_valid_ind = []

        batch_node_num = 0

        for i in range(num):

            #sample a center node
            chosen_ind = self.obtain_sub_graph_ind(valid_index, given_ind = self.cur_ind+i)

            batch_node_num = batch_node_num + chosen_ind.shape[0]
            extracted_ind.append(chosen_ind)

        batch_node_num = batch_node_num // num

        if preset_size is not None:
            batch_node_num = preset_size

        #adjust the graph size, form a batch
        for i in range(num):
            chosen_ind = extracted_ind[i]

            adj_extracted, node_extracted, pos_extracted, last_valid_ind, valid_ind = uniform_sub_graph(self.adj_src, chosen_ind, batch_node_num ,self.feature, require_valid_ind = True)
            extracted_ind[i] = valid_ind

            extracted_adj_src.append(adj_extracted)
            extracted_node_src.append(node_extracted)
            extracted_pos_src.append(pos_extracted)
            extracted_valid_ind.append(last_valid_ind)


        #get the tgt subgraph
        for i in range(num):
            chosen_ind = extracted_ind[i]

            adj_extracted, node_extracted, pos_extracted, last_valid_ind = uniform_sub_graph(self.adj_tgt, chosen_ind, batch_node_num ,self.feature)

            extracted_adj_tgt.append(adj_extracted)
            extracted_node_tgt.append(node_extracted)
            extracted_pos_tgt.append(pos_extracted)
            assert extracted_valid_ind[i] == last_valid_ind, "valid_ind_number in paired sample must be the same"


        extracted_adj_src = np.array(extracted_adj_src)
        extracted_adj_tgt = np.array(extracted_adj_tgt)
        extracted_node_src = np.array(extracted_node_src)
        extracted_pos_src = np.array(extracted_pos_src)
        extracted_pos_tgt = np.array(extracted_pos_tgt)
        extracted_valid_ind = np.array(extracted_valid_ind)



        self.cur_ind = self.cur_ind + num

        return extracted_node_src, extracted_adj_src, extracted_adj_tgt, extracted_pos_src, extracted_pos_tgt, extracted_valid_ind


    def save_batched_pickle(self, batch_id, node_src, adj_src, adj_tgt, pose_src, pose_tgt, valid_ind):
        ################################
        #used for preprocessing dataset
        #data form: a standard form. When load, need to be processed.
        #node_src: batchsize*node*embedding_size
        #adj_src/tgt: batchsize*node*node
        ################################



        #save 
        for index in range(node_src.shape[0]):
            data = {}
            data['node'] = node_src[index]
            data['adj_src'] = adj_src[index]
            data['adj_tgt'] = adj_tgt[index]
            data['pose_src'] = pose_src[index]
            data['pose_tgt'] = pose_tgt[index]
            data['valid_ind'] = valid_ind[index]

            number = batch_id*node_src.shape[0]+index
            address = './dataset/dblp_split/'+str(adj_src.shape[1])+'/'+str(number)+'.p'
            print('save '+address)
            pickle.dump(data, open(address, "wb"))


    def load_batched_pickle(self, valid_index=None, num=1, preset_size=None):
        if valid_index is not None:
            tot_num = len(valid_index)
        else:
            tot_num = self.tot_num

        if self.cur_ind+num >= tot_num:
            self.cur_ind = 0
            print("finish obtain one epoch of data")

        #for source graph:
        extracted_ind = []
        extracted_adj_src = []
        extracted_node_src = []
        extracted_pos_src = []
        extracted_adj_tgt = []
        extracted_node_tgt = []
        extracted_pos_tgt = []
        extracted_valid_ind = []

        for i in range(num):

            #sample a center node
            addr = './dataset/dblp_split/'+str(preset_size)+'/'+str(self.cur_ind+i)+'.p'
            data = pickle.load(open(addr, "rb"))

            

            extracted_adj_src.append(data['adj_src'])
            extracted_node_src.append(data['node'])
            extracted_pos_src.append(data[pose_src])
            extracted_valid_ind.append(data['valid_ind'])

            extracted_adj_tgt.append(data['adj_tgt'])
            extracted_node_tgt.append(data['node'])
            extracted_pos_tgt.append(data['pose_tgt'])


        extracted_adj_src = np.array(extracted_adj_src)
        extracted_adj_tgt = np.array(extracted_adj_tgt)
        extracted_node_src = np.array(extracted_node_src)
        extracted_pos_src = np.array(extracted_pos_src)
        extracted_pos_tgt = np.array(extracted_pos_tgt)
        extracted_valid_ind = np.array(extracted_valid_ind)



        self.cur_ind = self.cur_ind + num


        return extracted_node_src, extracted_adj_src, extracted_adj_tgt, extracted_pos_src, extracted_pos_tgt, extracted_valid_ind
        


class RawCsvDataset(GraphDataset):
    """docstring for PairedTransGraph"""
    def __init__(self, args):
        '''
        require adj in: csr matirx
        require feature in: np.array
        '''
        super(RawCsvDataset, self).__init__(args)
        self.args = args

        self.node_attr = []
        self.adj_src = []
        self.adj_tgt = []
        self.pose_src = []

        if args.dataset == 'BA':
            self.load_data('./dataset/BA_40')
        elif args.dataset == 'ER':
            self.load_data('./dataset/ER_40')
        else:
            ipdb.set_trace()

        print("data loaded successfully!")
        self.tot_num = len(self.node_attr)
        self.cur_ind = 0

    def load_data(self, csv_folder):

        #
        file_count = 0
        for filename in os.listdir(csv_folder):
            if filename.endswith(".csv"):
                file_count = file_count+1

        temp = 0
        for i in range(file_count):
            address = csv_folder+'/'+self.args.dataset+'-40-input-'+str(temp)+'.csv'
            while os.path.isfile(address)==False:
                temp = temp+1
                address = csv_folder+'/'+self.args.dataset+'-40-input-'+str(temp)+'.csv'

            new_data = np.genfromtxt(address, delimiter=',')
            np.fill_diagonal(new_data, 1.0)

            new_adj_src = np.clip(new_data, 0.0, 1.0)
            new_adj_tgt = np.matmul(new_adj_src, new_adj_src.T)
            new_adj_tgt = np.clip(new_adj_tgt, 0.0, 1.0)

            new_node_attr = copy.deepcopy(new_adj_src)
            new_adj_pos = np.eye(new_adj_src.shape[0])

            self.node_attr.append(new_node_attr)
            self.adj_src.append(new_adj_src)
            self.adj_tgt.append(new_adj_tgt)
            self.pose_src.append(new_adj_pos)

        return



    def return_full_index(self):

        return np.arange(self.tot_num)


    def get_ordered_sample(self, valid_index=None, num=1, preset_size=None):
        if valid_index is not None:
            tot_num = len(valid_index)
        else:
            tot_num = self.tot_num

        if self.cur_ind+num >= tot_num:
            self.cur_ind = 0
            print("finish obtain one epoch of data")

        #for source graph:
        extracted_adj_src = []
        extracted_node_src = []
        extracted_pos_src = []
        extracted_adj_tgt = []
        extracted_node_tgt = []
        extracted_pos_tgt = []
        extracted_valid_ind = []


        #adjust the graph size, form a batch
        for i in range(num):

            extracted_adj_src.append(self.adj_src[valid_index[self.cur_ind+i]])
            extracted_node_src.append(self.node_attr[valid_index[self.cur_ind+i]])
            extracted_pos_src.append(self.pose_src[valid_index[self.cur_ind+i]])
            extracted_valid_ind.append(self.adj_src[self.cur_ind+i].shape[0])

            extracted_adj_tgt.append(self.adj_tgt[valid_index[self.cur_ind+i]])
            extracted_node_tgt.append(self.node_attr[valid_index[self.cur_ind+i]])
            extracted_pos_tgt.append(self.pose_src[valid_index[self.cur_ind+i]])


        #change into torch tensor

        if self.args.cuda:
            extracted_adj_tgt = torch.FloatTensor(extracted_adj_tgt).cuda()
            extracted_adj_src = torch.FloatTensor(extracted_adj_src).cuda()
            if self.args.enable_feat:
                extracted_node_tgt = torch.FloatTensor(extracted_node_tgt).cuda()
                extracted_node_src = torch.FloatTensor(extracted_node_src).cuda()
            else:
                extracted_node_tgt = None
                extracted_node_src = None
            extracted_pos_tgt = torch.FloatTensor(extracted_pos_tgt).cuda()
            extracted_valid_ind = torch.IntTensor(extracted_valid_ind).cuda()
            extracted_pos_src = torch.FloatTensor(extracted_pos_src).cuda()

        else:
            extracted_adj_tgt = torch.FloatTensor(extracted_adj_tgt)
            extracted_adj_src = torch.FloatTensor(extracted_adj_src)
            if self.args.enable_feat:
                extracted_node_tgt = torch.FloatTensor(extracted_node_tgt)
                extracted_node_src = torch.FloatTensor(extracted_node_src)
            else:
                extracted_node_tgt = None
                extracted_node_src = None
            extracted_pos_tgt = torch.FloatTensor(extracted_pos_tgt)
            extracted_valid_ind = torch.IntTensor(extracted_valid_ind)
            extracted_pos_src = torch.FloatTensor(extracted_pos_src)
        

        self.cur_ind = self.cur_ind + num

        return extracted_node_src, extracted_adj_src, extracted_adj_tgt, extracted_pos_src, extracted_pos_tgt, extracted_valid_ind
        #return extracted_node_src, extracted_adj_src, extracted_adj_src, extracted_pos_src, extracted_pos_src, extracted_valid_ind


    def get_unpair(self, given_valid_index=None, num=1, preset_size=None):
        if given_valid_index is not None:
            tot_num = len(given_valid_index)
        else:
            tot_num = self.tot_num

        if num >= tot_num:
            print("unpaired dataset too small")

        extracted_adj_src = []
        extracted_node_src = []
        extracted_pos_src = []
        extracted_adj_tgt = []
        extracted_node_tgt = []
        extracted_pos_tgt = []
        extracted_valid_ind_src = []
        extracted_valid_ind_tgt = []

        #for source graph:
        valid_index = copy.deepcopy(given_valid_index)
        random.shuffle(valid_index)
        
        for i in range(num):
            extracted_adj_src.append(self.adj_src[valid_index[i]])
            extracted_node_src.append(self.node_attr[valid_index[i]])
            extracted_pos_src.append(self.pose_src[valid_index[i]])
            extracted_valid_ind_src.append(self.adj_src[self.cur_ind+i].shape[0])

        #for tgt graph:
        valid_index = copy.deepcopy(given_valid_index)
        random.shuffle(valid_index)

        for i in range(num):
            extracted_adj_tgt.append(self.adj_tgt[valid_index[i]])
            extracted_node_tgt.append(self.node_attr[valid_index[i]])
            extracted_pos_tgt.append(self.pose_src[valid_index[i]])
            extracted_valid_ind_tgt.append(self.adj_src[self.cur_ind+i].shape[0])

        #change into torch tensor

        if self.args.cuda:
            extracted_adj_tgt = torch.FloatTensor(extracted_adj_tgt).cuda()
            extracted_adj_src = torch.FloatTensor(extracted_adj_src).cuda()
            if self.args.enable_feat:
                extracted_node_tgt = torch.FloatTensor(extracted_node_tgt).cuda()
                extracted_node_src = torch.FloatTensor(extracted_node_src).cuda()
            else:
                extracted_node_tgt = None
                extracted_node_src = None
            extracted_pos_tgt = torch.FloatTensor(extracted_pos_tgt).cuda()
            extracted_valid_ind_src = torch.IntTensor(extracted_valid_ind_src).cuda()
            extracted_valid_ind_tgt = torch.IntTensor(extracted_valid_ind_tgt).cuda()
            extracted_pos_src = torch.FloatTensor(extracted_pos_src).cuda()

        else:
            extracted_adj_tgt = torch.FloatTensor(extracted_adj_tgt)
            extracted_adj_src = torch.FloatTensor(extracted_adj_src)
            if self.args.enable_feat:
                extracted_node_tgt = torch.FloatTensor(extracted_node_tgt)
                extracted_node_src = torch.FloatTensor(extracted_node_src)
            else:
                extracted_node_tgt = None
                extracted_node_src = None
            extracted_pos_tgt = torch.FloatTensor(extracted_pos_tgt)
            extracted_valid_ind = torch.IntTensor(extracted_valid_ind)
            extracted_valid_ind_tgt = torch.IntTensor(extracted_valid_ind_tgt)
            extracted_pos_src = torch.FloatTensor(extracted_pos_src)

        return extracted_node_src, extracted_node_tgt, extracted_adj_src, extracted_adj_tgt, extracted_pos_src, extracted_pos_tgt, extracted_valid_ind_src, extracted_valid_ind_tgt


class LoadProcessedDataset(GraphDataset):
    """docstring for PairedTransGraph"""
    def __init__(self, args):
        '''
        require adj in: csr matirx
        require feature in: np.array
        '''
        super(LoadProcessedDataset, self).__init__(args)
        self.args = args

        if args.dataset == 'dblp_split':
            self.tot_num = 16500
        else:
            ipdb.set_trace()

        self.cur_ind = 0

    def return_full_index(self):

        return np.arange(self.tot_num)


    def get_ordered_sample(self, valid_index=None, num=1, preset_size=100):
        if valid_index is not None:
            tot_num = len(valid_index)
        else:
            tot_num = self.tot_num

        if self.cur_ind+num >= tot_num:
            self.cur_ind = 0
            print("finish obtain one epoch of data")

        #for source graph:
        extracted_ind = []
        extracted_adj_src = []
        extracted_node_src = []
        extracted_pos_src = []
        extracted_adj_tgt = []
        extracted_node_tgt = []
        extracted_pos_tgt = []
        extracted_valid_ind = []

        for i in range(num):

            #sample a center node
            addr = './dataset/dblp_split/'+str(preset_size)+'/'+str(valid_index[self.cur_ind+i])+'.p'
            data = pickle.load(open(addr, "rb"))

            

            extracted_adj_src.append(data['adj_src'])
            extracted_node_src.append(data['node'])
            extracted_pos_src.append(data['pose_src'])
            extracted_valid_ind.append(data['valid_ind'])

            extracted_adj_tgt.append(data['adj_tgt'])
            extracted_node_tgt.append(data['node'])
            extracted_pos_tgt.append(data['pose_tgt'])


        extracted_adj_src = np.array(extracted_adj_src)
        extracted_adj_tgt = np.array(extracted_adj_tgt)
        extracted_node_src = np.array(extracted_node_src)
        extracted_node_tgt = np.array(extracted_node_tgt)
        extracted_pos_src = np.array(extracted_pos_src)
        extracted_pos_tgt = np.array(extracted_pos_tgt)
        extracted_valid_ind = np.array(extracted_valid_ind)

        if self.args.cuda:
            extracted_adj_tgt = torch.FloatTensor(extracted_adj_tgt).cuda()
            extracted_adj_src = torch.FloatTensor(extracted_adj_src).cuda()
            if self.args.enable_feat:
                extracted_node_tgt = torch.FloatTensor(extracted_node_tgt).cuda()
                extracted_node_src = torch.FloatTensor(extracted_node_src).cuda()
            else:
                extracted_node_tgt = None
                extracted_node_src = None
            extracted_pos_tgt = torch.FloatTensor(extracted_pos_tgt).cuda()
            extracted_valid_ind = torch.IntTensor(extracted_valid_ind).cuda()
            extracted_pos_src = torch.FloatTensor(extracted_pos_src).cuda()

        else:
            extracted_adj_tgt = torch.FloatTensor(extracted_adj_tgt)
            extracted_adj_src = torch.FloatTensor(extracted_adj_src)
            if self.args.enable_feat:
                extracted_node_tgt = torch.FloatTensor(extracted_node_tgt)
                extracted_node_src = torch.FloatTensor(extracted_node_src)
            else:
                extracted_node_tgt = None
                extracted_node_src = None
            extracted_pos_tgt = torch.FloatTensor(extracted_pos_tgt)
            extracted_valid_ind = torch.IntTensor(extracted_valid_ind)
            extracted_pos_src = torch.FloatTensor(extracted_pos_src)
        

        self.cur_ind = self.cur_ind + num

        return extracted_node_src, extracted_adj_src, extracted_adj_tgt, extracted_pos_src, extracted_pos_tgt, extracted_valid_ind

    def get_unpair(self, given_valid_index=None, num=1, preset_size=None):
        if given_valid_index is not None:
            tot_num = len(given_valid_index)
        else:
            tot_num = self.tot_num

        if num >= tot_num:
            print("unpaired dataset too small")

        extracted_adj_src = []
        extracted_node_src = []
        extracted_pos_src = []
        extracted_adj_tgt = []
        extracted_node_tgt = []
        extracted_pos_tgt = []
        extracted_valid_ind_src = []
        extracted_valid_ind_tgt = []

        #for source graph:
        valid_index = copy.deepcopy(given_valid_index)
        random.shuffle(valid_index)
        for i in range(num):
            #sample a center node
            addr = './dataset/dblp_split/'+str(preset_size)+'/'+str(valid_index[i])+'.p'
            data = pickle.load(open(addr, "rb"))

            extracted_adj_src.append(data['adj_src'])
            extracted_node_src.append(data['node'])
            extracted_pos_src.append(data['pose_src'])
            extracted_valid_ind_src.append(data['valid_ind'])

        #for tgt graph:
        valid_index = copy.deepcopy(given_valid_index)
        random.shuffle(valid_index)
        for i in range(num):
            #sample a center node
            addr = './dataset/dblp_split/'+str(preset_size)+'/'+str(valid_index[i])+'.p'
            data = pickle.load(open(addr, "rb"))

            extracted_adj_tgt.append(data['adj_src'])
            extracted_node_tgt.append(data['node'])
            extracted_pos_tgt.append(data['pose_src'])
            extracted_valid_ind_tgt.append(data['valid_ind'])


        extracted_adj_src = np.array(extracted_adj_src)
        extracted_adj_tgt = np.array(extracted_adj_tgt)
        extracted_node_src = np.array(extracted_node_src)
        extracted_node_tgt = np.array(extracted_node_tgt)
        extracted_pos_src = np.array(extracted_pos_src)
        extracted_pos_tgt = np.array(extracted_pos_tgt)
        extracted_valid_ind_src = np.array(extracted_valid_ind_src)
        extracted_valid_ind_tgt = np.array(extracted_valid_ind_tgt)

        #change into torch tensor

        if self.args.cuda:
            extracted_adj_tgt = torch.FloatTensor(extracted_adj_tgt).cuda()
            extracted_adj_src = torch.FloatTensor(extracted_adj_src).cuda()
            if self.args.enable_feat:
                extracted_node_tgt = torch.FloatTensor(extracted_node_tgt).cuda()
                extracted_node_src = torch.FloatTensor(extracted_node_src).cuda()
            else:
                extracted_node_tgt = None
                extracted_node_src = None
            extracted_pos_tgt = torch.FloatTensor(extracted_pos_tgt).cuda()
            extracted_valid_ind_src = torch.IntTensor(extracted_valid_ind_src).cuda()
            extracted_valid_ind_tgt = torch.IntTensor(extracted_valid_ind_tgt).cuda()
            extracted_pos_src = torch.FloatTensor(extracted_pos_src).cuda()

        else:
            extracted_adj_tgt = torch.FloatTensor(extracted_adj_tgt)
            extracted_adj_src = torch.FloatTensor(extracted_adj_src)
            if self.args.enable_feat:
                extracted_node_tgt = torch.FloatTensor(extracted_node_tgt)
                extracted_node_src = torch.FloatTensor(extracted_node_src)
            else:
                extracted_node_tgt = None
                extracted_node_src = None
            extracted_pos_tgt = torch.FloatTensor(extracted_pos_tgt)
            extracted_valid_ind_src = torch.IntTensor(extracted_valid_ind_src)
            extracted_valid_ind_tgt = torch.IntTensor(extracted_valid_ind_tgt)
            extracted_pos_src = torch.FloatTensor(extracted_pos_src)

        return extracted_node_src, extracted_node_tgt, extracted_adj_src, extracted_adj_tgt, extracted_pos_src, extracted_pos_tgt, extracted_valid_ind_src, extracted_valid_ind_tgt


class TrafficDataset(GraphDataset):
    """docstring for PairedTransGraph"""
    def __init__(self, args, data):
        '''
        require adj in: csr matirx
        require feature in: np.array
        '''
        super(TrafficDataset, self).__init__(args)
        self.args = args

        adj_set, valid_ind = data

        self.adj_set = adj_set
        self.id_to_graph = self.adj_set.keys()
        
        self.candidate_list = self.filter_graph()
        self.tot_num = len(self.candidate_list)

        print('traffic dataset size '+ str(self.tot_num))

        self.cur_ind = 0


    def filter_graph(self):
        candidate_list = []
        for i in range(len(self.adj_set.keys())):
            time = self.id_to_graph[i]
            if time%(31*24) >=6 and time%(31*24) <= 17:
                if time+1 in self.adj_set:
                    candidate_list.append(time)

        return candidate_list

    def return_full_index(self):


        return np.arange(self.tot_num)


    def get_ordered_sample(self, valid_index=None, num=1, preset_size=100):
        if valid_index is not None:
            tot_num = len(valid_index)
        else:
            tot_num = self.tot_num

        if self.cur_ind+num >= tot_num:
            self.cur_ind = 0
            print("finish obtain one epoch of data")

        #for source graph:
        extracted_ind = []
        extracted_adj_src = []
        extracted_node_src = []
        extracted_pos_src = []
        extracted_adj_tgt = []
        extracted_node_tgt = []
        extracted_pos_tgt = []
        extracted_valid_ind_src = []
        extracted_valid_ind_tgt = []

        for i in range(num):

            #source graph
            cand_index = self.candidate_list[cur_ind+i]
            adj_extracted = self.adj_set[cand_index].tocsc()

            valid_adj = copy.deepcopy(adj_extracted)
            pos_extracted = get_pos_with_anchor(valid_adj, anchor_number=self.args.npos)

            adj_extracted = np.array(adj_extracted.todense())

            if self.args.setting != 'checkpoint_full': 
                adj_extracted = np.clip(adj_extracted, 0.0,1.0)
            else:
                adj_extracted = np.clip(adj_extracted, 0.0,100.0)
                adj_extracted[adj_extracted<=2.0] = 0.0

            history = np.zeros(adj_extracted.shape)
            count = 0
            for j in range(10):
                if cand_index-j in self.adj_set:
                    history = history + adj_set[cand_index-j].todense()
            feature = history/count


            extracted_adj_src.append(adj_extracted)
            extracted_node_src.append(feature)
            extracted_pos_src.append(pos_extracted)
            extracted_valid_ind_src.append(adj_extracted.shape[0])

            ###target graph
            cand_index = self.candidate_list[cur_ind+i]+1
            adj_extracted = self.adj_set[cand_index].tocsc()

            valid_adj = copy.deepcopy(adj_extracted)
            pos_extracted = get_pos_with_anchor(valid_adj, anchor_number=self.args.npos)

            adj_extracted = np.array(adj_extracted.todense())

            if self.args.setting != 'checkpoint_full': 
                adj_extracted = np.clip(adj_extracted, 0.0,1.0)
            else:
                adj_extracted = np.clip(adj_extracted, 0.0,100.0)
                adj_extracted[adj_extracted<=2.0] = 0.0

            history = np.zeros(adj_extracted.shape)
            count = 0
            for j in range(10):
                if cand_index-j in self.adj_set:
                    history = history + adj_set[cand_index-j].todense()
            feature = history/count


            extracted_adj_tgt.append(adj_extracted)
            extracted_node_tgt.append(feature)
            extracted_pos_tgt.append(pos_extracted)
            extracted_valid_ind_tgt.append(adj_extracted.shape[0])


        extracted_adj_src = np.array(extracted_adj_src)
        extracted_adj_tgt = np.array(extracted_adj_tgt)
        extracted_node_src = np.array(extracted_node_src)
        extracted_node_tgt = np.array(extracted_node_tgt)
        extracted_pos_src = np.array(extracted_pos_src)
        extracted_pos_tgt = np.array(extracted_pos_tgt)
        extracted_valid_ind_src = np.array(extracted_valid_ind_src)
        extracted_valid_ind_tgt = np.array(extracted_valid_ind_tgt)

        if self.args.cuda:
            extracted_adj_tgt = torch.FloatTensor(extracted_adj_tgt).cuda()
            extracted_adj_src = torch.FloatTensor(extracted_adj_src).cuda()
            if self.args.enable_feat:
                extracted_node_tgt = torch.FloatTensor(extracted_node_tgt).cuda()
                extracted_node_src = torch.FloatTensor(extracted_node_src).cuda()
            else:
                extracted_node_tgt = None
                extracted_node_src = None
            extracted_pos_tgt = torch.FloatTensor(extracted_pos_tgt).cuda()
            extracted_valid_ind_src = torch.IntTensor(extracted_valid_ind_src).cuda()
            extracted_valid_ind_tgt = torch.IntTensor(extracted_valid_ind_tgt).cuda()
            extracted_pos_src = torch.FloatTensor(extracted_pos_src).cuda()

        else:
            extracted_adj_tgt = torch.FloatTensor(extracted_adj_tgt)
            extracted_adj_src = torch.FloatTensor(extracted_adj_src)
            if self.args.enable_feat:
                extracted_node_tgt = torch.FloatTensor(extracted_node_tgt)
                extracted_node_src = torch.FloatTensor(extracted_node_src)
            else:
                extracted_node_tgt = None
                extracted_node_src = None
            extracted_pos_tgt = torch.FloatTensor(extracted_pos_tgt)
            extracted_valid_ind_src = torch.IntTensor(extracted_valid_ind_src)
            extracted_valid_ind_tgt = torch.IntTensor(extracted_valid_ind_tgt)
            extracted_pos_src = torch.FloatTensor(extracted_pos_src)
        

        self.cur_ind = self.cur_ind + num

        return extracted_node_src, extracted_adj_src, extracted_adj_tgt, extracted_pos_src, extracted_pos_tgt, extracted_valid_ind

    def get_unpair(self, given_valid_index=None, num=1, preset_size=None):
        if given_valid_index is not None:
            tot_num = len(given_valid_index)
        else:
            tot_num = self.tot_num

        if 2*num >= tot_num:
            print("unpaired dataset too small")

        #for source graph:
        extracted_ind = []
        extracted_adj_src = []
        extracted_node_src = []
        extracted_pos_src = []
        extracted_adj_tgt = []
        extracted_node_tgt = []
        extracted_pos_tgt = []
        extracted_valid_ind_src = []
        extracted_valid_ind_tgt = []


        valid_index = copy.deepcopy(given_valid_index)
        random.shuffle(valid_index)

        for i in range(num):

            #source graph
            cand_index = self.candidate_list[valid_index[i*2]]
            adj_extracted = self.adj_set[cand_index].tocsc()

            valid_adj = copy.deepcopy(adj_extracted)
            pos_extracted = get_pos_with_anchor(valid_adj, anchor_number=self.args.npos)

            adj_extracted = np.array(adj_extracted.todense())

            if self.args.setting != 'checkpoint_full': 
                adj_extracted = np.clip(adj_extracted, 0.0,1.0)
            else:
                adj_extracted = np.clip(adj_extracted, 0.0,100.0)
                adj_extracted[adj_extracted<=2.0] = 0.0

            history = np.zeros(adj_extracted.shape)
            count = 0
            for j in range(10):
                if cand_index-j in self.adj_set:
                    history = history + adj_set[cand_index-j].todense()
            feature = history/count


            extracted_adj_src.append(adj_extracted)
            extracted_node_src.append(feature)
            extracted_pos_src.append(pos_extracted)
            extracted_valid_ind_src.append(adj_extracted.shape[0])

            ###target graph
            cand_index = self.candidate_list[valid_index[2*i+1]]
            adj_extracted = self.adj_set[cand_index].tocsc()

            valid_adj = copy.deepcopy(adj_extracted)
            pos_extracted = get_pos_with_anchor(valid_adj, anchor_number=self.args.npos)

            adj_extracted = np.array(adj_extracted.todense())

            if self.args.setting != 'checkpoint_full': 
                adj_extracted = np.clip(adj_extracted, 0.0,1.0)
            else:
                adj_extracted = np.clip(adj_extracted, 0.0,100.0)
                adj_extracted[adj_extracted<=2.0] = 0.0

            history = np.zeros(adj_extracted.shape)
            count = 0
            for j in range(10):
                if cand_index-j in self.adj_set:
                    history = history + adj_set[cand_index-j].todense()
            feature = history/count


            extracted_adj_tgt.append(adj_extracted)
            extracted_node_tgt.append(feature)
            extracted_pos_tgt.append(pos_extracted)
            extracted_valid_ind_tgt.append(adj_extracted.shape[0])


        extracted_adj_src = np.array(extracted_adj_src)
        extracted_adj_tgt = np.array(extracted_adj_tgt)
        extracted_node_src = np.array(extracted_node_src)
        extracted_node_tgt = np.array(extracted_node_tgt)
        extracted_pos_src = np.array(extracted_pos_src)
        extracted_pos_tgt = np.array(extracted_pos_tgt)
        extracted_valid_ind_src = np.array(extracted_valid_ind_src)
        extracted_valid_ind_tgt = np.array(extracted_valid_ind_tgt)

        if self.args.cuda:
            extracted_adj_tgt = torch.FloatTensor(extracted_adj_tgt).cuda()
            extracted_adj_src = torch.FloatTensor(extracted_adj_src).cuda()
            if self.args.enable_feat:
                extracted_node_tgt = torch.FloatTensor(extracted_node_tgt).cuda()
                extracted_node_src = torch.FloatTensor(extracted_node_src).cuda()
            else:
                extracted_node_tgt = None
                extracted_node_src = None
            extracted_pos_tgt = torch.FloatTensor(extracted_pos_tgt).cuda()
            extracted_valid_ind_src = torch.IntTensor(extracted_valid_ind_src).cuda()
            extracted_valid_ind_tgt = torch.IntTensor(extracted_valid_ind_tgt).cuda()
            extracted_pos_src = torch.FloatTensor(extracted_pos_src).cuda()

        else:
            extracted_adj_tgt = torch.FloatTensor(extracted_adj_tgt)
            extracted_adj_src = torch.FloatTensor(extracted_adj_src)
            if self.args.enable_feat:
                extracted_node_tgt = torch.FloatTensor(extracted_node_tgt)
                extracted_node_src = torch.FloatTensor(extracted_node_src)
            else:
                extracted_node_tgt = None
                extracted_node_src = None
            extracted_pos_tgt = torch.FloatTensor(extracted_pos_tgt)
            extracted_valid_ind_src = torch.IntTensor(extracted_valid_ind_src)
            extracted_valid_ind_tgt = torch.IntTensor(extracted_valid_ind_tgt)
            extracted_pos_src = torch.FloatTensor(extracted_pos_src)

        return extracted_node_src, extracted_node_tgt, extracted_adj_src, extracted_adj_tgt, extracted_pos_src, extracted_pos_tgt, extracted_valid_ind_src, extracted_valid_ind_tgt




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch', type=int, default=5000, help='number of training epochs')
    parser.add_argument('--Ds', type=int, default=10,help='The State Dimention')
    parser.add_argument('--Ds_inter', type=int, default=10,help='The State Dimention of inter state')
    parser.add_argument('--No', type=int, default=20, help='The Number of Objects')
    parser.add_argument('--Nr', type=int, default=380, help='The Number of Relations')
    parser.add_argument('--Dr', type=int, default=1,help='The Relationship Dimension')
    parser.add_argument('--Dr_inter', type=int, default=1,help='The Relationship Dimension of inter state')
    parser.add_argument('--Dx', type=int, default=2000,help='The External Effect Dimension')
    parser.add_argument('--De_o', type=int, default=20,help='The Effect Dimension on node')
    parser.add_argument('--De_r', type=int, default=20,help='The Effect Dimension on edge')
    parser.add_argument('--Mini_batch', type=int, default=1,help='The training mini_batch')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint20',help='models are saved here')
    parser.add_argument('--Type', dest='Type', default='train',help='train or test')
    parser.add_argument('--O_t', action='store_true', default=False,help='indicate whether use O as target')
    args = parser.parse_args()

    args.Nr = args.No*(args.No-1)

    adj, adj_tgt, feature = load_paired_data(path='./dataset/dblp', dataset="dblp")
    dataset = PairedTransData(args, adj, adj_tgt, feature)

    #ipdb.set_trace()
    for i in range(adj.shape[0]//args.Mini_batch):
        extracted_node_src, extracted_adj_src, extracted_adj_tgt, extracted_pos_src, extracted_pos_tgt, extracted_valid_ind = dataset.get_tosave_batch(num=args.Mini_batch, preset_size=args.No)

        dataset.save_batched_pickle(i, extracted_node_src, extracted_adj_src, extracted_adj_tgt, extracted_pos_src, extracted_pos_tgt, extracted_valid_ind)
    print("process transduct to induct dataset finished!!!")