import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
import ipdb
import utils
import copy
import scipy.sparse as sp
import networkx
from itertools import chain
import models
import time
import random

class Trainer(object):#trainer in the case: need to extract subgraph.
    def __init__(self,args, encoder_feat_src, encoder_feat_tgt,
        encoder_pos_src, encoder_pos_tgt,
        decoder_src, decoder_tgt,
        trans,readout,
        summary_writer = None):

        self.args = args
        self.lam = 1.3
        self.mu = 1.6

        self.trans = trans
        self.readout = readout
        self.encoder_pos_tgt = encoder_pos_tgt
        self.encoder_pos_src = encoder_pos_src
        self.encoder_feat_tgt = encoder_feat_tgt
        self.encoder_feat_src = encoder_feat_src
        self.decoder_src = decoder_src
        self.decoder_tgt = decoder_tgt

        self.writer = summary_writer

        #initialize optimizers
        if args.enable_feat and args.enable_pos:
            self.encoder_opt = optim.Adam(chain(self.encoder_pos_src.parameters(), 
                self.encoder_feat_src.parameters(), self.encoder_feat_tgt.parameters(),
                self.encoder_pos_tgt.parameters()),
                lr=args.lr, weight_decay=args.weight_decay)
            n_glob = args.nhid + args.npos
        elif args.enable_pos:
            self.encoder_opt = optim.Adam(chain(self.encoder_pos_src.parameters(),
                self.encoder_pos_tgt.parameters()),
                lr=args.lr, weight_decay=args.weight_decay)
            n_glob = args.npos
        elif args.enable_feat:
            self.encoder_opt = optim.Adam(chain(self.encoder_feat_src.parameters(), 
                self.encoder_feat_tgt.parameters()),
                lr=args.lr, weight_decay=args.weight_decay)
            n_glob = args.nhid
        else:
            print("pos and feature should be allowed at least one")
            return

        self.decoder_opt = optim.Adam(chain(self.decoder_tgt.parameters(),
            self.decoder_src.parameters()), 
            lr=args.lr, weight_decay=args.weight_decay)

        self.trans_opt = optim.Adam(self.trans.parameters(), 
            lr=args.lr, weight_decay=args.weight_decay)

        if self.args.require_adv:
            self.discriminator = models.Glob_Discriminator(n_glob)

            if args.cuda:
                self.discriminator = self.discriminator.cuda()

            self.adv_opt = optim.Adam(self.discriminator.parameters(),
                lr=args.lr, weight_decay=args.weight_decay)
        
        if self.args.require_MI:
            self.MI_est = models.Glob_MINE(n_glob)

            if args.cuda:
                self.MI_est = self.MI_est.cuda()

            self.MI_opt = optim.Adam(self.MI_est.parameters(), 
                lr=args.lr, weight_decay=args.weight_decay)


    @staticmethod
    def add_args(parser):
        parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--dropout', type=float, default=0.1)

        parser.add_argument('--batch_nums', type=int, default=6000, help='number of batches per epoch')
        parser.add_argument('--batch_size', type=int, default=40, help='number of batches per epoch')

        parser.add_argument('--require_adv', action='store_true', default=True)
        parser.add_argument('--require_MI', action='store_true', default=True)

        return

    def obtain_sub_graph_ind(self, adj, valid_ind=None, step=2, given_ind = None):
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

    def uniform_sub_graph(self, adj, chosen_ind, batch_node_num, node_attr=None, require_valid_ind=False):
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


        if self.args.enable_pos:
            pos_extracted = self.get_pos_with_anchor(valid_adj, anchor_number=self.args.npos)
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


    def obtain_unpaired_sub_graphs(self, adj_src, adj_tgt, node_attr=None, number=2, valid_ind=None):#returns a batch of dual graph, unpaired
        """
        return two tensors, which together form a batch. 
        here, assume src and tgt share the same node_attr, but different pos_attr
        return: batches of adj, attributes, for both source and target

        requirement: adj is scipy csr matrix
        """
        assert isinstance(adj_tgt, sp.csr_matrix), "cannot obtain subgraph, wrong data type. expecting csr matrix"
        assert isinstance(adj_src, sp.csr_matrix), "cannot obtain subgraph, wrong data type. expecting csr matrix"

        #for batch:
        extracted_ind = []
        extracted_adj_src = []
        extracted_node_src = []
        extracted_pos_src = []
        extracted_valid_ind = [] #to filter out padding in loss computation

        #obtain each sub-graph
        batch_node_num = 0

        for i in range(number):
            #sample a center node
            chosen_ind = self.obtain_sub_graph_ind(adj_src, valid_ind)
            batch_node_num = batch_node_num + chosen_ind.shape[0]
            extracted_ind.append(chosen_ind)

        batch_node_num = batch_node_num // number


        #adjust the graph size, form a batch
        for i in range(number):
            chosen_ind = extracted_ind[i]

            adj_extracted, node_extracted, pos_extracted, last_valid_ind = self.uniform_sub_graph(adj_src, chosen_ind, batch_node_num ,node_attr)

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

        #for tgt graph
        extracted_ind = []
        extracted_adj_tgt = []
        extracted_node_tgt = []
        extracted_pos_tgt = []
        extracted_valid_ind = []

        batch_node_num = 0
        for i in range(number):
            #sample a center node
            chosen_ind = self.obtain_sub_graph_ind(adj_tgt, valid_ind)
            extracted_ind.append(chosen_ind)
            batch_node_num = batch_node_num + chosen_ind.shape[0]

        batch_node_num = batch_node_num // number

        #adjust the graph size, form a batch
        for i in range(number):
            chosen_ind = extracted_ind[i]

            adj_extracted, node_extracted, pos_extracted, last_valid_ind = self.uniform_sub_graph(adj_tgt, chosen_ind, batch_node_num ,node_attr)

            extracted_adj_tgt.append(adj_extracted)
            extracted_node_tgt.append(node_extracted)
            extracted_pos_tgt.append(pos_extracted)
            extracted_valid_ind.append(last_valid_ind)

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


    def obtain_paired_sub_graphs(self, adj_src, adj_tgt, node_attr=None,  number=2, valid_ind=None, start_ind = None):#returns paired graph batch
        assert isinstance(adj_tgt, sp.csr_matrix), "cannot obtain subgraph, wrong data type. expecting csr matrix"
        assert isinstance(adj_src, sp.csr_matrix), "cannot obtain subgraph, wrong data type. expecting csr matrix"

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

        for i in range(number):

            #sample a center node
            if start_ind is not None:
                chosen_ind = self.obtain_sub_graph_ind(adj_src, valid_ind, given_ind = start_ind+i)
            else:
                chosen_ind = self.obtain_sub_graph_ind(adj_src, valid_ind)

            batch_node_num = batch_node_num + chosen_ind.shape[0]
            extracted_ind.append(chosen_ind)

        batch_node_num = batch_node_num // number


        #adjust the graph size, form a batch
        for i in range(number):
            chosen_ind = extracted_ind[i]

            adj_extracted, node_extracted, pos_extracted, last_valid_ind, valid_ind = self.uniform_sub_graph(adj_src, chosen_ind, batch_node_num ,node_attr, require_valid_ind = True)
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
        for i in range(number):
            chosen_ind = extracted_ind[i]

            adj_extracted, node_extracted, pos_extracted, last_valid_ind = self.uniform_sub_graph(adj_tgt, chosen_ind, batch_node_num ,node_attr)

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



    def get_pos_naiive(self, adj, adj_tgt = None):
        """
        use eye matrix to initialize each node
        """
        pos_src = np.array(np.identity(adj.shape[0]))

        if adj_tgt is not None:
            pos_tgt = np.array(np.identity(adj_tgt.shape[0]))

            return pos_src, pos_tgt

        return pos_src

    def get_pos_with_anchor(self, adj, adj_tgt=None, anchor_number = 8):
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
        

    def set_to_train(self):
        self.encoder_pos_src.train()
        if self.args.enable_feat:
            self.encoder_feat_src.train()
        self.decoder_src.train()
        self.encoder_pos_tgt.train()
        if self.args.enable_feat:
            self.encoder_feat_tgt.train()
        self.decoder_tgt.train()
        self.discriminator.train()
        self.trans.train()
        self.MI_est.train()

        return

    def set_to_test(self):

        self.encoder_pos_src.eval()
        if self.args.enable_feat:
            self.encoder_feat_src.eval()
        self.decoder_src.eval()
        self.encoder_pos_tgt.eval()
        if self.args.enable_feat:
            self.encoder_feat_tgt.eval()
        self.decoder_tgt.eval()
        self.discriminator.eval()
        self.trans.eval()
        self.MI_est.eval()

        return

    def train_paired_batch(self, data, use_ind, glob_step = 0):

        start_time = time.time()
        
        node_src, adj_src, adj_tgt, pos_src, pos_tgt, valid_num_src = data.get_ordered_sample(use_ind, self.args.batch_size, self.args.size)
        node_tgt = node_src
        valid_num_tgt = valid_num_src

        end_time = time.time()
        data_time = end_time - start_time
        #print("data_time: "+ str(data_time))        
        node_num_log = adj_src.shape[1]

        self.set_to_train()

        #get the output from each model

        start_time = time.time()

        pos_embed_src = self.encoder_pos_src(adj_src, pos_src)
        if self.args.enable_feat:
            node_embed_src = self.encoder_feat_src(adj_src, node_src)
            embed_src = torch.cat([node_embed_src, pos_embed_src],dim=-1)
        else:
            node_embed_src = None
            embed_src = pos_embed_src

        adj_rec_src, node_rec_src = self.decoder_src(node_embed_src, pos_embed_src)

        pos_embed_tgt = self.encoder_pos_tgt(adj_tgt, pos_tgt)
        if self.args.enable_feat:
            node_embed_tgt = self.encoder_feat_tgt(adj_tgt, node_tgt)
            embed_tgt = torch.cat([node_embed_tgt, pos_embed_tgt],dim=-1)
        else:
            node_embed_tgt = None
            embed_tgt = pos_embed_tgt
        adj_rec_tgt, node_rec_tgt = self.decoder_tgt(node_embed_tgt, pos_embed_tgt)

        readout_src = self.readout(embed_src)
        readout_tgt = self.readout(embed_tgt)
        transed_embed_tgt = self.trans(embed_src, readout_src)
        readout_tgt_fake = self.readout(transed_embed_tgt)

        end_time = time.time()
        forward_time = end_time - start_time
        #print("forward_time: "+ str(forward_time))

        #compute the loss
        ##update the main part
        ###recovery loss
        start_time = time.time()

        #get two loss masks
        adj_mask_src = adj_rec_src.new(adj_rec_src.shape).fill_(1.0)
        for i in range(adj_rec_src.shape[0]):
            adj_mask_src[i, :, valid_num_src[i]:] = 0.0
            adj_mask_src[i, valid_num_src[i]:, :] = 0.0

        adj_mask_tgt = adj_rec_tgt.new(adj_rec_tgt.shape).fill_(1.0)
        for i in range(adj_rec_tgt.shape[0]):
            adj_mask_tgt[i, :, valid_num_tgt[i]:] = 0.0
            adj_mask_tgt[i, valid_num_tgt[i]:, :] = 0.0

        feat_mask_src = adj_rec_src.new(np.zeros((adj_rec_src.shape[0], adj_rec_src.shape[1], 1))).fill_(1.0)
        for i in range(adj_rec_src.shape[0]):
            feat_mask_src[i, valid_num_src[i]:, :] = 0.0
        feat_mask_tgt = adj_rec_src.new(np.zeros((adj_rec_src.shape[0], adj_rec_src.shape[1], 1))).fill_(1.0)
        for i in range(adj_rec_src.shape[0]):
            feat_mask_tgt[i, valid_num_tgt[i]:, :] = 0.0

        adj_rec_src = torch.mul(adj_rec_src, adj_mask_src)
        adj_rec_tgt = torch.mul(adj_rec_tgt, adj_mask_tgt)
        transed_embed_tgt = torch.mul(transed_embed_tgt, feat_mask_tgt)
        embed_tgt = torch.mul(embed_tgt, feat_mask_tgt)

        if self.args.enable_feat:
            node_rec_src = torch.mul(node_rec_src, feat_mask_src)
            node_rec_tgt = torch.mul(node_rec_tgt,feat_mask_tgt)

            #if self.args.setting == 'checkpoint_nopos':
            if False:
                rec_loss_src = F.mse_loss(node_rec_src, node_src)*0 + F.mse_loss(adj_rec_src.flatten(), adj_src.flatten())
                rec_loss_tgt = F.mse_loss(node_rec_tgt, node_tgt)*0 + F.mse_loss(adj_rec_tgt.flatten(), adj_tgt.flatten())
            else:
                rec_loss_src = F.mse_loss(node_rec_src, node_src)*0 + utils.adj_mse_loss(adj_rec_src, adj_src, adj_mask_src)
                rec_loss_tgt = F.mse_loss(node_rec_tgt, node_tgt)*0 + utils.adj_mse_loss(adj_rec_tgt, adj_tgt, adj_mask_tgt)

        else:
            rec_loss_src = F.mse_loss(adj_rec_src.flatten(), adj_src.flatten())
            rec_loss_tgt = F.mse_loss(adj_rec_tgt.flatten(), adj_tgt.flatten())

        ###transition loss
        trans_loss = F.mse_loss(transed_embed_tgt, embed_tgt)

        if self.args.setting == 'checkpoint_full':
            trans_loss = trans_loss*2
        elif self.args.setting == 'checkpoint_nopos' or self.args.setting == 'checkpoint':
            trans_loss = trans_loss*4
        elif self.args.setting == 'checkpoint_nopos_q':
            trans_loss = trans_loss*8

        ###adv loss
        adv_loss = -0.1 * self.discriminator(readout_src, readout_tgt_fake).mean()

        ###MI loss
        MI_loss = -0.1 * self.MI_est(readout_src, readout_tgt_fake).mean()

        loss = rec_loss_src*self.lam+rec_loss_tgt*self.lam+trans_loss+0*adv_loss+MI_loss*0
        #if self.args.setting == 'checkpoint_nopos':
        #if True:
            #loss = loss*0

        self.encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()
        self.trans_opt.zero_grad()
        loss.backward()
        self.encoder_opt.step()
        self.decoder_opt.step()
        self.trans_opt.step()

        ##update the discriminator
        dis_loss_real = -1 * self.discriminator(readout_src.detach(), readout_tgt.detach()).mean()
        dis_loss_fake = self.discriminator(readout_src.detach(), readout_tgt_fake.detach()).mean()
        dis_grad_pen = dis_loss_fake*0
        #if self.args.setting != 'checkpoint_nopos':
         #   dis_loss_fake = dis_loss_fake*0
        #utils.calc_gradient_penalty(self.discriminator, readout_src.detach(), readout_tgt.detach(), readout_tgt_fake.detach())
        dis_loss = dis_loss_real + dis_loss_fake + dis_grad_pen


        if dis_loss.item() != dis_loss.item():
            ipdb.set_trace()

        self.adv_opt.zero_grad()
        dis_loss.backward()
        self.adv_opt.step()

        MI_est_loss = -1 * self.MI_est(readout_src.detach(), readout_tgt.detach()).mean()

        self.MI_opt.zero_grad()
        MI_est_loss.backward()
        self.MI_opt.step()

        if self.writer is not None:
            ##start to log the losses
            main_log_dict = {'rec_loss_src': rec_loss_src.item(),
                            'rec_loss_tgt': rec_loss_tgt.item(),
                            'trans_loss': trans_loss.item(),
                            'adv_loss': adv_loss.item(),
                            'MI_loss': MI_loss.item()}
            self.writer.add_scalars('train_paired_main', main_log_dict, glob_step)

            adv_log_dict = {'dis_loss_real': dis_loss_real.item(),
                            'dis_loss_fake': dis_loss_fake.item(),
                            'grad_pen': dis_grad_pen.item()}
            self.writer.add_scalars('train_paired_adv', adv_log_dict, glob_step)

            self.writer.add_scalar('train_paired_MI_est', MI_est_loss.item(), glob_step)


        end_time = time.time()
        opt_time = end_time - start_time
        #print("opt_time: "+ str(opt_time))
        #print("paired batch, node_num: "+str(node_num_log)+" data_time: "+str(data_time)+" forward_time: "+str(forward_time)+" opt_time: "+str(opt_time))
        print("paired train: rec_loss: "+str((rec_loss_tgt+rec_loss_src).item())+" trans_loss: "+str(trans_loss.item())+" adv_loss: "+str(adv_loss.item())+"MI loss: "+str(MI_loss.item())+" dis_loss: "+str(dis_loss.item())+" MI_est_loss: "+str(MI_est_loss.item()))

        return

    def train_unpaired_batch(self, data, use_ind, glob_step=0):

        start_time = time.time()

        node_src, node_tgt, adj_src, adj_tgt, pos_src, pos_tgt, valid_num_src, valid_num_tgt = data.get_unpair(use_ind, self.args.batch_size, self.args.size)

        end_time = time.time()
        data_time = end_time - start_time
        #print("data_time: "+ str(data_time))
        node_num_log = adj_src.shape[1]

        #get two loss masks
        adj_mask_src = adj_src.new(adj_src.shape).fill_(1.0)
        for i in range(adj_src.shape[0]):
            adj_mask_src[i, :, valid_num_src[i]:] = 0.0
            adj_mask_src[i, valid_num_src[i]:, :] = 0.0

        adj_mask_tgt = adj_tgt.new(adj_tgt.shape).fill_(1.0)
        for i in range(adj_tgt.shape[0]):
            adj_mask_tgt[i, :, valid_num_tgt[i]:] = 0.0
            adj_mask_tgt[i, valid_num_tgt[i]:, :] = 0.0

        #get the output from each model
        self.set_to_train()
        start_time = time.time()

        pos_embed_src = self.encoder_pos_src(adj_src, pos_src)
        if self.args.enable_feat:
            node_embed_src = self.encoder_feat_src(adj_src, node_src)
            embed_src = torch.cat([node_embed_src, pos_embed_src],dim=-1)
        else:
            node_embed_src = None
            embed_src = pos_embed_src
        adj_rec_src, node_rec_src = self.decoder_src(node_embed_src, pos_embed_src)

        pos_embed_tgt = self.encoder_pos_tgt(adj_tgt, pos_tgt)
        if self.args.enable_feat:
            node_embed_tgt = self.encoder_feat_tgt(adj_tgt, node_tgt)
            embed_tgt = torch.cat([node_embed_tgt, pos_embed_tgt],dim=-1)
        else:
            node_embed_tgt = None
            embed_tgt = pos_embed_tgt
        adj_rec_tgt, node_rec_tgt = self.decoder_tgt(node_embed_tgt, pos_embed_tgt)

        readout_src = self.readout(embed_src)
        readout_tgt = self.readout(embed_tgt)
        transed_embed_tgt = self.trans(embed_src, readout_src)
        readout_tgt_fake = self.readout(transed_embed_tgt)

        end_time = time.time()
        forward_time = end_time - start_time
        #print("forward_time: "+ str(forward_time))

        #compute the loss
        ##update the main part
        ###recovery loss
        start_time = time.time()

        feat_mask_src = adj_rec_src.new(np.zeros((adj_rec_src.shape[0], adj_rec_src.shape[1], 1))).fill_(1.0)
        for i in range(adj_rec_src.shape[0]):
            feat_mask_src[i, valid_num_src[i]:, :] = 0.0
        feat_mask_tgt = adj_rec_src.new(np.zeros((adj_rec_tgt.shape[0], adj_rec_tgt.shape[1], 1))).fill_(1.0)
        for i in range(adj_rec_src.shape[0]):
            feat_mask_tgt[i, valid_num_tgt[i]:,:] = 0.0

        adj_rec_src = torch.mul(adj_rec_src, adj_mask_src)
        adj_rec_tgt = torch.mul(adj_rec_tgt, adj_mask_tgt)
        transed_embed_tgt = torch.mul(transed_embed_tgt, feat_mask_src)
        embed_tgt = torch.mul(embed_tgt, feat_mask_tgt)

        if self.args.enable_feat:
            node_rec_src = torch.mul(node_rec_src, feat_mask_src)
            node_rec_tgt = torch.mul(node_rec_tgt,feat_mask_tgt)

            rec_loss_src = F.mse_loss(node_rec_src, node_src)*0 + utils.adj_mse_loss(adj_rec_src, adj_src, adj_mask_src)
            rec_loss_tgt = F.mse_loss(node_rec_tgt, node_tgt)*0 + utils.adj_mse_loss(adj_rec_tgt, adj_tgt, adj_mask_tgt)
        else:
            rec_loss_src = F.mse_loss(adj_rec_src.flatten(), adj_src.flatten())
            rec_loss_tgt = F.mse_loss(adj_rec_tgt.flatten(), adj_tgt.flatten())

        ###adv loss
        adv_loss = -0.1 * self.discriminator(readout_src, readout_tgt_fake).mean()

        ###MI loss
        MI_loss = -0.1 * self.MI_est(readout_src, readout_tgt_fake).mean()

        loss = rec_loss_src*self.lam+rec_loss_tgt*self.lam+adv_loss*0+MI_loss*self.mu
        
        if True:
            loss = loss*0
 
        self.encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()
        self.trans_opt.zero_grad()
        loss.backward()
        self.encoder_opt.step()
        self.decoder_opt.step()
        self.trans_opt.step()

        ##update the discriminator
        dis_loss = 0.5*self.discriminator(readout_src.detach(), readout_tgt_fake.detach()).mean()
        dis_grad_pen = dis_loss
        dis_loss = dis_loss + dis_grad_pen

        self.adv_opt.zero_grad()
        dis_loss.backward()
        self.adv_opt.step()

        if self.writer is not None:
            ##start to log the losses
            main_log_dict = {'rec_loss_src': rec_loss_src.item(),
                            'rec_loss_tgt': rec_loss_tgt.item(),
                            'adv_loss': adv_loss.item(),
                            'MI_loss': MI_loss.item()}
            self.writer.add_scalars('train_unpaired_main', main_log_dict, glob_step)

            adv_log_dict = {'dis_loss_fake': dis_loss.item(),
                            'grad_pen': dis_grad_pen.item()}
            self.writer.add_scalars('train_unpaired_adv', adv_log_dict, glob_step)


        end_time = time.time()
        opt_time = end_time - start_time
        #print("opt_time: "+ str(opt_time))

        #print("unpaired batch, node_num: "+str(node_num_log)+" data_time: "+str(data_time)+" forward_time: "+str(forward_time)+" opt_time: "+str(opt_time))
        #print("rec_loss: "+str((rec_loss_tgt+rec_loss_src).item())+" adv_loss: "+str(adv_loss.item())+"MI loss: "+str(MI_loss.item())+" dis_loss: "+str(dis_loss.item()))

        return

    def evaluate_paired_batch(self, data, use_ind, start_ind = None, glob_step=0):

        start_time = time.time()

        #ipdb.set_trace()
        node_src, adj_src, adj_tgt, pos_src, pos_tgt, valid_num_src = data.get_ordered_sample(use_ind, self.args.batch_size, self.args.size)
        node_tgt = node_src
        valid_num_tgt = valid_num_src

        end_time = time.time()
        data_time = end_time - start_time
        #print("data_time: "+ str(data_time))
        node_num_log = adj_src.shape[1]

        #get the output from each model

        self.set_to_test()


        start_time = time.time()

        pos_embed_src = self.encoder_pos_src(adj_src, pos_src)
        if self.args.enable_feat:
            node_embed_src = self.encoder_feat_src(adj_src, node_src)
            embed_src = torch.cat([node_embed_src, pos_embed_src],dim=-1)
        else:
            node_embed_src = None
            embed_src = pos_embed_src
        adj_rec_src, node_rec_src = self.decoder_src(node_embed_src, pos_embed_src)

        pos_embed_tgt = self.encoder_pos_tgt(adj_tgt, pos_tgt)
        if self.args.enable_feat:
            node_embed_tgt = self.encoder_feat_tgt(adj_tgt, node_tgt)
            embed_tgt = torch.cat([node_embed_tgt, pos_embed_tgt],dim=-1)
        else:
            node_embed_tgt = None
            embed_tgt = pos_embed_tgt
        adj_rec_tgt, node_rec_tgt = self.decoder_tgt(node_embed_tgt, pos_embed_tgt)

        readout_src = self.readout(embed_src)
        readout_tgt = self.readout(embed_tgt)
        transed_embed_tgt = self.trans(embed_src, readout_src)
        readout_tgt_fake = self.readout(transed_embed_tgt)

        #predict the target for testing purpose:
        if self.args.enable_feat:
            node_embed_tgt_fake = transed_embed_tgt[:, :, :node_embed_tgt.shape[-1]]
            pos_embed_tgt_fake = transed_embed_tgt[:, :, node_embed_tgt.shape[-1]:]
        else:
            node_embed_tgt_fake = None
            pos_embed_tgt_fake = transed_embed_tgt

        adj_rec_tgt_fake, node_rec_tgt_fake = self.decoder_tgt(node_embed_tgt_fake, pos_embed_tgt_fake)

        #save the src, tgt, src_rec, tgt_rec and translated result
        if start_ind % 2 == 0:
            self.vis_quintuple(adj_src[0,:,:].data, adj_tgt[0,:,:].data, adj_rec_src[0,:,:].data, adj_rec_tgt[0,:,:].data,adj_rec_tgt_fake[0, :,:].data, glob_step)

        end_time = time.time()
        forward_time = end_time - start_time

        # translate to the target
        #compute the loss
        ##update the main part
        ###recovery loss
        start_time = time.time()

        #get two loss masks
        adj_mask_src = adj_rec_src.new(adj_rec_src.shape).fill_(1.0)
        for i in range(adj_rec_src.shape[0]):
            adj_mask_src[i, :, valid_num_src[i]:] = 0.0
            adj_mask_src[i, valid_num_src[i]:, :] = 0.0

        adj_mask_tgt = adj_rec_tgt.new(adj_rec_tgt.shape).fill_(1.0)
        for i in range(adj_rec_tgt.shape[0]):
            adj_mask_tgt[i, :, valid_num_tgt[i]:] = 0.0
            adj_mask_tgt[i, valid_num_tgt[i]:, :] = 0.0

        feat_mask_src = adj_rec_src.new(np.zeros((adj_rec_src.shape[0], adj_rec_src.shape[1], 1))).fill_(1.0)
        for i in range(adj_rec_src.shape[0]):
            feat_mask_src[i, valid_num_src[i]:, :] = 0.0
        feat_mask_tgt = adj_rec_src.new(np.zeros((adj_rec_src.shape[0], adj_rec_src.shape[1], 1))).fill_(1.0)
        for i in range(adj_rec_src.shape[0]):
            feat_mask_tgt[i, valid_num_tgt[i]:, :] = 0.0

        adj_rec_src = torch.mul(adj_rec_src, adj_mask_src)
        adj_rec_tgt = torch.mul(adj_rec_tgt, adj_mask_tgt)
        transed_embed_tgt = torch.mul(transed_embed_tgt, feat_mask_tgt)
        embed_tgt = torch.mul(embed_tgt, feat_mask_tgt)
        adj_rec_tgt_fake = torch.mul(adj_rec_tgt_fake, adj_mask_src)

        if self.args.enable_feat:
            node_rec_src = torch.mul(node_rec_src, feat_mask_src)
            node_rec_tgt = torch.mul(node_rec_tgt,feat_mask_tgt)
            node_rec_tgt_fake = torch.mul(node_rec_tgt_fake,feat_mask_tgt)

            rec_loss_src = F.mse_loss(node_rec_src, node_src)*0 + utils.adj_mse_loss(adj_rec_src, adj_src, adj_mask_src)
            rec_loss_tgt = F.mse_loss(node_rec_tgt, node_tgt)*0 + utils.adj_mse_loss(adj_rec_tgt, adj_tgt, adj_mask_tgt)
        else:
            rec_loss_src = F.mse_loss(adj_rec_src.flatten(), adj_src.flatten())
            rec_loss_tgt = F.mse_loss(adj_rec_tgt.flatten(), adj_tgt.flatten())

        ###transition loss
        trans_loss = F.mse_loss(transed_embed_tgt, embed_tgt)

        ###adv loss
        adv_loss = -1 * self.discriminator(readout_src, readout_tgt_fake).mean()
        ###MI loss
        MI_loss = -1 * self.MI_est(readout_src, readout_tgt_fake).mean()


        if self.args.enable_feat:
            #loss = F.mse_loss(node_rec_tgt, node_tgt)*0 + F.mse_loss(adj_rec_src.flatten(), adj_src.flatten()) + F.mse_loss(adj_rec_tgt.flatten(), adj_tgt.flatten())
            loss = utils.adj_mse_loss(adj_rec_tgt_fake, adj_tgt, adj_mask_tgt)
        else:
            loss = F.mse_loss(adj_rec_tgt_fake.flatten(), adj_tgt.flatten()) + F.mse_loss(adj_rec_tgt.flatten(), adj_tgt.flatten())

        ##discriminator
        #ipdb.set_trace()
        dis_loss_real = -1 * self.discriminator(readout_src.detach(), readout_tgt.detach()).mean()
        dis_loss_fake = self.discriminator(readout_src.detach(), readout_tgt_fake.detach()).mean()
        dis_grad_pen = dis_loss_real*0
        dis_loss = dis_loss_fake + dis_grad_pen

        MI_est_loss = -1 * self.MI_est(readout_src.detach(), readout_tgt.detach()).mean()

        if self.writer is not None:
            ##start to log the losses
            main_log_dict = {'rec_loss_src': rec_loss_src.item(),
                            'rec_loss_tgt': rec_loss_tgt.item(),
                            'trans_loss': trans_loss.item(),
                            'loss': loss.item(),
                            'adv_loss': adv_loss.item(),
                            'MI_loss': MI_loss.item(),
                            'trans_full_loss': loss.item()}
            self.writer.add_scalars('test_paired_main', main_log_dict, glob_step)

            adv_log_dict = {'dis_loss_real': dis_loss_real.item(),
                            'dis_loss_fake': dis_loss_fake.item(),
                            'grad_pen': dis_grad_pen.item()}
            self.writer.add_scalars('test_paired_adv', adv_log_dict, glob_step)

            self.writer.add_scalar('test_paired_MI_est', MI_est_loss.item(), glob_step)



        end_time = time.time()
        opt_time = end_time - start_time

        #print("evaluate: paired batch, node_num: "+str(node_num_log)+" data_time: "+str(data_time)+" forward_time: "+str(forward_time)+" opt_time: "+str(opt_time))
        #print("rec_loss: "+str((rec_loss_tgt+rec_loss_src).item())+" trans_loss: "+str(trans_loss.item())+" loss: "+str(loss.item())+" adv_loss: "+str(adv_loss.item())+"MI loss: "+str(MI_loss.item())+" dis_loss: "+str(dis_loss.item())+" MI_est_loss: "+str(MI_est_loss.item()))

        return loss.item(), rec_loss_src.item(), rec_loss_tgt.item(), trans_loss.item(), MI_loss.item()

    def save_model(self, batch_num, loss):
        saved_content = {}

        if self.args.enable_feat:
            saved_content['encoder_feat_src'] = self.encoder_feat_src.state_dict()
        saved_content['encoder_pos_src'] = self.encoder_pos_src.state_dict()
        saved_content['decoder_src'] = self.decoder_src.state_dict()
        if self.args.enable_feat:
            saved_content['encoder_feat_tgt'] = self.encoder_feat_tgt.state_dict()
        saved_content['encoder_pos_tgt'] = self.encoder_pos_tgt.state_dict()
        saved_content['decoder_tgt'] = self.decoder_tgt.state_dict()
        saved_content['trans'] = self.trans.state_dict()
        saved_content['discriminator'] = self.discriminator.state_dict()
        saved_content['MI_est'] = self.MI_est.state_dict()
        saved_content['encoder_opt'] = self.encoder_opt.state_dict()
        saved_content['decoder_opt'] = self.decoder_opt.state_dict()

        torch.save(saved_content, self.args.setting+'/'+str(self.args.dataset)+"/{}_{}.pth".format(batch_num, loss))

        return

    def load_model(self, filename):
        loaded_content = torch.load(self.args.setting+'/' +str(self.args.dataset)+"/{}.pth".format(filename), map_location=lambda storage, loc: storage)


        if self.args.enable_feat:
            self.encoder_feat_src.load_state_dict(loaded_content['encoder_feat_src'])
        self.encoder_pos_src.load_state_dict(loaded_content['encoder_pos_src'])
        self.decoder_src.load_state_dict(loaded_content['decoder_src'])
        if self.args.enable_feat:
            self.encoder_feat_tgt.load_state_dict(loaded_content['encoder_feat_tgt'])
        self.encoder_pos_tgt.load_state_dict(loaded_content['encoder_pos_tgt'])
        self.decoder_tgt.load_state_dict(loaded_content['decoder_tgt'])
        self.trans.load_state_dict(loaded_content['trans'])
        self.discriminator.load_state_dict(loaded_content['discriminator'])
        self.MI_est.load_state_dict(loaded_content['MI_est'])

        self.encoder_opt.load_state_dict(loaded_content['encoder_opt'])
        self.decoder_opt.load_state_dict(loaded_content['decoder_opt'])

        print("successfully loaded: "+ filename)

        return

    def vis_quintuple(self, src, tgt, src_rec, tgt_rec, tgt_rec_fake, name):
        path = './vis/'+str(self.args.dataset)+'/'+self.args.setting+'/'+str(name)

        np.save(path+'_src.npy', src.cpu().numpy())
        np.save(path+'_tgt.npy', tgt.cpu().numpy())
        np.save(path+'_src_rec.npy', src_rec.cpu().numpy())
        np.save(path+'_tgt_rec.npy', tgt_rec.cpu().numpy())
        np.save(path+'_tgt_rec_fake.npy', tgt_rec_fake.cpu().numpy())

        return

    def train_epoch(self, dataset, train_split = None, epoch_num=0):
        """
        train_split: 
        """
        print('start new epoch')
        #ipdb.set_trace()
        if train_split is None:
            all_ind = dataset.return_full_index()
            #train_paired_ind = all_ind[:(adj.shape[0]*1//20)]
            train_paired_ind = all_ind[:(len(all_ind)*1//10)]
            #train_unpaired_ind = all_ind[(len(all_ind)*3//10):(len(all_ind)*5//10)]
            train_unpaired_ind = all_ind[(len(all_ind)*3//10):(len(all_ind)*7//10)]
            #train_unpaired_ind = all_ind[7000:8000]
            val_ind = all_ind[(len(all_ind)*6//10):(len(all_ind)*7//10)]
            #val_ind = all_ind[20:40]
            test_ind = all_ind[(len(all_ind)*7//10):(len(all_ind))]

        #to perform training on paired and unpaired data alternatively
        #if self.args.setting == 'checkpoint_nopos':
        #if True:
            #ipdb.set_trace()
            #self.load_model('pretrain_best')
        #'''
        if self.args.setting == 'checkpoint_nopos':
            #train_unpaired_ind = all_ind[(adj.shape[0]*3//10):(adj.shape[0]*4//10)]
            train_unpaired_ind = all_ind[(len(all_ind)*3//10):(len(all_ind)*8//10)]
            unsup_ratio = 1
        else:
            unsup_ratio = 1
            #'''
        #unsup_ratio = 0
        sup_ratio = 1
        glob_step = 0

        #ipdb.set_trace()

        for num in np.arange(len(train_paired_ind)//self.args.batch_size):
            for i in range(unsup_ratio):
                #self.train_unpaired_batch(data, train_unpaired_ind)
                #ipdb.set_trace()
                self.train_unpaired_batch(dataset, train_unpaired_ind, glob_step)
                glob_step = glob_step+1

            for i in range(sup_ratio):
                self.train_paired_batch(dataset, train_paired_ind, glob_step)
                glob_step = glob_step+1

            if num == 0:
                loss = 0.0
                rec_loss_src = 0.0
                rec_loss_tgt = 0.0
                trans_loss = 0.0
                MI_loss = 0.0
                dataset.cur_ind = 0
                for m in range(len(val_ind)//self.args.batch_size):
                    #ipdb.set_trace()
                    loss_add, rec_loss_src_add, rec_loss_tgt_add, trans_loss_add, MI_loss_add = self.evaluate_paired_batch(dataset, val_ind, start_ind = m*self.args.batch_size, glob_step=epoch_num)
                    glob_step = glob_step+1
                    loss = loss + loss_add
                    rec_loss_src = rec_loss_src + rec_loss_src_add
                    rec_loss_tgt = rec_loss_tgt + rec_loss_tgt_add
                    trans_loss = trans_loss +trans_loss_add
                    MI_loss = MI_loss + MI_loss_add
                dataset.cur_ind = 0
                loss = loss/(m+1)
                rec_loss_src = rec_loss_src/(m+1)
                rec_loss_tgt = rec_loss_tgt/(m+1)
                trans_loss = trans_loss/(m+1)
                MI_loss = MI_loss/(m+1)
                print("evaluate before training epoch %d , real loss: %f, rec_loss_src: %f, rec_loss_tgt: %f, trans_loss: %f, MI_loss: %f" 
                    % (epoch_num,loss, rec_loss_src, rec_loss_tgt, trans_loss, MI_loss))
                self.save_model(epoch_num, loss)

        #write the test and log code


        return

