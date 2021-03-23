import numpy as np
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
#import torch_geometric as geo
from sklearn.cluster import KMeans

import models
from utils import get_parser
from trainer import Trainer
from data_load import load_paired_data, load_auth_data, load_traffic_data
import dataset

import ipdb
import copy

#from torch.utils.tensorboard import SummaryWriter

# Training settings
parser = get_parser()

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')#not used yet

#load data
if args.dataset == 'dblp_split':
    loaded_data = dataset.LoadProcessedDataset(args)
elif args.dataset == 'BA' or args.dataset == 'ER':
    loaded_data = dataset.RawCsvDataset(args)
    args.size = 40



#ipdb.set_trace()

#add tensorboard 
#writer = SummaryWriter(str(args.dataset)+"_record")


#construct feature encoder model
if args.enable_feat:
    if args.dataset == 'BA' or args.dataset =='ER':
        nfeat = 40
        args.npos = 40
    elif args.dataset =='dblp_split':
        nfeat = 2000  
else:
    nfeat = 2

n_glob = 0

nhid = args.nhid
npos = args.npos

if args.enable_feat:
    encoder_feat_src = models.Encoder(nfeat, nhid*2, nhid, args.dropout)
    encoder_feat_tgt = models.Encoder(nfeat, nhid*2, nhid, args.dropout)
    n_glob += nhid
else:
    encoder_feat_src = None
    encoder_feat_tgt = None

if args.enable_pos:
    encoder_pos_src = models.Encoder(npos,npos,npos,args.dropout, feature_pre=args.pos_pre)
    encoder_pos_tgt = models.Encoder(npos,npos,npos,args.dropout, feature_pre=args.pos_pre)
    n_glob += npos

decoder_src = models.Decoder(npos,nhid = nhid, nfeat = nfeat,args = args, need_x = args.enable_feat, dropout = args.dropout)
decoder_tgt = models.Decoder(npos,nhid = nhid, nfeat = nfeat,args = args, need_x = args.enable_feat, dropout = args.dropout)

trans = models.Translator(n_glob*2, n_glob)#input:[node_level, global_level]
readout = models.ReadOut()

#move to cuda
if args.cuda:
    if args.enable_feat:
        encoder_feat_tgt = encoder_feat_tgt.cuda()
        encoder_feat_src = encoder_feat_src.cuda()
    #adj = adj.cuda()
    #adj_tgt = adj_tgt.cuda()
    trans = trans.cuda()
    encoder_pos_tgt = encoder_pos_tgt.cuda()
    encoder_pos_src = encoder_pos_src.cuda()
    decoder_src = decoder_src.cuda()
    decoder_tgt = decoder_tgt.cuda()
    readout = readout.cuda()


#set up trainer
if True:
    trainer = Trainer(args, encoder_feat_src, encoder_feat_tgt, 
        encoder_pos_src, encoder_pos_tgt,
        decoder_src, decoder_tgt,
        trans, readout)


#trainer.load_model('trans_pretrain')

for i in range(5000):
    trainer.train_epoch(loaded_data, epoch_num=i)

#writer.close()

#pretrain actor
'''
for ep in range(51):
    #test a step of forward
    trainer.train_epoch(adj, adj_tgt, feature, label, train_ind)

    trainer.test_epoch(adj, adj_tgt, feature, label, val_ind)
    torch.save(policy_model.state_dict(), 'checkpoint/pretrain_policy_inter{}.pth'.format(ep))

    trainer.test_epoch(adj, adj_tgt, feature, label, test_ind)
'''




