# Standard library imports
import random
import time

# Third-party imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', None)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree

from tqdm.notebook import tqdm
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

N_I = 943
N_U = 1682
K = 10

def idx_columns(data):
    le_user = pp.LabelEncoder()
    le_item = pp.LabelEncoder()
    data['user_id_idx'] = le_user.fit_transform(data['user id'].values)
    data['item_id_idx'] = le_item.fit_transform(data['item id'].values)
    return data

def data_loader(data, batch_size, n_usr=943, n_itm=1682):
    def sample_neg(x):
        while True:
            neg_id = random.randint(0, n_itm - 1)
            if neg_id not in x:
                return neg_id

    interected_items_df = data.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()
    users = interected_items_df['user_id_idx'].sample(batch_size).values
    
    users.sort()
    
    pos_items = [random.choice(x) for x  in interected_items_df.query("`user_id_idx` in @users")['item_id_idx']]
    neg_items = [sample_neg(x) for x  in interected_items_df.query("`user_id_idx` in @users")['item_id_idx']]
    
    return (
        torch.LongTensor(list(users)),
        torch.LongTensor(list(pos_items)) + n_usr,
        torch.LongTensor(list(neg_items)) + n_usr
    )

def create_edgeAttr(data):
    u_t = torch.LongTensor(data['user_id_idx'])
    i_t = torch.LongTensor(data['user_id_idx']) + N_U
    rat = torch.LongTensor(data['rating'])

    edge_index = torch.stack((
    torch.cat([u_t, i_t]),
    torch.cat([i_t, u_t])
    ))

    edge_weight = torch.cat((rat, rat))
    
    return edge_index, edge_weight

def get_topK(final_emb, K):
    final_user_Embed, final_item_Embed = torch.split(final_emb, (N_U, N_I))
    # compute the score of all user-item pairs
    relevance_score = torch.matmul(final_user_Embed, torch.transpose(final_item_Embed, 0, 1))

    # compute top scoring items for each user
    topk_relevance_indices = torch.topk(relevance_score, K).indices
    topk_relevance_indices_df = pd.DataFrame(topk_relevance_indices.cpu().numpy(),columns =['top_indx_'+str(x+1) for x in range(K)])
    topk_relevance_indices_df['user_ID'] = topk_relevance_indices_df.index
    topk_relevance_indices_df['top_rlvnt_itm'] = topk_relevance_indices_df[['top_indx_'+str(x+1) for x in range(K)]].values.tolist()
    topk_relevance_indices_df = topk_relevance_indices_df[['user_ID','top_rlvnt_itm']]

    return topk_relevance_indices_df

def get_metrics(topK_df, test_df):
    topk_relevance_indices_df = topK_df
    test_interacted_items = test_df.groupby('user_id_idx')['item_id_idx'].apply(list).reset_index()
    metrics_df = pd.merge(test_interacted_items, topk_relevance_indices_df, how= 'left', left_on = 'user_id_idx', right_on = ['user_ID'])
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df['item_id_idx'], metrics_df.top_rlvnt_itm)]

    metrics_df['recall'] = metrics_df.apply(lambda x : len(x['intrsctn_itm'])/len(x['item_id_idx']), axis = 1)
    metrics_df['precision'] = metrics_df.apply(lambda x : len(x['intrsctn_itm'])/K, axis = 1)

    return metrics_df['recall'].mean(), metrics_df['precision'].mean()


class NGCFConv(MessagePassing):
    def __init__(self, latent_dim, dropout, bias=True, **kwargs):
        super(NGCFConv, self).__init__(aggr='add', **kwargs)

        self.dropout = dropout

        self.lin_1 = nn.Linear(latent_dim, latent_dim, bias=bias)
        self.lin_2 = nn.Linear(latent_dim, latent_dim, bias=bias)

        self.init_parameters()


    def init_parameters(self):
        nn.init.xavier_uniform_(self.lin_1.weight)
        nn.init.xavier_uniform_(self.lin_2.weight)


    def forward(self, x, edge_index, edge_weight):
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
        norm = norm * edge_weight

        # Start propagating messages
        out = self.propagate(edge_index, x=(x, x), norm=norm)

        # Perform update after aggregation
        out += self.lin_1(x)
        out = F.dropout(out, self.dropout, self.training)
        return F.leaky_relu(out)


    def message(self, x_j, x_i, norm):
        return norm.view(-1, 1) * (self.lin_1(x_j) + self.lin_2(x_j * x_i))

class RecSysGNN(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_layers,
        num_users,
        num_items,
        model, # 'NGCF' or 'LightGCN'
        dropout=0.1 # Only used in NGCF
    ):
        super(RecSysGNN, self).__init__()

        self.model = model
        self.embedding = nn.Embedding(num_users + num_items, latent_dim)

        
        self.convs = nn.ModuleList(
            NGCFConv(latent_dim, dropout=dropout) for _ in range(num_layers)
        )
        
        self.init_parameters()


    def init_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight, gain=1)


    def forward(self, edge_index, edge_weight):
        emb0 = self.embedding.weight
        embs = [emb0]

        emb = emb0
        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index, edge_weight=edge_weight)
        embs.append(emb)

        out = (
        torch.cat(embs, dim=-1)
        )

        return emb0, out


    def encode_minibatch(self, users, pos_items, neg_items, edge_index, edge_weight):
        
        # print(f"in encode minibatch: {users.shape}, {pos_items.shape}, {neg_items.shape}")
        emb0, out = self(edge_index, edge_weight)
        # print(out.shape)
        return (
            out[users],
            out[pos_items],
            out[neg_items],
            emb0[users],
            emb0[pos_items],
            emb0[neg_items]
        )


u_data_columns = ['user id', 'item id', 'rating', 'timestamp']
u_data = pd.read_csv("data/raw/ml-100k/u.data", sep="\t", names=u_data_columns)
u1_test = pd.read_csv("data/raw/ml-100k/u1.test", sep="\t", names=u_data_columns)

u_data = idx_columns(u_data)
u1_test = idx_columns(u1_test)
main_edge_index, main_edge_weight = create_edgeAttr(u_data)

ngcf_model = RecSysGNN(
  latent_dim=64,
  num_layers=3,
  num_users=N_U,
  num_items=N_I,
  model='NGCF'
)

ngcf_model.load_state_dict(torch.load('models/ngcf_vf.pt'))
ngcf_model.eval()
with torch.no_grad():
    _, out = ngcf_model(main_edge_index, main_edge_weight)
    topK_df = get_topK(out, 10)

    test_topK_recall,  test_topK_precision = get_metrics(topK_df, u1_test)

    print(topK_df)
    print(f"Precision: {test_topK_precision}, Recall: {test_topK_recall}")