import torch
from torch import nn
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import degree
# from torch_geometric import nn as nng
from torch_scatter import scatter
from layers import (CoAttentionLayerDrugBank,
                    CoAttentionLayerTwosides,
                    )
import random
from selfattention import SelfAttention



class GmpnnCSNetDrugBank(nn.Module):
    def __init__(self, in_feats, edge_feats, hid_feats, rel_total, n_iter, dropout=0):

        super().__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.rel_total = rel_total
        self.n_iter = n_iter
        self.dropout = dropout
        self.snd_hid_feats = hid_feats * 2

        self.mlp = nn.Sequential(
            nn.Linear(in_feats, hid_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(hid_feats, hid_feats),
            nn.BatchNorm1d(hid_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(hid_feats, hid_feats),
            nn.BatchNorm1d(hid_feats), 
            CustomDropout(self.dropout),
        )


        self.propagation_layer = GmpnnBlock(edge_feats, self.hid_feats, self.n_iter, dropout)

        self.i_pro = nn.Parameter(torch.zeros(self.snd_hid_feats , self.hid_feats))
        self.j_pro = nn.Parameter(torch.zeros(self.snd_hid_feats, self.hid_feats))
        self.bias = nn.Parameter(torch.zeros(self.hid_feats ))
        self.co_attention_layer = CoAttentionLayerDrugBank(self.snd_hid_feats)
        self.self_attention_layer = SelfAttention(self.hid_feats, self.hid_feats, self.hid_feats)
        self.rel_embs = nn.Embedding(self.rel_total, self.hid_feats)
        # self.gat = nng.GAT(self.hid_feats, self.snd_hid_feats, 5)


        glorot(self.i_pro)
        glorot(self.j_pro)

        # self.w1 = nn.Parameter(torch.zeros(64, 64))
        # self.w2 = nn.Parameter(torch.zeros(64, 64))
        # glorot(self.w1)
        # glorot(self.w2)

        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(self.snd_hid_feats),
        )
        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_hid_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_hid_feats, self.snd_hid_feats)
        )
        # self.relu = nn.ReLU(inplace=True)
        



    def forward(self, batch):
        drug_data, unique_drug_pair, rels, drug_pair_indices, node_j_for_pairs, node_i_for_pairs = batch

        drug_data.x = self.mlp(drug_data.x)
        new_feats = self.propagation_layer(drug_data)
        drug_data.x = new_feats
        x_j = drug_data.x[node_j_for_pairs] #提取出点数据 仍旧是i,j中有一个是一个原子重复两遍：
        x_i = drug_data.x[node_i_for_pairs]
        attentions = self.co_attention_layer(x_j, x_i, unique_drug_pair)
        pair_repr = attentions.unsqueeze(-1) * ((x_i[unique_drug_pair.edge_index[1]] @ self.i_pro) * (x_j[unique_drug_pair.edge_index[0]] @ self.j_pro))
        #把所有的终点索引拿出来乘以起点索引指向的点数据，目的是now 分子<->分子变成了原子<->原子， 注意这里edge_index就是pair_edge_index
        x_i = x_j = None ## Just to free up some memory space
        drug_data = new_feats = None
        node_i_for_pairs = node_j_for_pairs = None 
        attentions = None
        pair_repr = scatter(pair_repr, unique_drug_pair.edge_index_batch, reduce='add', dim=0)[drug_pair_indices]

        p_scores, n_scores, n2_scores = self.compute_score(pair_repr, rels)
        return p_scores, n_scores, n2_scores


    
    def compute_score(self, pair_repr, rels):
        batch_size = len(rels)
        neg_n = (len(pair_repr) - batch_size) // batch_size  # I case of multiple negative samples per positive sample.
        neg2_rels = []
        for r in rels:
            tmp = [i for i in range(0, 86)]
            tmp.remove(r)
            neg2_rels.append(random.choice(tmp))
        neg2_rels = torch.LongTensor(neg2_rels).to(rels.device)
        rels = torch.cat([rels, rels, neg2_rels], dim=0)
        # rels = torch.cat([rels, torch.repeat_interleave(rels, neg_n, dim=0), neg2_rels], dim=0) #根据negative的数量复制了一下rels
        #对于每一组正负例样本，以一组为例：正例和负例样本的rels是一样的，并且在上面这行代码已经复制过了。所以我们只要在0-85的数字的list里减去这个

        rels = self.rel_embs(rels)
        
        a = pair_repr * rels
        a = a.unsqueeze(dim=1)
        b = self.self_attention_layer(a)
        a = a + b
        # a = self.relu(a)
        
        a = a.squeeze(dim=1)
        scores = (a).sum(-1)
        
        

        # scores = (pair_repr * rels).sum(-1) #通过相乘，为之后得到loss做铺垫，可以这样做是因为pair_repr有两组，rels有两个
        p_scores, n_scores, n2_scores = scores[:batch_size].unsqueeze(-1), scores[batch_size:batch_size*2].view(batch_size, -1, 1), scores[batch_size*2:].view(batch_size, -1, 1)
#把scores前半部分作posscores,后半部分做negscores.
        return p_scores, n_scores, n2_scores


class GmpnnCSNetTwosides(GmpnnCSNetDrugBank):
    def __init__(self, in_feats, edge_feats, hid_feats, rel_total, n_iter, dropout=0):
        super().__init__(in_feats, edge_feats, hid_feats, rel_total, n_iter, dropout)

        self.co_attention_layer = CoAttentionLayerTwosides(self.hid_feats * 2)
        self.rel_embs = nn.Embedding(self.rel_total, self.hid_feats * 2)
        self.rel_proj = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.hid_feats * 2, self.hid_feats * 2),
            nn.PReLU(),
            nn.Linear(self.hid_feats * 2, self.hid_feats),
        )
        self.s_pro = self.i_pro
        self.j_pro = self.i_pro =  None 

    def forward(self, batch):
        drug_data, drug_pairs, rels, batch_size, node_j_for_pairs, node_i_for_pairs = batch

        drug_data.x = self.mlp(drug_data.x)

        new_feats = self.propagation_layer(drug_data)
        drug_data.x = new_feats

        x_j = drug_data.x[node_j_for_pairs]
        x_i = drug_data.x[node_i_for_pairs]
        rels = self.rel_embs(rels)
        attentions = self.co_attention_layer(x_j, x_i, drug_pairs, rels)
        
        pair_repr = attentions.unsqueeze(-1) * ((x_i[drug_pairs.edge_index[1]] @ self.s_pro) * (x_j[drug_pairs.edge_index[0]] @ self.s_pro))

        drug_data = new_feats = None
        node_i_for_pairs = node_j_for_pairs = None 
        attentions = None
        x_i = x_j = None
        pair_repr = scatter(pair_repr, drug_pairs.edge_index_batch, reduce='add', dim=0)
        
        p_scores, n_scores = self.compute_score(pair_repr,  batch_size, rels)
        return p_scores, n_scores

    def compute_score(self, pair_repr, batch_size, rels):
        rels = self.rel_proj(rels)
        scores = (pair_repr * rels).sum(-1)
        p_scores, n_scores = scores[:batch_size].unsqueeze(-1), scores[batch_size:].view(batch_size, -1, 1)
        return p_scores, n_scores


class GmpnnBlock(nn.Module):
    def __init__(self, edge_feats, n_feats, n_iter, dropout):
        super().__init__()
        self.n_feats = n_feats
        self.n_iter = n_iter
        self.dropout = dropout
        self.snd_n_feats = n_feats * 2

        self.w_i = nn.Parameter(torch.Tensor(self.n_feats, self.n_feats))
        self.w_j = nn.Parameter(torch.Tensor(self.n_feats, self.n_feats))
        self.a = nn.Parameter(torch.Tensor(1, self.n_feats))
        self.bias = nn.Parameter(torch.zeros(self.n_feats))

        self.edge_emb = nn.Sequential(
            nn.Linear(edge_feats, self.n_feats)
        )

        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, self.snd_n_feats),
        )

        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )

        self.lin3 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )

        self.lin4 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats)
        )

        glorot(self.w_i)
        glorot(self.w_j)
        glorot(self.a)

        self.sml_mlp = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.n_feats, self.n_feats)
        )
    
    def forward(self, data):

        edge_index = data.edge_index
        edge_feats = data.edge_feats
        edge_feats = self.edge_emb(edge_feats)

        deg = degree(edge_index[1], data.x.size(0), dtype=data.x.dtype)

        assert len(edge_index[0]) == len(edge_feats) #进行图运算 eg. 一个分子45行顶点 每行64个点特征 往里面根据edge_index重复的起点/终点加入重复的行进而补充进边数据
        alpha_i = (data.x @ self.w_i)
        alpha_j = (data.x @ self.w_j)
        alpha = alpha_i[edge_index[1]] + alpha_j[edge_index[0]] + self.bias
        alpha = self.sml_mlp(alpha)

        assert alpha.shape == edge_feats.shape
        alpha = (alpha* edge_feats).sum(-1) #沿着最后一个维度求和意味着对每一行的所有元素进行求和

        alpha = alpha / (deg[edge_index[0]]) #节点归一化，不然一个节点有太多边连接数值会太大。
        edge_weights = torch.sigmoid(alpha)

        assert len(edge_weights) == len(edge_index[0])

        # abc = data.x[edge_index[0]]
        edge_attr = data.x[edge_index[0]] * edge_weights.unsqueeze(-1)

        assert len(alpha) == len(edge_attr)
        
        out = edge_attr
        for _ in range(self.n_iter):
            out = scatter(out[data.line_graph_edge_index[0]] , data.line_graph_edge_index[1], dim_size=edge_attr.size(0), dim=0, reduce='add')
            out = edge_attr + (out * edge_weights.unsqueeze(-1)) #直接相乘不是矩阵相乘

        x = data.x + scatter(out , edge_index[1], dim_size=data.x.size(0), dim=0, reduce='add') #scatter函数：把edgeindex相同的对应out相加
        x = self.mlp(x)

        return x

    def mlp(self, x):
        x = self.lin1(x)
        x = (self.lin3(self.lin2(x)) + x) / 2
        x = (self.lin4(x) + x) / 2

        return x


class CustomDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout = (lambda x: x ) if p == 0 else nn.Dropout(p)
    
    def forward(self, input):
        return self.dropout(input)
