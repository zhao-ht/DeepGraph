# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import joblib
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset
from functools import lru_cache
import pyximport
import torch.distributed as dist

pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos
import sys
from tqdm import tqdm

@torch.jit.script
def convert_to_single_emb(x, offset: int = 10):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def encode_token_single_tensor(graph, atom_id_max, edge_id_max):
    cur_id = graph.x.shape[0]
    graph.cur_id = cur_id
    _, index = np.unique(graph.identifiers[0, :], return_index=True)
    index = np.sort(index)
    token_ids = graph.identifiers[2, index].unsqueeze(1) + atom_id_max + 1
    graph.x = torch.cat([graph.x, token_ids.repeat([1, graph.x.shape[1]])], 0).type(torch.int64)

    edges = torch.cat([graph.identifiers[0, :].unsqueeze(0) + cur_id, graph.identifiers[1, :].unsqueeze(0)], 0).long()
    graph.edge_index = torch.cat([graph.edge_index, edges, edges[[1, 0], :]], 1)

    if graph.edge_attr is not None:
        edge_attr = graph.identifiers[2, :].unsqueeze(1).repeat([1, graph.edge_attr.shape[1]]) + edge_id_max + 2
        graph.edge_attr = torch.cat([graph.edge_attr, edge_attr, edge_attr], 0).long()

    graph.sorted_adj = None

    graph.sub_adj_mask = torch.ones([graph.x.shape[0],1]).long()
    graph.sub_adj_mask[0:graph.cur_id]=0
    graph.sub_adj_mask=graph.sub_adj_mask.long()

    return graph


def encode_token_single_tensor_with_adj(graph, local_attention_on_substructures):
    cur_id = graph.x.shape[0]
    graph.cur_id = cur_id

    _, index = np.unique(graph.identifiers[0, :], return_index=True)
    index = np.sort(index)
    token_ids = -1 * torch.ones_like(graph.identifiers[2, index].unsqueeze(1))

    graph.sorted_adj = torch.cat([torch.zeros([graph.num_nodes,
                                               graph.sorted_adj.shape[1],
                                               graph.sorted_adj.shape[2]]), graph.sorted_adj], 0)

    graph.x = torch.cat([graph.x, token_ids.repeat([1, graph.x.shape[1]])], 0).type(torch.int64)
    graph.sub_adj_mask = (graph.x < 0).long()
    if graph.sub_adj_mask.shape[1] > 1:
        graph.sub_adj_mask = graph.sub_adj_mask[:, 0].unsqueeze(1)


    # expanded edges is used for attention mask, not for position embedding
    edges = torch.cat([graph.identifiers[0, :].unsqueeze(0) + cur_id, graph.identifiers[1, :].unsqueeze(0)], 0).long()
    graph.edge_index = torch.cat([graph.edge_index, edges, edges[[1, 0], :]], 1)

    if not local_attention_on_substructures:
    # expanded edge_attr is actually not used in local_attention_on_substructures
        edge_id = graph.edge_attr.shape[0]
        graph.edge_id = edge_id
        if graph.edge_attr is not None:
            edge_attr = (-1 * torch.ones_like(graph.identifiers[2].unsqueeze(1))) \
                            .repeat([1, graph.edge_attr.shape[1]]) \
                            .type(torch.int64) + edge_id_max + 2
            graph.edge_attr = torch.cat([graph.edge_attr, edge_attr, edge_attr], 0).long()

    return graph

def  graph_data_modification_single(data,substructures,sorted_adj=None):

    if data.x.dim()==1:
        data.x=data.x.unsqueeze(1)
    if data.y.dim() == 2:
        data.y = data.y.squeeze(1)
    if hasattr(data, 'edge_features'):
        data.edge_attr=data.edge_features
        del(data.edge_features)
    if hasattr(data,'degree'):
        del(data.degrees)
    if hasattr(data,'graph_size'):
        del (data.graph_size)
    if data.edge_attr is not None and data.edge_attr.dim()==1:
        data.edge_attr=data.edge_attr.unsqueeze(1)
    assert hasattr(data,'edge_attr')
    assert hasattr(data, 'edge_index')
    setattr(data,'identifiers', substructures)
    setattr(data,'sorted_adj',sorted_adj)

    return data



def preprocess_item(item,local_attention_on_substructures=False,continuous_feature=False):
    max_dist_const=510
    substructure_dist_const=max_dist_const-1

    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x

    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj_matrix = torch.zeros([N, N], dtype=torch.bool)
    adj_matrix[edge_index[0, :], edge_index[1, :]] = True
    if local_attention_on_substructures:

        adj = adj_matrix[0:item.cur_id, 0:item.cur_id]
    else:
        adj = adj_matrix[0:,0:]



    if edge_attr is None:
        attn_edge_type = -1*torch.ones([N, N, 1], dtype=torch.long)
    else:
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        edge_attr=edge_attr.long()
        attn_edge_type[edge_index[0, :edge_attr.size(0)], edge_index[1, :edge_attr.size(0)]] = (
            convert_to_single_emb(edge_attr) + 1
        )


    #compute attention mask (attn_bias)
    if local_attention_on_substructures:
        adj_matrix = adj_matrix.float()
        adj_matrix[0:item.cur_id, 0:item.cur_id] = 1
        attn_bias=1-adj_matrix
        attn_bias[attn_bias>0]=float('-inf')
        attn_bias_res = torch.zeros([N + 1, N + 1], dtype=torch.float)
        attn_bias_res[1:,1:]=attn_bias
        attn_bias=attn_bias_res
    else:
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token



    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()

    #Expand the size of edge_input and spatial_pos
    if local_attention_on_substructures:
        edge_input_new=(-1*np.ones([N,N,edge_input.shape[2],edge_input.shape[3]])).astype(np.int64)
        edge_input_new[0:item.cur_id,0:item.cur_id,:,:]=edge_input
        edge_input=edge_input_new
        spatial_pos_new=(max_dist_const*torch.ones(N,N)).long()
        spatial_pos_new[0:item.cur_id,0:item.cur_id]=spatial_pos
        spatial_pos=spatial_pos_new


    # combine
    item.x = x.long()
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).long()


    return item



def preprocess_item_local_attention(item,local_attention_on_substructures=False,continuous_feature=False):
    max_dist_const=510
    substructure_dist_const=max_dist_const-1

    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if edge_attr is None:
        attn_edge_type = -1*torch.ones([N, N, 1], dtype=torch.long)
    else:
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        edge_attr=edge_attr.long()
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
            convert_to_single_emb(edge_attr) + 1
        )


    if local_attention_on_substructures:
        adj_matrix = torch.sparse_coo_tensor(item.edge_index, torch.ones_like(item.edge_index[0]),
                                                             [item.x.shape[0], item.x.shape[0]]
                                                             ).to_dense().float()
        adj_matrix[0:item.cur_id, 0:item.cur_id] = 1
        attn_bias=1-adj_matrix
        attn_bias[attn_bias>0]=float('-inf')
        attn_bias_res = torch.zeros([N + 1, N + 1], dtype=torch.float)
        attn_bias_res[1:,1:]=attn_bias
        attn_bias=attn_bias_res
    else:
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token


    if local_attention_on_substructures:
        adj_new=torch.zeros_like(adj)
        adj_new[0:item.cur_id, 0:item.cur_id]=adj[0:item.cur_id, 0:item.cur_id]
        adj=adj_new
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()

    if local_attention_on_substructures:
        mask=torch.tensor(shortest_path_result == max_dist_const) & adj_matrix.bool()
        shortest_path_result[mask]=substructure_dist_const

    # combine
    item.x = x.long()
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).long()


    return item


class MyPygPCQM4MDataset(PygPCQM4Mv2Dataset):
    def download(self):
        super(MyPygPCQM4MDataset, self).download()

    def process(self):
        super(MyPygPCQM4MDataset, self).process()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        return preprocess_item(item)


class MyPygGraphPropPredDataset(PygGraphPropPredDataset):
    def download(self):
        if dist.get_rank() == 0:
            super(MyPygGraphPropPredDataset, self).download()
        dist.barrier()

    def process(self):
        if dist.get_rank() == 0:
            super(MyPygGraphPropPredDataset, self).process()
        dist.barrier()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        item.y = item.y.reshape(-1)
        return preprocess_item(item)
