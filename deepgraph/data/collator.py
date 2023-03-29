# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_sub_adjs_unsqueeze(x, padlen):
    xlen, xdim1,xdim2 = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim1,xdim2], dtype=x.dtype)
        new_x[:xlen, :,:] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_3d_sequences(xs, padlen1, padlen2, padlen3):
    result=torch.zeros([len(xs),padlen1,padlen2,padlen3,xs[0].shape[-1]], dtype=xs[0].dtype)
    for i,x in enumerate(xs):
        result[i,:x.shape[0], :x.shape[1], :x.shape[2], :]=x+1
    return result



def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
        )
        for item in items
    ]
    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        edge_input=edge_input,
        y=y,
    )

def collator_adj(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
            item.sorted_adj,
            item.sub_adj_mask
        )
        for item in items
    ]
    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
        sorted_adjs,
        sub_adj_masks
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    if sorted_adjs[0] is not None:
        max_subs = max(i.size(0) for i in sorted_adjs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    # edge_input = torch.cat(
    #     [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    # )
    edge_input = pad_3d_sequences(edge_inputs, max_node_num, max_node_num, max_dist)
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])


    if sorted_adjs[0] is None:
        sorted_adj=None
    else:
        sorted_adj=torch.cat([pad_sub_adjs_unsqueeze(i, max_subs) for i in sorted_adjs])
    if sub_adj_masks[0] is None:
        sub_adj_mask=None
    else:
        sub_adj_mask = torch.cat([pad_2d_unsqueeze(i, max_node_num)  for i in sub_adj_masks])

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        edge_input=edge_input,
        y=y,
        sorted_adj=sorted_adj,
        sub_adj_mask=sub_adj_mask
    )

def collator_adj_cache(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [
        (
            item.idx,
            item.subgraph_tensor_res,
            item.sorted_adj_res
        )
        for item in items
    ]

    (
        idx,
        subgraph_tensor_res,
        sorted_adj_res,
    ) = zip(*items)

    return dict(
        idx=idx,
        subgraph_tensor_res=subgraph_tensor_res,
        sorted_adj_res=sorted_adj_res,
    )



def collator_node_label(labels):
    max_node_num = max(i.size(0) for i in labels)
    label_padded=torch.cat([pad_1d_unsqueeze(i,max_node_num) for i in labels])

    return label_padded



def collatorcontrust(datas, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    datas_data=[a_data.data for a_data in datas]
    datas_data1 = [a_data.data1 for a_data in datas]
    datas_data2 = [a_data.data2 for a_data in datas]
    result=[]
    for items in [datas_data,datas_data1,datas_data2]:
        result.append(collator(items, max_node=max_node, multi_hop_max_dist=multi_hop_max_dist, spatial_pos_max=spatial_pos_max))
    res_dic={}
    res_dic['data']=result[0]
    res_dic['data1'] = result[1]
    res_dic['data2'] = result[2]
    return res_dic


