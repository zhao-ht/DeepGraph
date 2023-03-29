
import os

import torch

from .utils import process_arguments_substructure, substructure_to_gt,transfer_subgraph_to_batchtensor_complete

from tqdm import tqdm

from deepgraph.data.subsampling.sampler import Subgraphs_Sampler
import lmdb

import numpy as np
from deepgraph.data.substructure_dataset_utils.neighbor_extractors import k_hop_subgraph,random_walk_subgraph

from collections import deque



def graph_data_modification(dataset,substructures,args):
    new_dataset=[]
    for i in tqdm(range(len(dataset))):
        if dataset[i].x.dim()==1:
            dataset[i].x=dataset[i].x.unsqueeze(1)
        if dataset[i].y.dim() == 2:
            dataset[i].y = dataset[i].y.squeeze(1)
        if hasattr(dataset[i], 'edge_features'):
            dataset[i].edge_attr=dataset[i].edge_features
            del(dataset[i].edge_features)
        if hasattr(dataset[i],'degree'):
            del(dataset[i].degrees)
        if hasattr(dataset[i],'graph_size'):
            del (dataset[i].graph_size)
        if dataset[i].edge_attr.dim()==1:
            dataset[i].edge_attr=dataset[i].edge_attr.unsqueeze(1)
        assert hasattr(dataset[i],'edge_attr')
        assert hasattr(dataset[i], 'edge_index')
        new_data=dataset[i]
        setattr(new_data,'identifiers', substructures[i])
        new_dataset.append(new_data)
    return new_dataset




def find_edges_of_substructure(substructures,prototype):
    edges_res=[]
    for nodes in substructures:
        for edge in prototype:
            edges_res.append([nodes[edge[0]],nodes[edge[1]]])
    return edges_res

def transfer_subgraph_to_batchtensor(substructures,must_select_list):
    id_lis=[]
    note_lis=[]
    cur_id=0
    must_select_list_data=[]
    for  id_sub,subtype in enumerate(substructures):
        for data in subtype:
            id_lis=id_lis+[cur_id]*len(data)
            if id_sub in must_select_list:
                must_select_list_data.append(cur_id)
            note_lis=note_lis+list(data)

            cur_id += 1

    res=torch.tensor([id_lis,note_lis])
    return res,must_select_list_data






def select_result(substructures,selected_subgraphs):
    id_lis=[]
    note_lis=[]
    cur_id=0
    substructures_new=[]
    for  subtype in substructures:
        subtype_new=set()
        for data in subtype:
            if cur_id in selected_subgraphs:
                subtype_new.add(data)
            cur_id+=1
        substructures_new.append(subtype_new)
    return substructures_new

def select_result_as_tensor(subgraph_tensor,selected_subgraphs):
    selected_index=(subgraph_tensor[0].unsqueeze(1) ==torch.tensor(selected_subgraphs).unsqueeze(0)).any(1)
    return subgraph_tensor[:,selected_index]

def re_tag(subgraph_ids):
    diff=((subgraph_ids[1:]-subgraph_ids[0:-1])>0).long()
    sum=diff.cumsum(dim=0)
    res=torch.cat([torch.zeros([1]) if subgraph_ids.shape[0]>0 else torch.tensor([]),sum]).long()
    return res



def bfs(graph_adj, start_node,sort_func):
    visited = set()  # set of visited nodes
    queue = deque([start_node])  # create a queue and enqueue the starting node

    while queue:

        node = queue.popleft()

        if node not in visited:
            # mark the node as visited
            visited.add(node)

            neighbors = (graph_adj[node].nonzero())[0]

            neighbors_not_visited=neighbors[~np.in1d(neighbors, visited)]

            sorted_neighbors_not_visited = sort_func(neighbors_not_visited)

            queue.extend(sorted_neighbors_not_visited.tolist())

    return visited

def sort_node(node_list, degrees,only_minimum=False):

    if only_minimum:
        #compute minimum over all the nodes
        min_indices = np.argwhere(np.array(degrees) == np.min(degrees)).flatten()
        res= np.random.permutation(min_indices)
    else:
        degrees=degrees[node_list]

        ind_per = np.random.permutation(len(degrees))
        node_list=node_list[ind_per]
        degrees=degrees[ind_per]

        ind_sort=np.argsort(degrees)
        res= node_list[ind_sort]

    return res


def generate_sorted_adj(edge_tensor,node_sorted,subgraph_max_size):



    adj_list = []
    if edge_tensor[0].shape[0]>0:
        number_subgraph = int(edge_tensor[0, :].max()) + 1
        max_node = int(edge_tensor[1:, :].max()) + 1
        # edges = torch.cat([edge_tensor, edge_tensor[[0, 2, 1], :]], 1)
        edge_sparse = torch.sparse_coo_tensor(edge_tensor, torch.ones_like(edge_tensor[0]), [number_subgraph, max_node, max_node]
                                              ).to_dense().long()
        node_sorted_id = []
        node_sorted_sequence = []
        node_sorted_cat = []
        for i, item in enumerate(node_sorted):
            node_sorted_cat += item
            node_sorted_sequence += list(range(len(item)))
            node_sorted_id += [i] * len(item)

        node_sorted_tensor = torch.sparse_coo_tensor([node_sorted_id, node_sorted_sequence, node_sorted_cat],
                                                     torch.ones([len(node_sorted_id)]),
                                                     [number_subgraph, subgraph_max_size, max_node]
                                                     ).to_dense().long()

        adjs = node_sorted_tensor @ edge_sparse @ (node_sorted_tensor.transpose(1, 2))

        return adjs
    else:
        return torch.ones([0,subgraph_max_size,subgraph_max_size])

class transform_sub:
    def __init__(self,args):
        args, extract_ids_fn, count_fn, automorphism_fn = process_arguments_substructure(args)
        self.args= args
        self.extract_ids_fn=extract_ids_fn
        self.count_fn=count_fn
        self.automorphism_fn=automorphism_fn
        self.subgraph_params = {'induced': args['induced'],
                           'edge_list': args['custom_edge_list'],
                           'directed': args['directed'],
                           'directed_orbits': args['directed_orbits'],
                           }
        self.must_select_list=args.must_select_list
        self.substructure_cache={}
        if args.subsampling:
            self.sampler = Subgraphs_Sampler(
                                          sampling_mode=args.sampling_mode,
                                          minimum_redundancy=args.sampling_redundancy,
                                          shortest_path_mode_stride=args.sampling_stride,
                                          random_mode_sampling_rate=args.sampling_random_rate,
                                          random_init=True,
                                          only_unused_nodes=(not self.args.not_only_unused_nodes))
    def __call__(self, data,substructure_tensor_provided):

        if substructure_tensor_provided is None:
            substructures = self.compute_substructures(data)
            substructure_tensor = transfer_subgraph_to_batchtensor_complete(substructures)
            subgraph_tensor = torch.Tensor([substructure_tensor[1], substructure_tensor[0],substructure_tensor[2]]).long()
        else:
            subgraph_tensor=substructure_tensor_provided

        if self.args.subsampling:

            must_select_list_data=[]
            for i in self.must_select_list:
                must_select_list_data+=subgraph_tensor[0][subgraph_tensor[2]==i].unique().long().tolist()

            selected_subgraphs, node_selected_times = self.sampler(data.edge_index, subgraph_tensor, data.num_nodes,must_select_list_data)

            subgraph_tensor = select_result_as_tensor(subgraph_tensor, selected_subgraphs)
            subgraph_tensor=torch.cat([re_tag(subgraph_tensor[0,:]).unsqueeze(0),subgraph_tensor[1:,:]])

        edge_tensor=self.compute_edge_tensor_of_substructures(data,subgraph_tensor)

        node_sorted=self.sort_subgraphs_randomly(data,subgraph_tensor)

        sorted_adj=generate_sorted_adj(edge_tensor,node_sorted,self.args['subgraph_max_size'])



        return subgraph_tensor,sorted_adj


    def compute_edge_tensor_of_substructures(self,data,subgraph_tensor):

        if subgraph_tensor[0].shape[0] > 0 and data.edge_index[0].shape[0]>0:
            try:
                num_subgraph=int(subgraph_tensor[0].max()+1)
                max_node_number=max(int(data.edge_index.max()+1),int(subgraph_tensor[1].max()+1))
                sub_tensor=torch.sparse_coo_tensor(subgraph_tensor[[0,1],:],
                                                   torch.ones_like(subgraph_tensor[0]),
                                                   [num_subgraph,max_node_number]).to_dense()
                edg_tensor=torch.sparse_coo_tensor(data.edge_index,torch.ones_like(data.edge_index[0]),[max_node_number,max_node_number]).to_dense()

                res=(sub_tensor.unsqueeze(1))*(edg_tensor.unsqueeze(0))*(sub_tensor.unsqueeze(2))
            except:
                print(subgraph_tensor)
                print(data.edge_index)
                print(num_subgraph)
                print(max_node_number)
                print(sub_tensor)
                print(edg_tensor)
            return res.nonzero().T

        else:
            return torch.zeros([3,0])



    def sort_subgraphs_randomly(self,data,subgraph_tensor):


        graph_adj = np.zeros((data.x.shape[0],data.x.shape[0]),dtype=np.int64)
        graph_adj[data.edge_index[0,:],data.edge_index[1,:]]=1

        if subgraph_tensor[0].shape[0]>0:
            nodes_record=[]
            for i in range(int(subgraph_tensor[0].max())+1):

                subgraph_node=subgraph_tensor[1][subgraph_tensor[0] == i]

                subgraph_adj=graph_adj[subgraph_node][:,subgraph_node]

                degrees = subgraph_adj.sum(0)

                if subgraph_adj.sum()>0:

                    root=sort_node(None,degrees,only_minimum=True)[0]

                    nodes=[n for n in bfs(subgraph_adj, start_node=root,sort_func=lambda node_array: sort_node(node_array,degrees))]

                    nodes_record.append(subgraph_node[nodes])

            return nodes_record
        else:
            return [[]]

    def compute_substructures(self,data,subgraph_dicts=None):
        if subgraph_dicts is None:
            subgraph_dicts = substructure_to_gt(self.subgraph_params, self.automorphism_fn, )
        substructures = self.extract_ids_fn(self.count_fn, data, subgraph_dicts, self.subgraph_params)
        return substructures



    def compute_neighbor(self,data):
        neighbor=None
        if 'k_neighborhood' in self.args['neighbor_type']:
            neighbor=k_hop_subgraph(data.edge_index,data.x.shape[0],self.args['neighbor_size'][self.args['neighbor_type'].index('k_neighborhood')])
        elif 'random_walk' in self.args['neighbor_type']:
            neighbor = random_walk_subgraph(data.edge_index, data.x.shape[0],
                                      self.args['neighbor_size'][self.args['neighbor_type'].index('random_walk')])
        neighbor_nodes = neighbor[0].nonzero().T
        return neighbor_nodes.tolist(),int(neighbor[0].T.sum(0).max())

    def substructure_as_tensor(self,substructure_tensor=None,neighbor_tensor=None):
        result=None
        if substructure_tensor is not None:
            result = torch.Tensor([substructure_tensor[1], substructure_tensor[0],substructure_tensor[2]]).long()
        if neighbor_tensor is not None:
            if substructure_tensor is not None and result.shape[1]>0:
                ids=(torch.Tensor([neighbor_tensor[0]])+result[0].max()+1).long()
                neighbor_tensor=torch.cat([ids,torch.Tensor([neighbor_tensor[1]]).long(),-torch.ones_like(ids)],0)
                result=torch.cat([result,neighbor_tensor],1)
            else:
                ids=(torch.Tensor([neighbor_tensor[0]])).long()
                type = (-1* torch.ones_like(ids)).long()
                result=torch.cat([ids,torch.Tensor([neighbor_tensor[1]]).long(),type],0)
        if result is None:
            result=torch.ones([3,0])
        return result

    def pre_compute_substructures_direct_to_lmdb(self,dataset):

        subgraph_dicts=substructure_to_gt(self.subgraph_params,self.automorphism_fn,)

        print('********************************transfer to lmdb type**************************')

        path_lmdb_tensor = os.path.join(self.args['data_dir'],self.args['lmdb_root_dir'],self.args['pre_defined_path'] + self.args['lmdb_dir'], 'inner')
        print('saving to ', path_lmdb_tensor)
        if not os.path.exists(path_lmdb_tensor):
            os.makedirs(path_lmdb_tensor)
        lmdb_env_tensor = lmdb.open(path_lmdb_tensor, map_size=1e12)
        txn_tensor = lmdb_env_tensor.begin(write=True)

        for idx, data in tqdm(enumerate(dataset)):

            substructure_computed=self.compute_substructures(data,subgraph_dicts)

            substructure_tensor=self.transfer_subgraph_to_batchtensor_complete(substructure_computed)

            txn_tensor.put(key=str(idx).encode(), value=str(substructure_tensor).encode())

        print('********************************commit substructures tensors**************************')
        txn_tensor.commit()
        lmdb_env_tensor.close()

    def pre_compute_neighbors_direct_to_lmdb(self,dataset):
        print('***********************************computing neighbors to lmdb type**************************')
        path_lmdb = os.path.join(self.args['data_dir'],self.args['lmdb_root_dir'],self.args['node_neighbor_path']+ self.args['lmdb_dir'], 'neighbor')
        print('saving to ', path_lmdb)
        if not os.path.exists(path_lmdb):
            os.makedirs(path_lmdb)
        lmdb_env = lmdb.open(path_lmdb, map_size=1e12)
        txn = lmdb_env.begin(write=True)
        max_size=0
        for idx, data in tqdm(enumerate(dataset)):
            substructure_computed,size=self.compute_neighbor(data)
            max_size=max(max_size,size)
            txn.put(key=str(idx).encode(), value=str(substructure_computed).encode())
        txn.put(key='max_size'.encode(),value=str(max_size).encode())
        print('********************************commit neighbors**************************')
        txn.commit()
        lmdb_env.close()


    def transfer_subgraph_to_batchtensor_complete(self,substructures):
        return transfer_subgraph_to_batchtensor_complete(substructures)



    def clean_cache(self):
        self.substructure_cache = {}
