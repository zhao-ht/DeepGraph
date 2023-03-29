from torch_geometric.data import Data
from deepgraph.data.subsampling.sampling import *

import re




class Subgraphs_Sampler(object):
    def __init__(self,
                 sampling_mode=None,
                 minimum_redundancy=2,
                 shortest_path_mode_stride=2,
                 random_mode_sampling_rate=0.3,
                 random_init=False,
                 only_unused_nodes=True):
        super().__init__()

        self.random_init = random_init
        self.sampling_mode = sampling_mode
        self.minimum_redundancy = minimum_redundancy
        self.shortest_path_mode_stride = shortest_path_mode_stride
        self.random_mode_sampling_rate = random_mode_sampling_rate
        self.only_unused_nodes=only_unused_nodes


    def __call__(self, edge_index,subgraphs_nodes_tensor,num_nodes,must_select_list):

        if subgraphs_nodes_tensor.shape[1]>0:
            selected_subgraphs, node_selected_times = subsampling_subgraphs_general(edge_index,
                                                                            subgraphs_nodes_tensor,
                                                                            num_nodes,
                                                                            sampling_mode=self.sampling_mode,
                                                                            random_init=self.random_init,
                                                                            minimum_redundancy=self.minimum_redundancy,
                                                                            shortest_path_mode_stride=self.shortest_path_mode_stride,
                                                                            random_mode_sampling_rate=self.random_mode_sampling_rate,
                                                                            must_select_list=must_select_list,only_unused_nodes=self.only_unused_nodes)
        else:
            selected_subgraphs=[]
            node_selected_times=None

        return selected_subgraphs,node_selected_times




def subsampling_subgraphs_general(edge_index, subgraphs_nodes, num_nodes=None, sampling_mode='shortest_path', random_init=False, minimum_redundancy=0,
                          shortest_path_mode_stride=2, random_mode_sampling_rate=0.5,must_select_list=None,only_unused_nodes=None):

    assert sampling_mode in ['shortest_path', 'random', 'min_set_cover','min_set_cover_random']
    if sampling_mode == 'random':
        selected_subgraphs, node_selected_times = random_sampling_general(subgraphs_nodes, num_nodes=num_nodes, rate=random_mode_sampling_rate, minimum_redundancy=minimum_redundancy,must_select_list=must_select_list)
    if sampling_mode == 'shortest_path':
        selected_subgraphs, node_selected_times = shortest_path_sampling_general(edge_index, subgraphs_nodes, num_nodes=num_nodes, minimum_redundancy=minimum_redundancy,
                                                                         stride=max(1, shortest_path_mode_stride), random_init=random_init)
    if sampling_mode in ['min_set_cover']:

        selected_subgraphs, node_selected_times = min_set_cover_sampling_general(subgraphs_nodes,
                                                                               minimum_redundancy=minimum_redundancy, random_init=random_init,num_nodes=num_nodes,must_select_list=must_select_list,only_unused_nodes=only_unused_nodes)

    if sampling_mode in ['min_set_cover_random']:

        selected_subgraphs, node_selected_times = min_set_cover_random_sampling_general(subgraphs_nodes,
                                                                               minimum_redundancy=minimum_redundancy, random_init=random_init,num_nodes=num_nodes,must_select_list=must_select_list,only_unused_nodes=only_unused_nodes)
    else:
        raise ValueError('not supported sampling mode')

    return selected_subgraphs, node_selected_times




    



