import os
from .utils_graph_processing import subgraph_isomorphism_edge_counts, subgraph_isomorphism_vertex_counts, induced_edge_automorphism_orbits, edge_automorphism_orbits, automorphism_orbits,subgraph_isomorphism_vertex_extraction,subgraph_counts2ids,subgraph_2token

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.data import Data
import glob
import re
import types
from ast import literal_eval
from omegaconf import open_dict
import lmdb
from tqdm import tqdm
import time

def get_custom_edge_list(ks, substructure_type=None, filename=None):
    '''
        Instantiates a list of `edge_list`s representing substructures
        of type `substructure_type` with sizes specified by `ks`.
    ''' 
    if substructure_type is None and filename is None:
        raise ValueError('You must specify either a type or a filename where to read substructures from.')
    edge_lists = []
    for k in ks:
        if substructure_type is not None:
            graphs_nx = getattr(nx, substructure_type)(k)
        else:
            graphs_nx = nx.read_graph6(os.path.join(filename, 'graph{}c.g6'.format(k)))
        if isinstance(graphs_nx, list) or isinstance(graphs_nx, types.GeneratorType):
            edge_lists += [list(graph_nx.edges) for graph_nx in graphs_nx]
        else:
            edge_lists.append(list(graphs_nx.edges))
    return edge_lists


def process_arguments_substructure(args):
    with open_dict(args):
        args['k']=literal_eval(args['ks'])
        args['subgraph_max_size']=0
        tem= (args['id_type'])
        del args['id_type']
        args['id_type']=tem.split('+')
        tem= (args['must_select_sub'])
        del args['must_select_sub']
        args['must_select_sub']=tem.split('+')


    with open_dict(args):
        del args['custom_edge_list']
        args['custom_edge_list']=[]

    if args['extra_method']=='feature':
        extract_id_fn = subgraph_counts2ids
    else:
        extract_id_fn = subgraph_2token
    ###### choose the function that computes the automorphism group and the orbits #######
    if args['edge_automorphism'] == 'induced':
        automorphism_fn = induced_edge_automorphism_orbits if args['id_scope'] == 'local' else automorphism_orbits
    elif args['edge_automorphism'] == 'line_graph':
        automorphism_fn = edge_automorphism_orbits if args['id_scope'] == 'local' else automorphism_orbits
    else:
        raise NotImplementedError

    ###### choose the function that computes the subgraph isomorphisms #######
    if args['extra_method']=='feature':
        count_fn = subgraph_isomorphism_edge_counts if args['id_scope'] == 'local' else subgraph_isomorphism_vertex_counts
    else:
        count_fn = subgraph_isomorphism_vertex_extraction

    ###### choose the substructures: usually loaded from networkx,
    ###### except for 'all_simple_graphs' where they need to be precomputed,
    ###### or when a custom edge list is provided in the input by the user
    with open_dict(args):
        del (args['custom_edge_list'])
        args['custom_edge_list']=[]
        args['must_select_list']=[]
        args['neighbor_type']=[]
        args['neighbor_size'] = []
        args['pre_defined_path']=args['dataset_name']+'_'+'substructure'
        args['node_neighbor_path']=args['dataset_name']+'_'+'neighbor'
        args['transform_cache_path']=args['dataset_name']+'_'+args['sampling_mode']+'_'+str(args['sampling_redundancy'])+'_'+str(args['transform_cache_number'])

    for number,id_type in enumerate(args['id_type']):

        if id_type in ['cycle_graph','complete_graph',
                               'binomial_tree',
                               'star_graph',
                               'nonisomorphic_trees']:
            # args['k'] = args['k'][0]
            k_max = args['k'][number]
            k_min = 2 if id_type == 'star_graph' else 3
            args['subgraph_max_size']=max(args['subgraph_max_size'],k_max+1 if id_type=='star_graph' else k_max)
            with open_dict(args):
                add_sub=get_custom_edge_list(list(range(k_min, k_max + 1)), id_type)
                if id_type in args['must_select_sub']:
                    args['must_select_list']+=list(range(len(args['custom_edge_list']),
                                                         len(args['custom_edge_list'])+len(add_sub)
                                                         ))
                args['custom_edge_list'] +=add_sub
                args['pre_defined_path'] +='_'+id_type+'_'+str(args['k'][number])
                args['transform_cache_path'] += '_'+id_type+'_'+str(args['k'][number])


        elif id_type in ['path_graph']:
            k_min = args['k'][number]
            k_max = 8
            args['subgraph_max_size'] = max(args['subgraph_max_size'], k_max+1)
            with open_dict(args):
                add_sub=get_custom_edge_list(list(range(k_min, k_max + 1)), id_type)
                if id_type in args['must_select_sub']:
                    args['must_select_list']+=list(range(len(args['custom_edge_list']),
                                                         len(args['custom_edge_list'])+len(add_sub)
                                                         ))
                args['custom_edge_list'] +=add_sub
                args['pre_defined_path'] += '_' + id_type + '_' + str(args['k'][number])
                args['transform_cache_path'] += '_' + id_type + '_' + str(args['k'][number])

        elif id_type in ['k_neighborhood', 'random_walk']:
            with open_dict(args):
                args['neighbor_type'].append(id_type)
                args['neighbor_size'].append(args['k'][number])
                args['node_neighbor_path']+='_'+id_type+'_'+str(args['k'][number])
                args['transform_cache_path'] += '_' + id_type + '_' + str(args['k'][number])



        elif id_type in ['cycle_graph_chosen_k',
                                 'path_graph_chosen_k',
                                 'complete_graph_chosen_k',
                                 'binomial_tree_chosen_k',
                                 'star_graph_chosen_k',
                                 'nonisomorphic_trees_chosen_k']:
            print('warning: not complete subgraph_max_size for ',id_type)
            with open_dict(args):
                args['custom_edge_list'] += get_custom_edge_list(args['k'], id_type.replace('_chosen_k', ''))

        elif id_type in ['all_simple_graphs']:
            # args['k'] = args['k'][0]
            k_max = args['k'][number]
            k_min = 3
            args['subgraph_max_size'] = max(args['subgraph_max_size'], k_max)
            filename = os.path.join(args['root_folder'], 'all_simple_graphs')
            with open_dict(args):
                args['custom_edge_list'] += get_custom_edge_list(list(range(k_min, k_max + 1)), filename=filename)

        elif id_type in ['all_simple_graphs_chosen_k']:
            print('warning: not complete subgraph_max_size for ', id_type)
            filename = os.path.join(args['root_folder'], 'all_simple_graphs')
            with open_dict(args):
                args['custom_edge_list'] += get_custom_edge_list(args['k'], filename=filename)

        elif id_type in ['diamond_graph']:
            print('warning: not complete subgraph_max_size for ', id_type)
            # args['k'] = None
            graph_nx = nx.diamond_graph()
            with open_dict(args):
                args['custom_edge_list'] += [list(graph_nx.edges)]

        elif id_type == 'custom':
            print('warning: not complete subgraph_max_size for ', id_type)
            assert args['custom_edge_list'] is not None, "Custom edge list must be provided."

        else:
            raise NotImplementedError("Identifiers {} are not currently supported.".format(id_type))

    return args, extract_id_fn, count_fn, automorphism_fn






def transfer_subgraph_to_batchtensor_complete(substructures):
    id_lis=[]
    note_lis=[]
    type_list = []
    cur_id=0
    for  id_type,subtype in enumerate(substructures):
        for data in subtype:
            id_lis=id_lis+[cur_id]*len(data)
            type_list=type_list+[int(id_type)]*len(data)
            note_lis=note_lis+list(data)
            cur_id += 1

    return note_lis,id_lis,type_list


def _prepare_process(data, subgraph_dicts, subgraph_params,ex_fn, cnt_fn):
    if data.edge_index.shape[1] == 0 and cnt_fn.__name__ == 'subgraph_isomorphism_edge_counts':
        setattr(data, 'identifiers', torch.zeros((0, sum(orbit_partition_sizes))).long())
    else:
        new_data = ex_fn(cnt_fn, data, subgraph_dicts, subgraph_params)

    return new_data


def substructure_to_gt(subgraph_params,automorphism_fn):
    subgraph_dicts = []
    if 'edge_list' not in subgraph_params:
        raise ValueError('Edge list not provided.')
    for edge_list in subgraph_params['edge_list']:
        subgraph, orbit_partition, orbit_membership, aut_count = \
            automorphism_fn(edge_list=edge_list,
                            only_graph=True,
                            directed=subgraph_params['directed'],
                            directed_orbits=subgraph_params['directed_orbits'])
        subgraph_dicts.append({'subgraph': subgraph})
    return subgraph_dicts






def load_dataset(data_file):
    '''
        Loads dataset from `data_file`.
    '''
    print("Loading dataset from {}".format(data_file))
    dataset_obj = torch.load(data_file)
    graphs_ptg = dataset_obj[0]
    num_classes = dataset_obj[1]
    orbit_partition_sizes = dataset_obj[2]

    return graphs_ptg, num_classes, orbit_partition_sizes


def try_downgrading(data_folder, id_type, induced, directed_orbits, k, k_min):
    '''
        Extracts the substructures of size up to the `k`, if a collection of substructures
        with size larger than k has already been computed.
    '''
    found_data_filename, k_found = find_id_filename(data_folder, id_type, induced, directed_orbits, k)
    if found_data_filename is not None:
        graphs_ptg, num_classes, orbit_partition_sizes = load_dataset(found_data_filename)
        print("Downgrading k from dataset {}...".format(found_data_filename))
        graphs_ptg, orbit_partition_sizes = downgrade_k(graphs_ptg, k, orbit_partition_sizes, k_min)
        return True, graphs_ptg, num_classes, orbit_partition_sizes
    else:
        return False, None, None, None


def find_id_filename(data_folder, id_type, induced, directed_orbits, k):
    '''
        Looks for existing precomputed datasets in `data_folder` with counts for substructure 
        `id_type` larger `k`.
    '''
    if induced:
        if directed_orbits:
            pattern = os.path.join(data_folder, '{}_induced_directed_orbits_[0-9]*.pt'.format(id_type))
        else:
            pattern = os.path.join(data_folder, '{}_induced_[0-9]*.pt'.format(id_type))
    else:
        if directed_orbits:
            pattern = os.path.join(data_folder, '{}_directed_orbits_[0-9]*.pt'.format(id_type))
        else:
            pattern = os.path.join(data_folder, '{}_[0-9]*.pt'.format(id_type))
    filenames = glob.glob(pattern)
    for name in filenames:
        k_found = int(re.findall(r'\d+', name)[-1])
        if k_found >= k:
            return name, k_found
    return None, None

def downgrade_k(dataset, k, orbit_partition_sizes, k_min):
    '''
        Donwgrades `dataset` by keeping only the orbits of the requested substructures.
    '''
    feature_vector_size = sum(orbit_partition_sizes[0:k-k_min+1])
    graphs_ptg = list()
    for data in dataset:
        new_data = Data()
        for attr in data.__iter__():
            name, value = attr
            setattr(new_data, name, value)
        setattr(new_data, 'identifiers', data.identifiers[:, 0:feature_vector_size])
        graphs_ptg.append(new_data)
    return graphs_ptg, orbit_partition_sizes[0:k-k_min+1]
    
    