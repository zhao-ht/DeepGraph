import networkx as nx
import torch
import numpy as np 
import time



def random_sampling(subgraphs_nodes, rate=0.5, minimum_redundancy=0, num_nodes=None):
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1
    while True:
        selected = np.random.choice(num_nodes, int(num_nodes*rate), replace=False)
        node_selected_times = torch.bincount(subgraphs_nodes[1][check_values_in_set(subgraphs_nodes[0], selected)], minlength=num_nodes)
        if node_selected_times.min() >= minimum_redundancy:
            # rate += 0.1 # enlarge the sampling rate 
            break
    return selected, node_selected_times


def random_sampling_general(subgraphs_nodes, rate=0.5, minimum_redundancy=2,num_nodes=None,must_select_list=None):

    subgraphs_nodes_mask=transfer_edge_index_to_mask(subgraphs_nodes,num_nodes)   # number_subgraph* number_nodes True when node j in subgraph i



    node_selected_times = torch.zeros(num_nodes)
    selected_all = []

    if must_select_list is not None:
        for selected in must_select_list:
            try:
                selected_all.append(selected)
                node_selected_times[subgraphs_nodes_mask[selected]] += 1
            except:
                print('selected ',selected)
                print('num_nodes ',num_nodes)
                print('subgraphs_nodes_mask',subgraphs_nodes_mask)
                raise ValueError("error ")

    num_subgraphs = int(subgraphs_nodes[0].max())+1
    init_rate = min(subgraphs_nodes.shape[0]/num_subgraphs,1)
    rate=min(init_rate*minimum_redundancy,1)

    for i in range(int(subgraphs_nodes[0].max())+1):
        selected_all = np.random.choice(num_subgraphs, int(num_subgraphs*rate), replace=False)
        node_selected_times = subgraphs_nodes_mask[selected_all].sum(0)

        if node_selected_times.min() >= minimum_redundancy or rate >=1:
            break

        rate=min(rate+init_rate,1)




    return list(selected_all), node_selected_times


# Approach 1: based on shortets path distance 
def shortest_path_sampling(edge_index, subgraphs_nodes, stride=2, minimum_redundancy=0, random_init=False, num_nodes=None):
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1
    G = nx.from_edgelist(edge_index.t().numpy())
    G.add_nodes_from(range(num_nodes))
    if random_init:
        farthest = np.random.choice(num_nodes) # here can also choose the one with highest degree
    else:
        subgraph_size = torch.bincount(subgraphs_nodes[0], minlength=num_nodes)
        farthest = subgraph_size.argmax().item()

    distance = np.ones(num_nodes)*1e10
    selected = []
    node_selected_times = torch.zeros(num_nodes)

    for i in range(num_nodes):
        selected.append(farthest)
        node_selected_times[subgraphs_nodes[1][subgraphs_nodes[0] == farthest]] += 1
        length_shortest_dict = nx.single_source_shortest_path_length(G, farthest)
        length_shortest = np.ones(num_nodes)*1e10
        length_shortest[list(length_shortest_dict.keys())] = list(length_shortest_dict.values())
        mask = length_shortest < distance
        distance[mask] = length_shortest[mask]
        
        if (distance.max() < stride) and (node_selected_times.min() >= minimum_redundancy): # stop criterion 
            break
        farthest = np.argmax(distance)
    return selected, node_selected_times


# Approach 1: based on shortets path distance
def shortest_path_sampling_general(edge_index, subgraphs_nodes, stride=2, minimum_redundancy=0, random_init=False,
                           num_nodes=None):
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1
    G = nx.from_edgelist(edge_index.t().numpy())
    G.add_nodes_from(range(num_nodes))
    if random_init:
        farthest = np.random.choice(num_nodes)  # here can also choose the one with highest degree
    else:
        subgraph_size = torch.bincount(subgraphs_nodes[0], minlength=num_nodes)
        farthest = subgraph_size.argmax().item()

    distance = np.ones(num_nodes) * 1e10
    selected = []
    node_selected_times = torch.zeros(num_nodes)

    for i in range(num_nodes):
        selected.append(farthest)
        node_selected_times[subgraphs_nodes[1][subgraphs_nodes[0] == farthest]] += 1
        length_shortest_dict = nx.single_source_shortest_path_length(G, farthest)
        length_shortest = np.ones(num_nodes) * 1e10
        length_shortest[list(length_shortest_dict.keys())] = list(length_shortest_dict.values())
        mask = length_shortest < distance
        distance[mask] = length_shortest[mask]

        if (distance.max() < stride) and (node_selected_times.min() >= minimum_redundancy):  # stop criterion
            break
        farthest = np.argmax(distance)
    return selected, node_selected_times



def check_values_in_set(x, set, approach=1):
    assert min(x.shape) > 0
    assert min(set.shape) > 0
    if approach == 0:
        mask = sum(x==i for i in set)
    else:
        mapper = torch.zeros(max(x.max()+1, set.max()+1), dtype=torch.bool)
        mapper[set] = True
        mask = mapper[x]
    return mask

############################################### use dense input ##################################################
### this part is hard to change
 
def min_set_cover_sampling(edge_index, subgraphs_nodes_mask, random_init=False, minimum_redundancy=2):

    num_nodes = subgraphs_nodes_mask.size(0)
    if random_init:
        selected = np.random.choice(num_nodes) 
    else:
        selected = subgraphs_nodes_mask.sum(-1).argmax().item() # choose the maximum subgraph size one to remove randomness

    node_selected_times = torch.zeros(num_nodes)
    selected_all = []

    for i in range(num_nodes):
        # selected_subgraphs[selected] = True
        selected_all.append(selected)
        node_selected_times[subgraphs_nodes_mask[selected]] += 1
        if node_selected_times.min() >= minimum_redundancy: # stop criterion 
            break
        # calculate how many unused nodes in each subgraph (greedy set cover)
        unused_nodes = ~ ((node_selected_times - node_selected_times.min()).bool())
        num_unused_nodes = (subgraphs_nodes_mask & unused_nodes).sum(-1)
        scores = num_unused_nodes
        scores[selected_all] = 0
        selected = np.argmax(scores).item()

    return selected_all, node_selected_times


def transfer_edge_index_to_mask(subgraphs_nodes,num_nodes):
    res=torch.zeros([int(subgraphs_nodes[0].max()+1),num_nodes])
    res[subgraphs_nodes[0,:],subgraphs_nodes[1,:]]=1
    out=res>0
    return out

def min_set_cover_sampling_general(subgraphs_nodes, random_init=False, minimum_redundancy=2,num_nodes=None,must_select_list=None,only_unused_nodes=True):


    subgraphs_nodes_mask=transfer_edge_index_to_mask(subgraphs_nodes,num_nodes)

    node_selected_times = torch.zeros(num_nodes)
    selected_all = []

    if must_select_list is not None:
        for selected in must_select_list:
            try:
                selected_all.append(selected)
                node_selected_times[subgraphs_nodes_mask[selected]] += 1
            except:
                print('selected ',selected)
                print('num_nodes ',num_nodes)
                print('subgraphs_nodes_mask',subgraphs_nodes_mask)
                raise ValueError("error ")
    # num_nodes = subgraphs_nodes_mask.size(0)
    if random_init:
        selected = np.random.choice(int(subgraphs_nodes[0].max())+1)
    else:
        selected = subgraphs_nodes_mask.sum(-1).argmax().item() # choose the maximum subgraph size one to remove randomness

    for i in range(int(subgraphs_nodes[0].max())+1):
        # selected_subgraphs[selected] = True
        selected_all.append(selected)
        node_selected_times[subgraphs_nodes_mask[selected]] += 1
        if node_selected_times.min() >= minimum_redundancy: # stop criterion
            break
        # calculate how many unused nodes in each subgraph (greedy set cover)
        if only_unused_nodes:
            unused_nodes = ~ ((node_selected_times - node_selected_times.min()).bool())
        else:
            unused_nodes = ((node_selected_times - minimum_redundancy)<0)
        num_unused_nodes = (subgraphs_nodes_mask & unused_nodes).sum(-1)
        scores = num_unused_nodes
        scores[selected_all] = 0
        if scores.sum()==0:
            break
        selected = np.argmax(scores).item()

    return selected_all, node_selected_times

def min_set_cover_random_sampling_general(subgraphs_nodes, random_init=False, minimum_redundancy=2,num_nodes=None,must_select_list=None,must_select_list_ratio=0.2,only_unused_nodes=True,ratio_init_drop=0.5,ratio_each_iter=0.5,balance_different_substructure=True):


    # The algorithm implies:
    # At the beginning, random drop some substructures for randomness (controled by ratio_init_drop)
    # Randomly sample top K substructures (controled by ratio_each_iter)
    # The final sampled substructure number adapted to the total substructure number. Expectation is cover number plus one
    # Balanced substructure types (controled by balance_different_substructure)
    # times = []
    # times.append(time.time())

    num_subgraphs = int(subgraphs_nodes[0].max())+1
    number_each_iter = max(int(num_nodes*subgraphs_nodes.shape[0] / num_subgraphs), 1)

    subgraphs_nodes_mask=transfer_edge_index_to_mask(subgraphs_nodes,num_nodes)

    init_drop=np.random.choice(num_subgraphs, int(num_subgraphs*ratio_init_drop), replace=False)


    if balance_different_substructure:

        # It requires the geometric substructures before the neighbor substructures

        ids=subgraphs_nodes[0,subgraphs_nodes[2]==-1]
        if ids.shape[0]>0:
            number_predefined=int(ids[0])
            number_neighbor = num_subgraphs-number_predefined
            if number_predefined>number_neighbor:
                predefined_drop=np.random.choice(number_predefined,(number_predefined-number_neighbor), replace=False)
                init_drop=np.concatenate([init_drop,predefined_drop],0)



    node_selected_times = torch.zeros(num_nodes)
    selected_all = []

    if must_select_list is not None:
        try:
            number_must_select=max(int(num_nodes*must_select_list_ratio),1)
            if len(must_select_list)>=number_must_select:
                must_select_list=[must_select_list[i] for i in np.random.choice(len(must_select_list), number_must_select, replace=False)]
            selected_all += must_select_list
            node_selected_times += subgraphs_nodes_mask[must_select_list].sum(0)
        except:
            print('selected ',must_select_list)
            print('num_nodes ',num_nodes)
            print('subgraphs_nodes_mask',subgraphs_nodes_mask)
            raise ValueError("error ")
    # num_nodes = subgraphs_nodes_mask.size(0)
    if random_init:
        selected = [np.random.choice(int(subgraphs_nodes[0].max())+1)]
    else:
        selected = [subgraphs_nodes_mask.sum(-1).argmax().item()] # choose the maximum subgraph size one to remove randomness


    for i in range(int(subgraphs_nodes[0].max())+1):
        # selected_subgraphs[selected] = True
        selected_all+=selected
        node_selected_times += subgraphs_nodes_mask[selected].sum(0)
        if node_selected_times.min() >= minimum_redundancy: # stop criterion
            break
        # calculate how many unused nodes in each subgraph (greedy set cover)
        if only_unused_nodes:
            unused_nodes = ~ ((node_selected_times - node_selected_times.min()).bool())
        else:
            unused_nodes = ((node_selected_times - minimum_redundancy)<0)
        num_unused_nodes = (subgraphs_nodes_mask & unused_nodes).sum(-1)
        scores = num_unused_nodes
        scores[selected_all] = 0

        scores[init_drop] = 0

        if scores.sum()==0 :
            break
        number_lefted=int((scores>0).sum())
        _,index=torch.topk(scores, min(int(number_each_iter/ratio_each_iter),number_lefted))
        selected=index[np.random.choice(len(index), min(number_each_iter,number_lefted), replace=False)].tolist()


    return selected_all, node_selected_times

