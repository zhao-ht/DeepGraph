import torch
import sys
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, MinMaxScaler, StandardScaler

import numpy as np
from tqdm import tqdm


def find_max_id(dataset_list):
    atom_id_max = 0
    edge_id_max = 0
    for dataset in dataset_list:
        for id,graph in tqdm(enumerate(dataset)):
            try:
                atom_id_max = max(atom_id_max, graph.x.max())
                edge_id_max = max(edge_id_max, graph.edge_attr.max())
            except:
                pass

    return int(atom_id_max),int(edge_id_max)


class one_hot_unique:
    
    def __init__(self, tensor_list, **kwargs):
        tensor_list = torch.cat(tensor_list, 0)
        self.d = list()
        self.corrs = dict()
        for col in range(tensor_list.shape[1]):
            uniques, corrs = np.unique(tensor_list[:, col], return_inverse=True, axis=0)
            self.d.append(len(uniques))
            self.corrs[col] = corrs
        return       
            
    def fit(self, tensor_list):
        pointer = 0
        encoded_tensors = list()
        for tensor in tensor_list:
            n = tensor.shape[0]
            for col in range(tensor.shape[1]):
                translated = torch.LongTensor(self.corrs[col][pointer:pointer+n]).unsqueeze(1)
                encoded = torch.cat((encoded, translated), 1) if col > 0 else translated
            encoded_tensors.append(encoded)
            pointer += n
        return encoded_tensors
        

class one_hot_max:
    
    def __init__(self, tensor_list, **kwargs):
        tensor_list = torch.cat(tensor_list,0)
        self.d = [int(tensor_list[:,i].max()+1) for i in range(tensor_list.shape[1])]
    
    def fit(self, tensor_list):
        return tensor_list


