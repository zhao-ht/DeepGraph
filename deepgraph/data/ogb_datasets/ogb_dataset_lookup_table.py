# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Dataset
from ..substructure_dataset import SubstructureDataset
import torch.distributed as dist
import os
import torch
from torch_scatter import scatter
import numpy as np
from torchvision import transforms



class MyPygPCQM4Mv2Dataset(PygPCQM4Mv2Dataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPygPCQM4Mv2Dataset, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPygPCQM4Mv2Dataset, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyPygPCQM4MDataset(PygPCQM4MDataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPygPCQM4MDataset, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPygPCQM4MDataset, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyPygGraphPropPredDataset(PygGraphPropPredDataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPygGraphPropPredDataset, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPygGraphPropPredDataset, self).process()
        if dist.is_initialized():
            dist.barrier()



def extract_node_feature(data, reduce='add'):
    if reduce in ['mean', 'max', 'add']:
        data.x = scatter(data.edge_attr,
                         data.edge_index[0],
                         dim=0,
                         dim_size=data.num_nodes,
                         reduce=reduce)
    else:
        raise Exception('Unknown Aggregation Type')
    return data


def get_vocab_mapping(seq_list, num_vocab):
    '''
        Input:
            seq_list: a list of sequences
            num_vocab: vocabulary size
        Output:
            vocab2idx:
                A dictionary that maps vocabulary into integer index.
                Additioanlly, we also index '__UNK__' and '__EOS__'
                '__UNK__' : out-of-vocabulary term
                '__EOS__' : end-of-sentence
            idx2vocab:
                A list that maps idx to actual vocabulary.
    '''

    vocab_cnt = {}
    vocab_list = []
    for seq in seq_list:
        for w in seq:
            if w in vocab_cnt:
                vocab_cnt[w] += 1
            else:
                vocab_cnt[w] = 1
                vocab_list.append(w)

    cnt_list = np.array([vocab_cnt[w] for w in vocab_list])
    topvocab = np.argsort(-cnt_list, kind = 'stable')[:num_vocab]

    print('Coverage of top {} vocabulary:'.format(num_vocab))
    print(float(np.sum(cnt_list[topvocab]))/np.sum(cnt_list))

    vocab2idx = {vocab_list[vocab_idx]: idx for idx, vocab_idx in enumerate(topvocab)}
    idx2vocab = [vocab_list[vocab_idx] for vocab_idx in topvocab]

    # print(topvocab)
    # print([vocab_list[v] for v in topvocab[:10]])
    # print([vocab_list[v] for v in topvocab[-10:]])

    vocab2idx['__UNK__'] = num_vocab
    idx2vocab.append('__UNK__')

    vocab2idx['__EOS__'] = num_vocab + 1
    idx2vocab.append('__EOS__')

    # test the correspondence between vocab2idx and idx2vocab
    for idx, vocab in enumerate(idx2vocab):
        assert(idx == vocab2idx[vocab])

    # test that the idx of '__EOS__' is len(idx2vocab) - 1.
    # This fact will be used in decode_arr_to_seq, when finding __EOS__
    assert(vocab2idx['__EOS__'] == len(idx2vocab) - 1)

    return vocab2idx, idx2vocab


def augment_edge(data):
    '''
        Input:
            data: PyG data object
        Output:
            data (edges are augmented in the following ways):
                data.edge_index: Added next-token edge. The inverse edges were also added.
                data.edge_attr (torch.Long):
                    data.edge_attr[:,0]: whether it is AST edge (0) for next-token edge (1)
                    data.edge_attr[:,1]: whether it is original direction (0) or inverse direction (1)
    '''

    ##### AST edge
    edge_index_ast = data.edge_index
    edge_attr_ast = torch.zeros((edge_index_ast.size(1), 2))

    ##### Inverse AST edge
    edge_index_ast_inverse = torch.stack([edge_index_ast[1], edge_index_ast[0]], dim=0)
    edge_attr_ast_inverse = torch.cat(
        [torch.zeros(edge_index_ast_inverse.size(1), 1), torch.ones(edge_index_ast_inverse.size(1), 1)], dim=1)

    ##### Next-token edge

    ## Obtain attributed nodes and get their indices in dfs order
    # attributed_node_idx = torch.where(data.node_is_attributed.view(-1,) == 1)[0]
    # attributed_node_idx_in_dfs_order = attributed_node_idx[torch.argsort(data.node_dfs_order[attributed_node_idx].view(-1,))]

    ## Since the nodes are already sorted in dfs ordering in our case, we can just do the following.
    attributed_node_idx_in_dfs_order = torch.where(data.node_is_attributed.view(-1, ) == 1)[0]

    ## build next token edge
    # Given: attributed_node_idx_in_dfs_order
    #        [1, 3, 4, 5, 8, 9, 12]
    # Output:
    #    [[1, 3, 4, 5, 8, 9]
    #     [3, 4, 5, 8, 9, 12]
    edge_index_nextoken = torch.stack([attributed_node_idx_in_dfs_order[:-1], attributed_node_idx_in_dfs_order[1:]],
                                      dim=0)
    edge_attr_nextoken = torch.cat(
        [torch.ones(edge_index_nextoken.size(1), 1), torch.zeros(edge_index_nextoken.size(1), 1)], dim=1)

    ##### Inverse next-token edge
    edge_index_nextoken_inverse = torch.stack([edge_index_nextoken[1], edge_index_nextoken[0]], dim=0)
    edge_attr_nextoken_inverse = torch.ones((edge_index_nextoken.size(1), 2))

    data.edge_index = torch.cat(
        [edge_index_ast, edge_index_ast_inverse, edge_index_nextoken, edge_index_nextoken_inverse], dim=1)
    data.edge_attr = torch.cat([edge_attr_ast, edge_attr_ast_inverse, edge_attr_nextoken, edge_attr_nextoken_inverse],
                               dim=0)

    return data

def encode_seq_to_arr(seq, vocab2idx, max_seq_len):
    '''
    Input:
        seq: A list of words
        output: add y_arr (torch.Tensor)
    '''

    augmented_seq = seq[:max_seq_len] + ['__EOS__'] * max(0, max_seq_len - len(seq))
    return torch.tensor([[vocab2idx[w] if w in vocab2idx else vocab2idx['__UNK__'] for w in augmented_seq]], dtype = torch.long)


def encode_code2(data, vocab2idx, max_seq_len):
    '''
    Input:
        data: PyG graph object
        output: add y_arr to data
    '''

    # PyG >= 1.5.0
    data.x=torch.cat([data.x[:,0].unsqueeze(1),data.node_depth,data.x[:,1].unsqueeze(1)],1)
    seq = data.y

    # PyG = 1.4.3
    # seq = data.y[0]
    data.y_ori=seq
    data.y = encode_seq_to_arr(seq, vocab2idx, max_seq_len)

    return data



class OGBDatasetLookupTable:
    @staticmethod
    def GetOGBDataset(dataset_name: str, seed: int,args=None,**kwargs) -> Optional[Dataset]:

        inner_dataset = None
        train_idx = None
        valid_idx = None
        test_idx = None
        if dataset_name == "ogbg-molhiv":
            folder_name = dataset_name.replace("-", "_")
            os.system(f"mkdir -p {os.path.join(kwargs['data_dir'],folder_name)}")
            os.system(f"touch {os.path.join(kwargs['data_dir'],folder_name,'RELEASE_v1.txt')}")
            inner_dataset = MyPygGraphPropPredDataset(dataset_name,root=kwargs['data_dir'])
            idx_split = inner_dataset.get_idx_split()
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test"]
        elif dataset_name == "ogbg-molpcba":
            folder_name = dataset_name.replace("-", "_")
            os.system(f"mkdir -p {os.path.join(kwargs['data_dir'],folder_name)}")
            os.system(f"touch {os.path.join(kwargs['data_dir'],folder_name,'RELEASE_v1.txt')}")
            inner_dataset = MyPygGraphPropPredDataset(dataset_name,root=kwargs['data_dir'])
            idx_split = inner_dataset.get_idx_split()
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test"]
        elif dataset_name == "pcqm4mv2":
            os.system(f"mkdir -p {os.path.join(kwargs['data_dir'],'pcqm4m-v2')}")
            os.system(f"touch {os.path.join(kwargs['data_dir'],'pcqm4m-v2','RELEASE_v1.txt')}")
            inner_dataset = MyPygPCQM4Mv2Dataset()
            idx_split = inner_dataset.get_idx_split()
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test-dev"]
        elif dataset_name == "pcqm4m":
            os.system(f"mkdir -p {os.path.join(kwargs['data_dir'],'pcqm4m_kddcup2021')}")
            os.system(f"touch {os.path.join(kwargs['data_dir'],'pcqm4m_kddcup2021','RELEASE_v1.txt')}")
            inner_dataset = MyPygPCQM4MDataset(root=kwargs['data_dir'])
            idx_split = inner_dataset.get_idx_split()
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test"]
        elif dataset_name == 'pcqm4m_contrust_pretraining':
            os.system(f"mkdir -p {os.path.join(kwargs['data_dir'],'pcqm4m_kddcup2021')}")
            os.system(f"touch {os.path.join(kwargs['data_dir'],'pcqm4m_kddcup2021','RELEASE_v1.txt')}")
            inner_dataset = MyPygPCQM4MDataset(root=kwargs['data_dir'])
            idx_split = inner_dataset.get_idx_split()
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test"]

        elif dataset_name in ['ogbg-ppa']:
            from functools import partial
            transform = partial(extract_node_feature, reduce='add')
            folder_name = dataset_name.replace("-", "_")
            os.system(f"mkdir -p {os.path.join(kwargs['data_dir'],folder_name)}")
            os.system(f"touch {os.path.join(kwargs['data_dir'],folder_name,'RELEASE_v1.txt')}")
            inner_dataset =  MyPygGraphPropPredDataset(name=dataset_name, root=kwargs['data_dir'], transform=transform)
            idx_split = inner_dataset.get_idx_split()
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test"]

        elif dataset_name in ['ogbg-code2']:
            inner_dataset = PygGraphPropPredDataset(name=dataset_name, root=kwargs['data_dir'])
            idx_split = inner_dataset.get_idx_split()
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test"]
            split_idx = inner_dataset.get_idx_split()
            #
            # ### building vocabulary for sequence predition. Only use training data.
            vocab2idx, idx2vocab = get_vocab_mapping([inner_dataset.data.y[i] for i in split_idx['train']], 5000)
            #
            # ### set the transform function
            # # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
            # # encode_y_to_arr: add y_arr to PyG data object, indicating the array representation of a sequence.
            inner_dataset.transform = transforms.Compose([
                augment_edge, lambda data: encode_code2(data, vocab2idx, 5)
            ])

        else:
            raise ValueError(f"Unknown dataset name {dataset_name} for ogb source.")


        result=None if inner_dataset is None  else SubstructureDataset(
            inner_dataset, seed, train_idx, valid_idx, test_idx,args=args
        )

        return result
