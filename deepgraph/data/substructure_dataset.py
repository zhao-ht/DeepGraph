# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch_geometric.data import Dataset
from sklearn.model_selection import train_test_split

import torch
import numpy as np

from .wrapper import preprocess_item,graph_data_modification_single,encode_token_single_tensor,encode_token_single_tensor_with_adj


from deepgraph.data.substructure_dataset_utils.substructure_transform import transform_sub

import copy
import lmdb
import os

import torch.distributed as dist


class concat_dataset:
    def __init__(self,dataset_list):
        self.dataset_list=dataset_list
    def __getitem__(self,idx):
        if idx<len(self.dataset_list[0]):
            return self.dataset_list[0][idx]
        elif idx<len(self.dataset_list[0])+len(self.dataset_list[1]):
            return self.dataset_list[1][idx-len(self.dataset_list[0])]
        else:
            return self.dataset_list[2][idx-len(self.dataset_list[0])-len(self.dataset_list[1])]
    def __len__(self):
        return len(self.dataset_list[0])+len(self.dataset_list[1])+len(self.dataset_list[2])

class recompute_idx():
    def __init__(self,size_train,size_valid,subset):
        self.subset = subset
        self.size_train=size_train
        self.size_valid=size_valid
    def compute(self,idx):
        if self.subset == 'train':
            return idx
        elif self.subset=='valid':
            return idx+self.size_train
        elif self.subset=='test':
            return idx + self.size_train+self.size_valid
        else:
            raise ValueError("invalid subset")

class recompute_idx_byindex():
    def __init__(self,train_idx,valid_idx,test_idx,subset):
        self.train_idx = train_idx
        self.valid_idx=valid_idx
        self.test_idx=test_idx
        self.subset=subset
    def compute(self,idx):
        if self.subset == 'train':
            return int(self.train_idx[idx])
        elif self.subset=='valid':
            return int(self.valid_idx[idx])
        elif self.subset=='test':
            return int(self.test_idx[idx])
        elif self.subset=='inner':
            return idx
        else:
            raise ValueError("invalid subset")



class SubstructureDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        seed: int = 0,
        train_idx=None,
        valid_idx=None,
        test_idx=None,
        train_set=None,
        valid_set=None,
        test_set=None,
        not_re_define=False,
        args=None
    ):
        self.args=args
        self.dataset = dataset
        self.not_re_define = not_re_define
        self.local_attention_on_substructures=args.local_attention_on_substructures if 'local_attention_on_substructures' in args else False
        self.continuous_feature=args.continuous_feature
        self.extra_method=args['extra_method'] if 'extra_method' in args else None
        if self.dataset is not None:
            self.num_data = len(self.dataset)
        self.seed = seed
        self.use_transform_cache=args['use_transform_cache'] if 'use_transform_cache' in args else False
        self.transform_cache_number=args['transform_cache_number'] if 'transform_cache_number' in args else 1
        self.only_substructure=args['recache_transform'] if 'recache_transform' in args else False

        # data transform
        self.transform=None
        if 'add_substructure' in args and args['add_substructure'] == 'transform':

            self.transform = transform_sub(args)


        # self.lmdb_tensor_env=None
        if train_idx is None and train_set is None:
            train_valid_idx, test_idx = train_test_split(
                np.arange(self.num_data),
                test_size=self.num_data // 10,
                random_state=seed,
            )
            train_idx, valid_idx = train_test_split(
                train_valid_idx, test_size=self.num_data // 5, random_state=seed
            )
            self.train_idx = torch.from_numpy(train_idx)
            self.valid_idx = torch.from_numpy(valid_idx)
            self.test_idx = torch.from_numpy(test_idx)
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
        elif train_set is not None:
            self.num_data = len(train_set) + len(valid_set) + len(test_set)
            self.train_idx = torch.Tensor(list(range(len(train_set)))).type(torch.int64)
            self.valid_idx = torch.Tensor(list(range(len(train_set),
                                                     len(train_set)+len(valid_set)))).type(torch.int64)
            self.test_idx = torch.Tensor(list(range(len(train_set)+len(valid_set),
                                                    len(train_set)+len(valid_set)+len(test_set)))).type(torch.int64)
            self.train_data = self.create_subset(train_set)
            self.valid_data = self.create_subset(valid_set)
            self.test_data = self.create_subset(test_set)
        else:
            self.num_data = len(train_idx) + len(valid_idx) + len(test_idx)
            self.train_idx = train_idx
            self.valid_idx = valid_idx
            self.test_idx = test_idx
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)

        self.__indices__ = None

        if not dist.is_initialized() or dist.get_rank() == 0:
            if self.dataset is None:
                self.dataset = concat_dataset([train_set, valid_set, test_set])


        #cache transform

        #use_lmdb_cache is set default true in GraphPredictionSubstructureConfig
        if ('use_lmdb_cache' in args and args.use_lmdb_cache):
            print('************************* Using lmdb cache **************************')

            if len(args['custom_edge_list']) > 0:
                if args.recache_lmdb or not os.path.exists(os.path.join(args['data_dir'],
                                                                        args['lmdb_root_dir'],
                                                                        args['pre_defined_path']+ args['lmdb_dir'],
                                                                        'inner')):
                    print('************************* Warning recache substructures**************************')
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        self.transform.pre_compute_substructures_direct_to_lmdb(self.dataset)

                    if dist.is_initialized():
                        dist.barrier()
            if len(args['neighbor_type']) > 0:
                if args.recache_lmdb or not os.path.exists(os.path.join(args['data_dir'],args['lmdb_root_dir'],args['node_neighbor_path']+ args['lmdb_dir'],'neighbor')):
                    print('************************* Warning recache neighbors**************************')
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        self.transform.pre_compute_neighbors_direct_to_lmdb(self.dataset)

                    if dist.is_initialized():
                        dist.barrier()



            # if self.lmdb_tensor_env is None:

            if len(args['custom_edge_list'])>0:
                self.lmdb_tensor_env=lmdb.open(os.path.join(args['data_dir'],args['lmdb_root_dir'],args['pre_defined_path']+ args['lmdb_dir'], 'inner'),map_size=1e12, readonly=True,
                        lock=False, readahead=False, meminit=False)
            else:
                self.lmdb_tensor_env=None
            if len(args['neighbor_type'])>0:
                self.lmdb_neighbor_env=lmdb.open(os.path.join(args['data_dir'],args['lmdb_root_dir'],args['node_neighbor_path']+ args['lmdb_dir'],'neighbor'),map_size=1e12, readonly=True,
                        lock=False, readahead=False, meminit=False)
                txn = self.lmdb_neighbor_env.begin(write=False)
                max_size=eval(txn.get(('max_size').encode()))
                self.transform.args['subgraph_max_size']=max(self.transform.args['subgraph_max_size'],max_size)
            else:
                self.lmdb_neighbor_env=None




            self.train_data.lmdb_tensor_env=self.lmdb_tensor_env
            self.train_data.lmdb_neighbor_env = self.lmdb_neighbor_env

            self.valid_data.lmdb_tensor_env = self.lmdb_tensor_env
            self.valid_data.lmdb_neighbor_env = self.lmdb_neighbor_env

            self.test_data.lmdb_tensor_env = self.lmdb_tensor_env
            self.test_data.lmdb_neighbor_env = self.lmdb_neighbor_env

            if train_idx is not None:
                self.recomputeind=recompute_idx_byindex(train_idx,valid_idx,test_idx,'inner')
                self.train_data.recomputeind=recompute_idx_byindex(train_idx,valid_idx,test_idx,'train')
                self.test_data.recomputeind = recompute_idx_byindex(train_idx,valid_idx,test_idx,'test')
                if args['valid_on_test']:
                    self.valid_data.recomputeind = recompute_idx_byindex(train_idx, valid_idx, test_idx, 'test')
                else:
                    self.valid_data.recomputeind = recompute_idx_byindex(train_idx, valid_idx, test_idx, 'valid')
            else:
                self.recomputeind=recompute_idx(len(self.train_data),len(self.valid_data),'train')
                self.train_data.recomputeind=recompute_idx(len(self.train_data),len(self.valid_data),'train')
                self.test_data.recomputeind = recompute_idx(len(self.train_data), len(self.valid_data), 'test')
                if args['valid_on_test']:
                    self.valid_data.recomputeind = recompute_idx(len(self.train_data), len(self.valid_data), 'test')
                else:
                    self.valid_data.recomputeind = recompute_idx(len(self.train_data), len(self.valid_data), 'valid')



            if 'use_transform_cache' in args and args.use_transform_cache:
                print('************************* Using transform cache **************************')

                transform_cache_path=os.path.join(args['data_dir'], args['lmdb_root_dir'], args['transform_cache_path'] + args['transform_dir'])
                if not os.path.exists(transform_cache_path):
                    print(transform_cache_path+' do not exits; creating**************')
                    os.makedirs(transform_cache_path)
                try:
                    self.transform_cache_env=lmdb.open(transform_cache_path,map_size=1e12, readonly=True,
                            lock=False, readahead=False, meminit=False)
                    self.train_data.transform_cache_env = self.transform_cache_env
                    self.valid_data.transform_cache_env = self.transform_cache_env
                    self.test_data.transform_cache_env = self.transform_cache_env
                except:
                    assert self.only_substructure # The only situation where use_transform_cache but no cache directory is generating cache


    def index_select(self, idx):
        dataset = copy.copy(self)
        if not isinstance(self.dataset,list):
            dataset.dataset = self.dataset.index_select(idx)
        else:
            dataset.dataset = [self.dataset[i] for i in idx.tolist()]
        if isinstance(idx, torch.Tensor):
            dataset.num_data = idx.size(0)
        else:
            dataset.num_data = idx.shape[0]
        dataset.__indices__ = idx
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None

        return dataset

    def create_subset(self, subset):
        dataset = copy.copy(self)
        dataset.dataset = subset
        dataset.num_data = len(subset)
        dataset.__indices__ = None
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None

        return dataset



    # @lru_cache(maxsize=16)
    def __getitem__(self, idx,parallel=False):

        if isinstance(idx, int):

            item = self.dataset[idx]

            if self.transform is not None:

                if (not self.use_transform_cache) or self.only_substructure:

                    if (self.lmdb_tensor_env is not None):
                        txn = self.lmdb_tensor_env.begin(write=False)
                        tem=txn.get(str(self.recomputeind.compute(idx)).encode())
                        if tem is not None:
                            substructure_tensor=eval(tem)
                        else:
                            raise ValueError("substructure_tensor not prepared")
                    else:
                        substructure_tensor=None
                    if (self.lmdb_neighbor_env is not None):
                        txn = self.lmdb_neighbor_env.begin(write=False)
                        tem=txn.get(str(self.recomputeind.compute(idx)).encode())
                        if tem is not None:
                            neighbor_tensor=eval(tem)
                        else:
                            raise ValueError("substructure_tensor not prepared")
                    else:
                        neighbor_tensor=None

                    substructure_tensor=self.transform.substructure_as_tensor(substructure_tensor,neighbor_tensor)

                    if self.only_substructure:
                        #caching #transform_cache_number sampled substructures

                        subgraph_tensor_res=[]
                        sorted_adj_res=[]

                        for i in range(self.transform_cache_number):


                            subgraph_tensor, sorted_adj = self.transform(item, substructure_tensor)


                            subgraph_tensor=subgraph_tensor.tolist()
                            sorted_adj=sorted_adj.tolist()
                            subgraph_tensor_res.append(subgraph_tensor)
                            sorted_adj_res.append(sorted_adj)

                        item.idx=idx
                        item.subgraph_tensor_res = subgraph_tensor_res
                        item.sorted_adj_res=sorted_adj_res
                        item.y = item.y.reshape(-1)
                        return item

                    else:
                        subgraph_tensor,sorted_adj = self.transform(item,substructure_tensor)

                else:
                    txn = self.transform_cache_env.begin(write=False)
                    id=np.random.randint(self.transform_cache_number)
                    tem = txn.get((str(self.recomputeind.compute(idx))+'_'+str(id)).encode())
                    if tem is not None:
                        subgraph_tensor, sorted_adj = eval(tem)
                    else:
                        raise ValueError("substructure_tensor not prepared")
                    subgraph_tensor = torch.tensor(subgraph_tensor)
                    sorted_adj = torch.tensor(sorted_adj)

                item = graph_data_modification_single(item, subgraph_tensor, sorted_adj)

                if self.extra_method == 'token':
                    item = encode_token_single_tensor(item, self.args.atom_id_max, self.args.edge_id_max, )
                elif self.extra_method == 'adj':
                    item = encode_token_single_tensor_with_adj(item, local_attention_on_substructures=self.local_attention_on_substructures)

            if (not self.not_re_define) or (not hasattr(item, 'attn_bias')):
                item.idx = idx
                item.y = item.y.reshape(-1)

                item= preprocess_item(item,local_attention_on_substructures=self.local_attention_on_substructures,continuous_feature=self.continuous_feature)
                return item

        else:
            raise TypeError("index to a GraphormerPYGDataset can only be an integer.")

    def __len__(self):
        return self.num_data

