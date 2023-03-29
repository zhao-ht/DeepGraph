# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
import graph_tool
from torch_geometric.datasets import *
from torch_geometric.data import Dataset
from ..substructure_dataset import SubstructureDataset
import torch.distributed as dist
import torch

class MyQM7b(QM7b):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyQM9(QM9):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).process()
        if dist.is_initialized():
            dist.barrier()

class MyZINC(ZINC):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).process()
        if dist.is_initialized():
            dist.barrier()

class MYZINC_UNI(torch.utils.data.Dataset):
    def __init__(self,data_list):
        self.data_list=data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data=self.data_list[idx]
        return data


class MyMoleculeNet(MoleculeNet):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).process()
        if dist.is_initialized():
            dist.barrier()




class PYGDatasetLookupTable:
    @staticmethod
    def GetPYGDataset(dataset_spec: str, seed: int,args=None,**kwargs) -> Optional[Dataset]:
        split_result = dataset_spec.split(":")
        if len(split_result) == 2:
            name, params = split_result[0], split_result[1]
            params = params.split(",")
        elif len(split_result) == 1:
            name = dataset_spec
            params = []
        inner_dataset = None
        num_class = 1

        train_set = None
        valid_set = None
        test_set = None

        root = "dataset"
        if name == "qm7b":
            inner_dataset = MyQM7b(root=kwargs['data_dir'])
        elif name == "qm9":
            inner_dataset = MyQM9(root=kwargs['data_dir'])
        elif name == "zinc_full":
            inner_dataset = MyZINC(root=kwargs['data_dir'],subset=False)
            train_set = MyZINC(root=kwargs['data_dir'],subset=False, split="train")
            valid_set = MyZINC(root=kwargs['data_dir'],subset=False, split="val")
            test_set = MyZINC(root=kwargs['data_dir'],subset=False, split="test")
        elif name == "zinc":
            inner_dataset = MyZINC(root=kwargs['data_dir'],subset=True)
            train_set = MyZINC(root=kwargs['data_dir'],subset=True, split="train")
            valid_set = MyZINC(root=kwargs['data_dir'],subset=True, split="val")
            test_set = MyZINC(root=kwargs['data_dir'],subset=True, split="test")
        elif name == "zinc_val_on_test":
            inner_dataset = MyZINC(root=kwargs['data_dir'],subset=True)
            train_set = MyZINC(root=kwargs['data_dir'],subset=True, split="train")
            valid_set = MyZINC(root=kwargs['data_dir'],subset=True, split="test")
            test_set = MyZINC(root=kwargs['data_dir'],subset=True, split="test")
        elif name == 'zinc_uniform':
            inner_dataset = MyZINC(root=kwargs['data_dir'],subset=True)
            train_set = MyZINC(root=kwargs['data_dir'],subset=True, split="train")
            valid_set = MyZINC(root=kwargs['data_dir'],subset=True, split="val")
            test_set = MyZINC(root=kwargs['data_dir'],subset=True, split="test")
            total_data=[]
            for i in range(len(train_set)):
                total_data.append(train_set[i])
            for i in range(len(valid_set)):
                total_data.append(valid_set[i])
            for i in range(len(test_set)):
                total_data.append(test_set[i])
            train_data,valid_data,test_data=torch.utils.data.random_split(total_data, [len(train_set), len(valid_set),len(test_set)])
            train_set=MYZINC_UNI(train_data)
            valid_set = MYZINC_UNI(valid_data)
            test_set = MYZINC_UNI(test_data)
            inner_dataset=train_set
        elif name == 'zinc_uniform_val_on_test':
            inner_dataset = MyZINC(root=kwargs['data_dir'],subset=True)
            train_set = MyZINC(root=kwargs['data_dir'],subset=True, split="train")
            valid_set = MyZINC(root=kwargs['data_dir'],subset=True, split="val")
            test_set = MyZINC(root=kwargs['data_dir'],subset=True, split="test")
            total_data=[]
            for i in range(len(train_set)):
                total_data.append(train_set[i])
            for i in range(len(valid_set)):
                total_data.append(valid_set[i])
            for i in range(len(test_set)):
                total_data.append(test_set[i])
            train_data,valid_data,test_data=torch.utils.data.random_split(total_data, [len(train_set), len(valid_set),len(test_set)])
            train_set=MYZINC_UNI(train_data)
            valid_set = MYZINC_UNI(test_data)
            test_set = MYZINC_UNI(test_data)
            inner_dataset=train_set
        elif name == "moleculenet":
            nm = None
            for param in params:
                name, value = param.split("=")
                if name == "name":
                    nm = value
            inner_dataset = MyMoleculeNet(root=kwargs['data_dir'], name=nm)

        elif name in ["CLUSTER","PATTERN"]:
            train_set = GNNBenchmarkDataset(root=kwargs['data_dir'], name=name, split='train')
            valid_set = GNNBenchmarkDataset(root=kwargs['data_dir'], name=name, split='val')
            test_set = GNNBenchmarkDataset(root=kwargs['data_dir'], name=name, split='test')

        else:
            raise ValueError(f"Unknown dataset name {name} for pyg source.")

        if args['valid_on_test']:
            valid_set = test_set

        if train_set is not None:
            result= SubstructureDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                args['not_re_define'],
                args=args
                )
        else:
            result= (
                None
                if inner_dataset is None
                else SubstructureDataset(inner_dataset, seed)
            )

        return result
