# Are More Layers Beneficial to Graph Transformers?


## Introduction

This is the code of our work [Are More Layers Beneficial to Graph Transformers?](https://openreview.net/pdf?id=uagC-X9XMi8) published on ICLR 2023.
<div align=center>
<img src="https://github.com/zhao-ht/DeeoGraph/blob/master/overview.png" width="600px">
</div>




## Installation
To run DeepGraph, please clone the repository to your local machine and install the required dependencies using the script provided.
#### Note
Please note that we use CUDA 10.2 and python 3.7. If you are using a different version of CUDA or python, please adjust the package version as necessary.
#### Environment
```
conda create -n DeepGraph python=3.7

source activate DeepGraph

pip3 install torch==1.9.0+cu102 torchaudio torchvision  -f https://download.pytorch.org/whl/cu102/torch_stable.html --user

wget https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl 
pip install torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl  
pip install torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl 
pip install torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl 

pip install torch-geometric==1.7.2

conda install -c conda-forge/label/cf202003 graph-tool

pip install lmdb
pip install tensorboardX==2.4.1
pip install ogb==1.3.2
pip install rdkit-pypi==2021.9.3
pip install dgl==0.7.2 -f https://data.dgl.ai/wheels/repo.html
pip install tqdm
pip install wandb
pip install networkx
pip install setuptools==59.5.0
pip install multiprocess

git clone -b 0.12.2-release https://github.com/facebookresearch/fairseq
cd fairseq
pip install ./
python setup.py build_ext --inplace
cd ..

```


## Run the Application

We provide the training scripts for ZINC, CLUSTER, PATTERN and PCQM4M-LSC. 


####Script for ZINC
```
CUDA_VISIBLE_DEVICES=0 python train.py --user-dir ./deepgraph --save-dir ckpts/zinc --ddp-backend=legacy_ddp --dataset-name zinc --dataset-source pyg --data-dir dataset/ --task graph_prediction_substructure --id-type cycle_graph+path_graph+star_graph+k_neighborhood  --ks [8,4,6,2]   --sampling-redundancy 6     --valid-on-test --criterion l1_loss --arch graphormer_slim --deepnorm --num-classes 1 --num-workers 16 --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 --optimizer adam --adam-betas '(0.9,0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.01 --lr-scheduler polynomial_decay --power 1 --warmup-updates 640000 --total-num-update 2560000 --lr 2e-4 --end-learning-rate 1e-6 --batch-size 16 --fp16 --data-buffer-size 20 --encoder-layers 48 --encoder-embed-dim 80 --encoder-ffn-embed-dim 80 --encoder-attention-heads 8 --max-epoch 10000 --keep-best-checkpoints 2 --keep-last-epochs 3  

```

####Script for CLUSTER

```
CUDA_VISIBLE_DEVICES=0  python train.py   --user-dir ./deepgraph --save-dir ckpts/CLUSTER  --ddp-backend=legacy_ddp --dataset-name CLUSTER --dataset-source pyg --node-level-task  --data-dir dataset/ --task graph_prediction_substructure  --id-type random_walk  --ks [10]    --sampling-redundancy 2   --valid-on-test --criterion node_multiclass_cross_entropy  --arch graphormer_slim  --deepnorm   --num-classes 6    --num-workers 16    --attention-dropout 0.4 --act-dropout 0.4 --dropout 0.4     --optimizer adam --adam-betas '(0.9,0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.01     --lr-scheduler polynomial_decay --power 1 --warmup-updates 300 --total-num-update 40000     --lr 5e-4 --end-learning-rate 5e-4     --batch-size 64    --fp16     --data-buffer-size 20     --encoder-layers 48     --encoder-embed-dim 48     --encoder-ffn-embed-dim 96     --encoder-attention-heads 8   --max-nodes 1024  --max-epoch 10000 --keep-best-checkpoints 2 --keep-last-epochs 3  --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric 

```

####Script for PATTERN

```
CUDA_VISIBLE_DEVICES=0 python train.py   --user-dir ./deepgraph --save-dir ckpts/PATTERN  --ddp-backend=legacy_ddp     --dataset-name PATTERN     --dataset-source pyg --node-level-task  --data-dir dataset/  --task graph_prediction_substructure --id-type random_walk   --ks [10]   --sampling-redundancy 2   --valid-on-test  --criterion node_multiclass_cross_entropy --arch graphormer_slim  --deepnorm  --num-classes 2    --num-workers 16    --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0     --optimizer adam --adam-betas '(0.9,0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.01     --lr-scheduler polynomial_decay --power 1 --warmup-updates 6000 --total-num-update 40000     --lr 2e-4 --end-learning-rate 1e-6     --batch-size 64 --fp16     --data-buffer-size 20     --encoder-layers 48     --encoder-embed-dim 80     --encoder-ffn-embed-dim 80     --encoder-attention-heads 8   --max-nodes 512  --max-epoch 10000 --keep-best-checkpoints 2 --keep-last-epochs 3 --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric  

```

####Script for PCQM4M
```
CUDA_VISIBLE_DEVICES=0 python train.py --user-dir ./deepgraph --save-dir  ckpts/pcqm4m  --ddp-backend=legacy_ddp --dataset-name pcqm4m --dataset-source ogb --data-dir dataset/  --task graph_prediction_substructure  --id-type cycle_graph+path_graph+star_graph+k_neighborhood --ks [8,4,6,2]  --sampling-redundancy 6   --criterion l1_loss --arch graphormer_base --deepnorm --num-classes 1 --num-workers 16  --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.01 --lr-scheduler polynomial_decay --power 1 --warmup-updates 50000 --total-num-update 500000 --lr 1e-4 --end-learning-rate 1e-5 --batch-size 100 --fp16 --data-buffer-size 20 --encoder-layers 48 --encoder-embed-dim 768 --encoder-ffn-embed-dim 768 --encoder-attention-heads 32 --max-epoch 10000 --keep-best-checkpoints 2 --keep-last-epochs 3 

```

## Guidence for Further Development

Our code consists of several folds, and it is compatible with other Fairseq-based frameworks. You can easily integrate DeepGraph with other Fairseq frameworks by merging the folders. Additionally, you can apply further development to our code by adding corresponding modules.
```
deepgraph
├─criterions
├─data
│  ├─ogb_datasets
│  ├─pyg_datasets
│  ├─subsampling
│  └─substructure_dataset_utils
├─evaluate
├─models
├─modules
├─pretrain
└─tasks
```


##Citation

Please kindly cite this paper if our work is useful:
```
@inproceedings{zhaomore,
  title={Are More Layers Beneficial to Graph Transformers?},
  author={Zhao, Haiteng and Ma, Shuming and Zhang, Dongdong and Deng, Zhi-Hong and Wei, Furu},
  booktitle={International Conference on Learning Representations}
}
```

