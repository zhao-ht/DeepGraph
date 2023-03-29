
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

import os

import sys
from os import path

from multiprocessing import Pool

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import logging
import lmdb

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

def f(x):
    return x * x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))



parser = options.get_training_parser()

res_lis = []
args = options.parse_args_and_arch(parser, modify_parser=None)
logger = logging.getLogger(__name__)

checkpoint_path = None
logger.info(f"evaluating checkpoint file {checkpoint_path}")




cfg = convert_namespace_to_omegaconf(args)
np.random.seed(cfg.common.seed)
utils.set_torch_seed(cfg.common.seed)

# initialize task
task = tasks.setup_task(cfg.task)
model = task.build_model(cfg.model)


split='inner'
task.load_dataset(split)
batch_iterator = task.get_batch_iterator(
    dataset=task.dataset(split),
    max_tokens=cfg.dataset.max_tokens_valid,
    max_sentences=cfg.dataset.batch_size_valid,
    max_positions=utils.resolve_max_positions(
        task.max_positions(),
        model.max_positions(),
    ),
    ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
    seed=cfg.common.seed,
    num_workers=cfg.dataset.num_workers,
    epoch=0,
    data_buffer_size=cfg.dataset.data_buffer_size,
    disable_iterator_cache=False,
)
itr = batch_iterator.next_epoch_itr(
    shuffle=False, set_dataset_epoch=False
)
progress = progress_bar.progress_bar(
    itr,
    log_format=cfg.common.log_format,
    log_interval=cfg.common.log_interval,
    default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple")
)


path_lmdb = os.path.join(task.dm.dataset.args['data_dir'],task.dm.dataset.args['lmdb_root_dir'],task.dm.dataset.args['transform_cache_path']+ task.dm.dataset.args['transform_dir'])
print('saving to ', path_lmdb)

if not os.path.exists(path_lmdb):
    os.makedirs(path_lmdb)
lmdb_env = lmdb.open(path_lmdb, map_size=1e12)
txn = lmdb_env.begin(write=True)



for data in progress:

    idx=data["net_input"]["batched_data"]['idx']
    subgraph_tensor_res=data["net_input"]["batched_data"]['subgraph_tensor_res']
    sorted_adj_res=data["net_input"]["batched_data"]['sorted_adj_res']

    for i in range(len(idx)):
        id_cur=idx[i]
        subgraph_tensor = subgraph_tensor_res[i]
        sorted_adj = sorted_adj_res[i]
        assert len(subgraph_tensor)==args.transform_cache_number
        if len(subgraph_tensor)>1:
            assert subgraph_tensor[0] != subgraph_tensor[1]
        for sample_id in range(args.transform_cache_number):
            result = [subgraph_tensor[sample_id], sorted_adj[sample_id]]
            txn.put(key=(str(id_cur) + '_' + str(sample_id)).encode(), value=str(result).encode())


print('********************************commit substructure sampling caching **************************')
txn.commit()
lmdb_env.close()




