import sys
from rdkit import Chem
import graph_tool
import torch_geometric
import joblib
from ogb.graphproppred import PygGraphPropPredDataset
import warnings

warnings.filterwarnings('ignore')
# import fairseq
# from fairseq.distributed import utils as distributed_utils
# from fairseq.logging import meters, metrics, progress_bar  # noqa

# from fairseq.logging import metrics

# sys.modules["fairseq.distributed_utils"] = distributed_utils
# sys.modules["fairseq.meters"] = meters
# sys.modules["fairseq.metrics"] = metrics
# sys.modules["fairseq.progress_bar"] = progress_bar
from fairseq_cli.train import cli_main
import logging
logging.getLogger().setLevel(logging.INFO)



if __name__ == "__main__":
    cli_main()