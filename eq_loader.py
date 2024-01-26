import os
import rootutils
import sys
root = rootutils.setup_root(os.path.abspath(""), dotenv=True, pythonpath=True, cwd=False)
sys.path.append(os.environ["EVENT_STREAM_PATH"])
import os
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime, timedelta
from humanize import naturalsize, naturaldelta
from pathlib import Path
from sparklines import sparklines
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from typing import Callable
import random

from EventStream.data.dataset_polars import Dataset
from EventStream.data.config import PytorchDatasetConfig
from EventStream.data.types import PytorchBatch
from EventStream.data.pytorch_dataset import PytorchDataset
from EventStream.tasks.profile import add_tasks_from

COHORT_NAME = "ESD_09-01-23-1"
TASK_NAME = "readmission_30d_all"
PROJECT_DIR = Path(os.environ["PROJECT_DIR"])
dataset_dir = f"/storage/shared/mgh-hf-dataset/processed/{COHORT_NAME}" # PROJECT_DIR / "data" / COHORT_NAME

pyd_config = PytorchDatasetConfig(
    save_dir=dataset_dir,
    max_seq_len=256,
    train_subset_size=0.001,
    train_subset_seed=79163,
    do_include_start_time_min=True,
    #cache_for_epochs=1,
)

pyd = PytorchDataset(
    config=pyd_config, 
    split='train'
)
# inputs, query, answer = pyd[0]

dataloader = torch.utils.data.DataLoader(
    pyd,
    batch_size=8,
    num_workers=8,
    collate_fn=pyd.collate,
    shuffle=True,
)

for batch in dataloader: 
    print(batch)

# data/data_embedding_layer.py
# data embedding layer to reuse 
# CIPPT -> encoder -> input_layer -> data_embedding_layer 
# data_embedding_layer has several "modes", check to use one...  
# ESGPT train -> define LM -> line 109 in LM to initialize model -> encoder 