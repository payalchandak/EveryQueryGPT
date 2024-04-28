import os
import rootutils
import sys
root = rootutils.setup_root(os.path.abspath(""), dotenv=True, pythonpath=True, cwd=False)
sys.path.append(os.environ["EVENT_STREAM_PATH"])
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
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from EventStream.data.eval_queries import EVAL_CODES
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
    fixed_code_mode=False,
    # fixed_code=q,
    fixed_time_mode=True, 
    fixed_time={'duration':365, 'offset':0}
)
    
pyd = PytorchDataset(
    config=pyd_config, 
    split='train'
)

pyd_config.set_to_dataset(pyd)

i = 0
for x in pyd:
    i+=1 
    print(x['query']['code_name'])
    if i>1000: break

'''
Genetic counseling
Persistent vomiting
Bitten by dog, subsequent encounter
Blood typing; ABO
Ehlers-Danlos syndrome
Lupus anticoagulant syndrome
Screening for malignant neoplasms of colon
Burn (any degree) involving 20-29 percent of body surface with third degree burn of 20-29%
Flat foot
'''

# print(pyd.frac_future_is_observed)
# pr = pyd._build_population_rates()


# dataloader = torch.utils.data.DataLoader(
#     pyd,
#     batch_size=8,
#     num_workers=8,
#     collate_fn=pyd.collate,
#     shuffle=True,
# )

# for batch in dataloader: 
#     # inputs, query, answer = batch 
#     break

# print(batch)