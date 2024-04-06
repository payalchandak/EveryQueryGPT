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

for q in EVAL_CODES: 
    pyd_config = PytorchDatasetConfig(
    save_dir=dataset_dir,
    max_seq_len=256,
    train_subset_size=0.001,
    train_subset_seed=79163,
    do_include_start_time_min=True,
    fixed_code_mode=True,
    fixed_code=q,
    fixed_time_mode=True, 
    fixed_time={'duration':365, 'offset':0}
)
    code = pyd_config.sample_code()
    if code['has_value']:
        print(f"{q['name']} with range {(q['range_min'], q['range_max'])}")
    else: 
        print(f"{q['name']}")
    print(f"observation freq ~ {code['obs_freq']:.0e}\n")
    break 

# add fixed time mode in addition to static query mode

# codes = pyd_config._all_query_codes
# diagnoses = [c for c in codes if c['type']=='diagnosis_name']
# print(set([c['name'] for c in diagnoses if 'Androgenic alopecia' in c['name']]))

pyd = PytorchDataset(
    config=pyd_config, 
    split='train'
)

# print(pyd.frac_future_is_observed)
# pr = pyd._build_population_rates()

pyd_config.set_to_dataset(pyd)

print(pyd_config.sample_code())

# print(pyd[0])

# do in post init if time-specific query is defined 
# counter = 0
# for datapoint in pyd: 
#     inputs, query, answer, flag = datapoint 
#     counter += flag
# valid_frac = counter/len(pyd)
# print(valid_frac)

# have a setting in pyd_config to control whether to filter [depends on if you mention speciifc time in your query]

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