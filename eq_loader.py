import os, ipdb
import rootutils
import sys
root = rootutils.setup_root(os.path.abspath(""), dotenv=True, pythonpath=True, cwd=False)
sys.path.append(os.environ["EVENT_STREAM_PATH"])
import numpy as np
import torch
import lightning as L
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

from EventStream.data.eval_queries import EVAL_CODES, HEART_CODES, OTHER_CODES
from EventStream.data.dataset_polars import Dataset
from EventStream.data.config import PytorchDatasetConfig
from EventStream.data.types import PytorchBatch
from EventStream.data.pytorch_dataset import PytorchDataset
from EventStream.tasks.profile import add_tasks_from

import pandas as pd, seaborn as sns, matplotlib.pyplot as plt


COHORT_NAME = "ESD_09-01-23-1"
TASK_NAME = "readmission_30d_all"
PROJECT_DIR = Path(os.environ["PROJECT_DIR"])
dataset_dir = f"/storage/shared/mgh-hf-dataset/processed/{COHORT_NAME}" # PROJECT_DIR / "data" / COHORT_NAME

codes = EVAL_CODES
heart_answers = {code['name']:[] for code in codes}
for code in codes: 
    L.seed_everything(1)
    pyd_config = PytorchDatasetConfig(
        save_dir=dataset_dir,
        max_seq_len=256,
        train_subset_size=0.1,
        train_subset_seed=79163,
        do_include_start_time_min=True,
        fixed_code_mode=True,
        fixed_code=code,
        fixed_time_mode=True, 
        fixed_time={'duration':365, 'offset':0}
    )   
    pyd = PytorchDataset(
    config=pyd_config, 
    split='train'
    )
    print(len(pyd))
    pyd_config.set_to_dataset(pyd)
    for x in pyd:
        heart_answers[code['name']].append(x['answer'])

df = pd.DataFrame(heart_answers)
corr_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Answers")
plt.savefig(f"tmp/corr.png")
plt.close()

ipdb.set_trace()

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