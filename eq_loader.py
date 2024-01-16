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


OBS_FREQ_NUM_BUCKETS = 4

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

pyd = PytorchDataset(config=pyd_config, split='train')


def sample_code(): 
    codes = []
    vocab = Dataset.load(Path(dataset_dir)).unified_vocabulary_idxmap
    for key, cfg in pyd.measurement_configs.items(): 
        has_value = 'regression' in cfg.modality
        ofoc = cfg.observation_rate_over_cases
        ofpc = cfg.observation_rate_per_case
        for code_name, code_idx in vocab[key].items(): 
            if code_name=="UNK": continue 
            if '__EQ_' in code_name: has_value = False
            if cfg.vocabulary is None: 
                print('VOCAB MISSING FOR ',code_name, code_idx) # ERROR !! 
                continue 
            vocab_obs_freq = cfg.vocabulary.obs_frequencies[cfg.vocabulary[code_name]]
            obs_freq = ofoc * ofpc * vocab_obs_freq
            codes.append( (code_name, code_idx, has_value, obs_freq) ) 

    buckets = [(0,1e-5)]+[(10**x,10**(x+1)) for x in range(-5,0)]
    obs_freq_start, obs_freq_end = random.choice(buckets)
    codes_in_bucket = [code for code in codes if (code[-1] >= obs_freq_start) and (code[-1] <= obs_freq_end)]
    code_name, code_idx, has_value, obs_freq = random.choice(codes_in_bucket)
    return code_name, code_idx, has_value

sample_code()
# collate query and answer "_static_and_dynamic_collate"

# data/data_embedding_layer.py
# data embedding layer to reuse 
# CIPPT -> encoder -> input_layer -> data_embedding_layer 
# data_embedding_layer has several "modes", check to use one...  

print(f"Dataset has {len(pyd)} rows")
inputs, query, ans = pyd[0]
print('context',inputs.keys())
print('query',query)
print('answer',ans)