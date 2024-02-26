import os
from cycler import V
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

queries = [
    {
        'name': 'High NT-proBNP',
        'code':'N-terminal pro-brain natriuretic peptide',
        'range':(125, 100000),
    },
    {
        'name': 'High Trop-T',
        'code':'Troponin T cardiac',
        'range':(0.05, 10),
    },
    {
        'name': 'High Creatinine',
        'code':'Creatinine',
        'range':(1.0, 10),
    },
    {
        'name': 'Low Potassium',
        'code':'Potassium',
        'range':(0, 3.5),
    },
    {
        'name': 'Normal Potassium',
        'code':'Potassium',
        'range':(3.5, 5),
    },
    {
        'name': 'High Potassium',
        'code':'Potassium',
        'range':(5, 10),
    },
    {
        'name': 'Low Sodium',
        'code':'Sodium',
        'range':(0,136),
    },
    {
        'name': 'Normal Sodium',
        'code':'Sodium',
        'range':(136,145),
    },
    {
        'name': 'High Sodium',
        'code':'Sodium',
        'range':(145,200),
    },
    {
        'name': 'High BUN',
        'code':'Urea nitrogen',
        'range':(20, 200),
    },
    {
        'name': 'Low GFR',
        'code':'Glomerular filtration rate',
        'range':(0, 60),
    },
    {
        'name': 'High HbA1c',
        'code':'Hemoglobin A1c/Hemoglobin total',
        'range':(8.5, 30),
    },
    {
        'name': 'LVEF < 40',
        'code':'lv_ef_value',
        'range':(0, 40),
    },
    {
        'name': 'LVEF > 40',
        'code':'lv_ef_value',
        'range':(41, 100),
    },
    {
        'name': 'High PCWP',
        'code':'mean_wedge_pressure',
        'range':(15, 200),
    },
    {
        'name': 'Stage 4 CKD',
        'code':'Chronic kidney disease, Stage IV (severe)',
        'range':(.0,.0),
    },
    {
        'name': 'Cardiogenic shock',
        'code':'Cardiogenic shock',
        'range':(.0,.0),
    },
    {
        'name': 'Hypertension',
        'code':'Essential (primary) hypertension',
        'range':(.0,.0),
    },
    {
        'name': 'Sepsis after procedure',
        'code':'Sepsis following a procedure, initial encounter',
        'range':(.0,.0),
    },
    {
        'name': 'Thrombus/Embolism',
        'code':'Thrombus/Embolism',
        'range':(.0,.0),
    },
    {
        'name': 'Ventricular tachycardia',
        'code':'Ventricular tachycardia',
        'range':(.0,.0),
    },
    {
        'name': 'Endocarditis',
        'code':'Endocarditis, valve unspecified, unspecified cause',
        'range':(.0,.0),
    },
    {
        'name': 'Atrial fibrillation',
        'code':'Atrial fibrillation',
        'range':(.0,.0),
    },
    {
        'name': 'Acute pulmonary edema',
        'code':'Acute pulmonary edema',
        'range':(.0,.0),
    },
    {
        'name':'Congestive heart failure',
        'code':'Congestive heart failure',
        'range':(.0,.0),
    },
    {
        'name': 'Shortness of breath',
        'code':'Shortness of breath',
        'range':(.0,.0),
    },
    {
        'name': 'Loss of weight',
        'code':'Loss of weight',
        'range':(.0,.0),
    },
    {
        'name': 'Syncope and collapse',
        'code':'Syncope and collapse',
        'range':(.0,.0),
    },
    {
        'name':'Got ECG',
        'code':'Electrocardiogram, routine ECG with at least 12 leads; with interpretation and report',
        'range':(.0,.0),
    },
    {
        'name':'Got TTE',
        'code':'TTE',
        'range':(.0,.0),
    },
    {
        'name':'Got TEE',
        'code':'TEE',
        'range':(.0,.0),
    },
    {
        'name':'Got pacemaker interrogation',
        'code':'Pacemaker interrogation-Oncall',
        'range':(.0,.0),
    },

]

for q in queries: 
    pyd_config = PytorchDatasetConfig(
    save_dir=dataset_dir,
    max_seq_len=256,
    train_subset_size=0.001,
    train_subset_seed=79163,
    do_include_start_time_min=True,
    static_query_mode=True,
    static_query_name=q['code'],
    static_query_range=q['range'],
)
    code = pyd_config.sample_code()
    print(f" {q['name']} {code['obs_freq']:.0e}")
