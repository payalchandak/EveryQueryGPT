import dataclasses
import json, pandas as pd
import os, sys
from pathlib import Path
from typing import Any
import pickle
import lightning as L
import omegaconf
import torch
import torch.multiprocessing
import torchmetrics
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelAveragePrecision,
)
from transformers import get_polynomial_decay_schedule_with_warmup
from ...data.config import PytorchDatasetConfig
from ...data.pytorch_dataset import PytorchDataset
from ...data.types import DataModality, PytorchBatch
from ...data.eval_queries import EVAL_TIMES, EVAL_CODES
from ...utils import hydra_dataclass, task_wrapper
from ..conditionally_independent_model import CIPPTForGenerativeSequenceModeling
from ..config import (
    Averaging,
    MetricCategories,
    Metrics,
    MetricsConfig,
    OptimizationConfig,
    Split,
    StructuredEventProcessingMode,
    StructuredTransformerConfig,
)
from ..model_output import GenerativeSequenceModelOutput
from ..utils import expand_indexed_regression, str_summary
from .custom_callbacks import MonitorInputCallback, AnomalyDetectionCallback
import wandb, numpy as np 
from EventStream.transformer.lightning_modules.generative_modeling import (
    PretrainConfig,
    ESTForGenerativeSequenceModelingLM,
)
import ipdb

@task_wrapper
def dump_preditions(cfg: PretrainConfig, device:int=0):
    pretrained_weights_fp = f"{cfg.save_dir}/pretrained_weights"
    results_dir = cfg.save_dir+'/specific_query_predictions/'
    os.makedirs(results_dir, exist_ok=True)

    L.seed_everything(cfg.seed)
    if cfg.do_use_filesystem_sharing:
        torch.multiprocessing.set_sharing_strategy("file_system")

    train_pyd = PytorchDataset(cfg.data_config, split="train")
    config = cfg.config
    optimization_config = cfg.optimization_config
    data_config = cfg.data_config
    config.set_to_dataset(train_pyd)
    optimization_config.set_to_dataset(train_pyd)
    # data_config.set_to_dataset(train_pyd)

    LM = ESTForGenerativeSequenceModelingLM(
        config=config,
        optimization_config=optimization_config,
        metrics_config=cfg.pretraining_metrics_config,
        pretrained_weights_fp=pretrained_weights_fp,
    )
    LM.metrics_config = cfg.final_validation_metrics_config
    
    trainer_kwargs = dict(
        **cfg.trainer_config,
    )
    trainer_kwargs['devices'] = [device]
    trainer_kwargs['num_nodes'] = 1
    trainer = L.Trainer(**trainer_kwargs)

    data_config.fixed_code_mode = True 
    data_config.fixed_time_mode = True 
    for t in EVAL_TIMES: 
        data_config.fixed_time = t
        for c in EVAL_CODES:
            LM.build_metrics()
            query = f"{c['name']}_{t['offset']}_{t['duration']}"
            filename = f"{results_dir}/{query}.pkl"
            if os.path.exists(filename): 
                print(f"skipping {query}, already computed")
                continue
            with open(filename, "wb") as f: 
                pickle.dump([], f)
                print(f"Wrote empty {query}")
            try:
                LM.static_query_prefix = query
                data_config.fixed_code = c
                held_out_pyd = PytorchDataset(data_config, split="held_out")
                held_out_dataloader = torch.utils.data.DataLoader(
                    held_out_pyd,
                    batch_size=optimization_config.validation_batch_size,
                    num_workers=optimization_config.num_dataloader_workers,
                    collate_fn=held_out_pyd.collate,
                    shuffle=False,
                )
                results = trainer.predict(model=LM, dataloaders=held_out_dataloader)
                results = {key: [d[key] for d in results] for key in results[0]}
                with open(filename, "wb") as f: 
                    pickle.dump(results, f)
                    print(f"Wrote {query}")
            except:
                if os.path.exists(filename): 
                    os.remove(filename)
                    print(f"Failed - deleted {query}")
                sys.exit(0)
    
    with open(cfg.save_dir+'experiment.json', 'r') as file:
        exp = json.load(file)
    runs = pd.read_csv(f'{exp['dir']}runs.csv')
    runs.loc[exp['run'],'zeroshot_finished'] = True
    runs.to_csv(f'{cfg.experiment["dir"]}runs.csv')

    return 