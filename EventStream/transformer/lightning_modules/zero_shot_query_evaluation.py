import dataclasses
import json
import os, sys
from pathlib import Path
from typing import Any
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
def train(cfg: PretrainConfig):
    WANDB_RUN_ID = "487l51nc"
    api = wandb.Api()
    run = api.run(f"payal-collabs/EveryQueryGPT/{WANDB_RUN_ID}")
    pretrained_weights_fp = f"{run.config['save_dir']}/pretrained_weights"

    L.seed_everything(cfg.seed)
    if cfg.do_use_filesystem_sharing:
        torch.multiprocessing.set_sharing_strategy("file_system")

    train_pyd = PytorchDataset(cfg.data_config, split="train")

    config = cfg.config
    optimization_config = cfg.optimization_config
    data_config = cfg.data_config

    config.set_to_dataset(train_pyd)
    optimization_config.set_to_dataset(train_pyd)
    data_config.set_to_dataset(train_pyd)

    LM = ESTForGenerativeSequenceModelingLM(
        config=config,
        optimization_config=optimization_config,
        metrics_config=cfg.pretraining_metrics_config,
        pretrained_weights_fp=pretrained_weights_fp,
    )
    
    trainer_kwargs = dict(
        **cfg.trainer_config,
    )
    trainer_kwargs['devices'] = [2]
    trainer_kwargs['num_nodes'] = 1
    trainer_kwargs["logger"] = WandbLogger(entity="payal-collabs", project="EveryQueryGPT",id=WANDB_RUN_ID, resume='must')
    trainer = L.Trainer(**trainer_kwargs)

    held_out_pyd = PytorchDataset(data_config, split="held_out")
    held_out_dataloader = torch.utils.data.DataLoader(
        held_out_pyd,
        batch_size=optimization_config.validation_batch_size,
        num_workers=optimization_config.num_dataloader_workers,
        collate_fn=held_out_pyd.collate,
        shuffle=False,
    )

    LM.metrics_config = cfg.final_validation_metrics_config

    data_config.fixed_code_mode = True 
    data_config.fixed_time_mode = True 
    for t in EVAL_TIMES: 
        data_config.fixed_time = t
        for c in reversed(EVAL_CODES):
            LM.build_metrics()
            query = f"{c['name']}_{t['offset']}_{t['duration']} new"
            # print(c)
            # if sum([query in k for k in run.summary.keys()]): 
            #     print(f"skipping {query}, already computed")
            #     continue
            if query=="Thrombus or embolism_0_365 new": 
                print(query)
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
                metrics = trainer.test(model=LM, dataloaders=held_out_dataloader)
                results = trainer.predict(model=LM, dataloaders=held_out_dataloader)
                results = {key: [d[key] for d in results] for key in results[0]}
                y_prob = torch.cat(results['zero_prob'])
                y_true = torch.cat(results['zero_truth'])
                t_rate = torch.hstack([x for x in results['truncated_rate'] if x.nelement()])
                t_ans = torch.hstack([x for x in results['truncated_answer'] if x.nelement()])
                from torchmetrics.functional.classification import binary_auroc
                from torchmetrics.classification import BinaryAUROC
                from sklearn.metrics import roc_auc_score, r2_score
                # print(roc_auc_score(y_true, y_prob))
                # print(binary_auroc(y_prob, y_true.long()))
                # print(r2_score(t_ans, t_rate))
                ipdb.set_trace()
    return 
