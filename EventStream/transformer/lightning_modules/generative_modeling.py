import dataclasses
import json
import os
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
from ...data.eval_queries import EVAL_TIME, EVAL_CODES
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

class ESTForGenerativeSequenceModelingLM(L.LightningModule):
    """A PyTorch Lightning Module for a `ESTForGenerativeSequenceModeling`."""

    TRAIN_SKIP_METRICS = ("AUROC", "AUPRC", "per_class")
    CLASSIFICATION = {
        DataModality.SINGLE_LABEL_CLASSIFICATION,
        DataModality.MULTI_LABEL_CLASSIFICATION,
    }

    def __init__(
        self,
        config: StructuredTransformerConfig | dict[str, Any],
        optimization_config: OptimizationConfig | dict[str, Any],
        metrics_config: MetricsConfig | dict[str, Any],
        pretrained_weights_fp: Path | None = None,
    ):
        """Initializes the Lightning Module.

        Args:
            config (`Union[StructuredEventstreamTransformerConfig, Dict[str, Any]]`):
                The configuration for the underlying
                `ESTForGenerativeSequenceModeling` model. Should be
                in the dedicated `StructuredTransformerConfig` class or be a dictionary
                parseable as such.
            optimization_config (`Union[OptimizationConfig, Dict[str, Any]]`):
                The configuration for the optimization process handled by the Lightning module. Should
                be in the dedicated `OptimizationConfig` class or be a dictionary parseable
                as such.
        """
        super().__init__()

        # If the configurations are dictionaries, convert them to class objects. They may be passed as
        # dictionaries when the lightning module is loaded from a checkpoint, so we need to support
        # this functionality.
        if type(config) is dict:
            config = StructuredTransformerConfig(**config)
        if type(optimization_config) is dict:
            optimization_config = OptimizationConfig(**optimization_config)
        if type(metrics_config) is dict:
            metrics_config = MetricsConfig(**metrics_config)

        self.config = config
        self.optimization_config = optimization_config
        self.metrics_config = metrics_config
        self.static_query_prefix = ""

        self.save_hyperparameters(
            {
                "config": config.to_dict(),
                "optimization_config": dataclasses.asdict(optimization_config),
            }
        )
        self.build_metrics()

        model_cls = CIPPTForGenerativeSequenceModeling
        if pretrained_weights_fp is None:
            self.model = model_cls(config)
        else:
            self.model = model_cls.from_pretrained(pretrained_weights_fp, config=config)

    def save_pretrained(self, model_dir: Path):
        fp = model_dir / "pretrained_weights"
        self.model.save_pretrained(fp)

    def build_metrics(self):
        """Build the various torchmetrics we'll use to track performance."""

        self.rate_regression_metrics = torch.nn.ModuleDict(
            {
                "r2score": torchmetrics.R2Score(),
                "mse": torchmetrics.MeanSquaredError(),
            }
        )

        self.tte_metrics = torch.nn.ModuleDict(
            {
                "MSE": torchmetrics.MeanSquaredError(),
                "MSLE": torchmetrics.MeanSquaredLogError(),
                "explained_variance": torchmetrics.ExplainedVariance(),
            }
        )

        self.metrics = torch.nn.ModuleDict()
        for task_type, measurements in self.config.measurements_per_generative_mode.items():
            for measurement in measurements:
                vocab_size = self.config.vocab_sizes_by_measurement[measurement]

                if measurement not in self.metrics:
                    self.metrics[measurement] = torch.nn.ModuleDict()
                if task_type not in self.metrics[measurement]:
                    self.metrics[measurement][task_type] = torch.nn.ModuleDict()

                match task_type:
                    case DataModality.SINGLE_LABEL_CLASSIFICATION:
                        cat = MetricCategories.CLASSIFICATION
                        metrics = {
                            Metrics.ACCURACY: (
                                MulticlassAccuracy,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                            Metrics.AUROC: (
                                MulticlassAUROC,
                                [Averaging.MACRO, Averaging.WEIGHTED],
                            ),
                            Metrics.AUPRC: (
                                MulticlassAveragePrecision,
                                [Averaging.MACRO, Averaging.WEIGHTED],
                            ),
                        }
                        metric_kwargs = {
                            "num_classes": vocab_size,
                            "ignore_index": 0,
                            "validate_args": self.metrics_config.do_validate_args,
                        }
                    case DataModality.MULTI_LABEL_CLASSIFICATION:
                        cat = MetricCategories.CLASSIFICATION
                        metrics = {
                            Metrics.ACCURACY: (
                                MultilabelAccuracy,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                            Metrics.AUROC: (
                                MultilabelAUROC,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                            Metrics.AUPRC: (
                                MultilabelAveragePrecision,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                        }
                        metric_kwargs = {
                            "num_labels": vocab_size,
                            "validate_args": self.metrics_config.do_validate_args,
                        }
                    case DataModality.UNIVARIATE_REGRESSION:
                        cat = MetricCategories.REGRESSION
                        metrics = {
                            Metrics.MSE: (torchmetrics.MeanSquaredError, [None]),
                            Metrics.EXPLAINED_VARIANCE: (torchmetrics.ExplainedVariance, [None]),
                        }
                        metric_kwargs = {}
                    case DataModality.MULTIVARIATE_REGRESSION:
                        cat = MetricCategories.REGRESSION
                        metrics = {
                            Metrics.MSE: (torchmetrics.MeanSquaredError, [None]),
                            Metrics.EXPLAINED_VARIANCE: (
                                torchmetrics.ExplainedVariance,
                                [Averaging.MACRO, Averaging.WEIGHTED],
                            ),
                        }
                        metric_kwargs = {}
                    case _:
                        raise ValueError(f"Unrecognized modality {task_type}!")

                auc_kwargs = {
                    **metric_kwargs,
                    "thresholds": self.metrics_config.n_auc_thresholds,
                    "compute_on_cpu": True,
                }
                for metric, (metric_cls, averagings) in metrics.items():
                    if metric in (Metrics.AUROC, Metrics.AUPRC):
                        metric_cls_kwargs = {**auc_kwargs}
                    else:
                        metric_cls_kwargs = {**metric_kwargs}

                    for averaging in averagings:
                        if averaging is None:
                            metric_name = str(metric)
                            averaging_kwargs = {}
                        else:
                            metric_name = f"{averaging}_{metric}"
                            if metric == Metrics.EXPLAINED_VARIANCE:
                                if averaging == Averaging.MACRO:
                                    avg_str = "uniform_average"
                                elif averaging == Averaging.WEIGHTED:
                                    avg_str = "variance_weighted"
                                else:
                                    raise ValueError(f"{averaging} not supported for explained variance.")

                                averaging_kwargs = {"multioutput": avg_str}
                            else:
                                averaging_kwargs = {"average": averaging}

                        if self.metrics_config.do_log_any(cat, metric_name):
                            self.metrics[measurement][task_type][metric_name] = metric_cls(
                                **metric_cls_kwargs, **averaging_kwargs
                            )

    def log_metrics(self, results: dict, split: Split):
        """Logs metric results for a given output result.

        Args:
            `results` (`transformerForGenerativeSequenceModelOutput`):
                The results to assess across the suite of metrics.
            `split` (`str`): The split that should be used when logging metric results.
        """

        log_kwargs = {"batch_size": self.optimization_config.batch_size, "sync_dist": True}

        for metric_name, metric in self.rate_regression_metrics.items():
            try: 
                val = metric(results["rate"], results["answer"].float())
                if self.static_query_prefix: 
                    self.log(f"{split}/{self.static_query_prefix} {metric_name}", val, **log_kwargs)
                else: 
                    self.log(f"{split}/{metric_name}", val, **log_kwargs)
            except: 
                print(f"failed to compute {metric_name} from {results['rate']} and {results['answer'].float()}")
            

        if self.static_query_prefix: return 

        for k in results.keys(): 
            if k.endswith('loss'): 
                self.log(f"{split}/{k}", results[k], **log_kwargs)

        if split != 'train': 
            self.logger.experiment.log({
                f"{split}/rate": wandb.Histogram(np.array(results["rate"].tolist())),
            })
       
        return 

    def training_step(self, batch: PytorchBatch, batch_idx: int) -> torch.Tensor:
        """Training step.

        Skips logging all AUROC, AUPRC, and per_class metric to save compute.
        """
        out = self.model(batch)
        self.log_metrics(out, split=Split.TRAIN)

        return out["loss"]

    def validation_step(self, batch: PytorchBatch, batch_idx: int):
        """Validation step.

        Differs from training only in that it does not skip metrics.
        """
        out = self.model(batch)
        self.log_metrics(out, split=Split.TUNING)

    def test_step(self, batch: PytorchBatch, batch_idx: int):
        """Validation step.

        Differs from training only in that it does not skip metrics.
        """
        out = self.model(batch)
        self.log_metrics(out, split=Split.HELD_OUT)

    def configure_optimizers(self):
        """Configures optimizer and learning rate scheduler.

        Currently this module uses the AdamW optimizer, with configurable weight_decay, with a learning rate
        warming up from 0 on a per-step manner to the configurable `self.optimization_config.init_lr`, then
        undergoes polynomial decay as specified via `self.optimization_config`.
        """
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimization_config.init_lr,
            weight_decay=self.optimization_config.weight_decay,
        )
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=opt,
            num_warmup_steps=self.optimization_config.lr_num_warmup_steps,
            num_training_steps=self.optimization_config.max_training_steps,
            power=self.optimization_config.lr_decay_power,
            lr_end=self.optimization_config.end_lr,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


SKIP_CFG_PARAMS = {"seq_attention_layers", "dep_graph_attention_layers", "hidden_size"}

@hydra_dataclass
class PretrainConfig:
    do_overwrite: bool = False
    seed: int = 1

    config: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "_target_": "EventStream.transformer.config.StructuredTransformerConfig",
            **{
                k: v
                for k, v in StructuredTransformerConfig(measurements_per_dep_graph_level=[]).to_dict().items()
                if k not in SKIP_CFG_PARAMS
            },
        }
    )
    optimization_config: OptimizationConfig = dataclasses.field(default_factory=lambda: OptimizationConfig())
    data_config: PytorchDatasetConfig = dataclasses.field(default_factory=lambda: PytorchDatasetConfig())
    pretraining_metrics_config: MetricsConfig = dataclasses.field(
        default_factory=lambda: MetricsConfig(
            include_metrics={Split.TRAIN: {MetricCategories.LOSS_PARTS: True}},
        )
    )
    final_validation_metrics_config: MetricsConfig = dataclasses.field(
        default_factory=lambda: MetricsConfig(do_skip_all_metrics=False)
    )

    trainer_config: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "accelerator": "auto",
            "devices": "auto",
            "detect_anomaly": False,
            "default_root_dir": "${save_dir}/model_checkpoints",
            "log_every_n_steps": 10,
            "strategy": None,
            "gradient_clip_val": None, 
            "gradient_clip_algorithm": None,
            "fast_dev_run": False,
        }
    )

    experiment_dir: str = omegaconf.MISSING
    save_dir: str = "${experiment_dir}/pretrain/${now:%Y-%m-%d_%H-%M-%S}"

    wandb_logger_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "generative_event_stream_transformer",
            "entity": None,
            "project": None,
            "team": None,
            "log_model": True,
            "do_log_graph": True,
        }
    )

    wandb_experiment_config_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "save_dir": "${save_dir}",
        }
    )

    do_final_validation_on_metrics: bool = True
    do_use_filesystem_sharing: bool = True

    # compile: bool = True

    def __post_init__(self):
        if type(self.save_dir) is str and self.save_dir != omegaconf.MISSING:
            self.save_dir = Path(self.save_dir)
        if "max_epochs" in self.trainer_config:
            raise ValueError("Max epochs is set in the optimization_config, not the trainer config!")
        if "callbacks" in self.trainer_config:
            raise ValueError("Callbacks are built internally, not set via trainer_config!")


@task_wrapper
def train(cfg: PretrainConfig):
    """Runs the end to end training procedure for the pre-training model.

    Args:
        cfg: The pre-training config defining the generative modeling task.
    """

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

    tuning_pyd = PytorchDataset(data_config, split="tuning")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        cfg.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving config files...")
        config_fp = cfg.save_dir / "config.json"
        if config_fp.exists() and not cfg.do_overwrite:
            raise FileExistsError(f"{config_fp} already exists!")
        else:
            logger.info(f"Writing to {config_fp}")
            config.to_json_file(config_fp)

        data_config.to_json_file(cfg.save_dir / "data_config.json", do_overwrite=cfg.do_overwrite)
        optimization_config.to_json_file(
            cfg.save_dir / "optimization_config.json", do_overwrite=cfg.do_overwrite
        )
        cfg.pretraining_metrics_config.to_json_file(
            cfg.save_dir / "pretraining_metrics_config.json", do_overwrite=cfg.do_overwrite
        )
        cfg.final_validation_metrics_config.to_json_file(
            cfg.save_dir / "final_validation_metrics_config.json", do_overwrite=cfg.do_overwrite
        )

    LM = ESTForGenerativeSequenceModelingLM(
        config=config,
        optimization_config=optimization_config,
        metrics_config=cfg.pretraining_metrics_config,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_pyd,
        batch_size=optimization_config.batch_size,
        num_workers=optimization_config.num_dataloader_workers,
        collate_fn=train_pyd.collate,
        shuffle=True, 
        drop_last=True, # TODO REMOVE 
    )
    tuning_dataloader = torch.utils.data.DataLoader(
        tuning_pyd,
        batch_size=optimization_config.validation_batch_size,
        num_workers=optimization_config.num_dataloader_workers,
        collate_fn=tuning_pyd.collate,
        shuffle=False,
    )
    
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        # MonitorInputCallback(),
        AnomalyDetectionCallback(action='zero', print_batch_on_anomaly=True, checkpoint_on_anomaly=False),
    ]
    if optimization_config.patience is not None:
        callbacks.append(
            EarlyStopping(monitor="tuning/loss", mode="min", patience=optimization_config.patience)
        )

    trainer_kwargs = dict(
        **cfg.trainer_config,
        max_epochs=optimization_config.max_epochs,
        callbacks=callbacks,
    )

    if cfg.wandb_logger_kwargs.get("name", None):
        if "do_log_graph" in cfg.wandb_logger_kwargs:
            do_log_graph = cfg.wandb_logger_kwargs.pop("do_log_graph")
        else:
            do_log_graph = False

        wandb_logger = WandbLogger(
            **{k: v for k, v in cfg.wandb_logger_kwargs.items() if v is not None},
            save_dir=cfg.save_dir,
        )

        if os.environ.get("LOCAL_RANK", "0") == "0":
            if do_log_graph:
                # Watching the model naturally tracks parameter values and gradients.
                wandb_logger.watch(LM, log="all", log_graph=True, log_freq=50)

            if cfg.wandb_experiment_config_kwargs:
                wandb_logger.experiment.config.update(cfg.wandb_experiment_config_kwargs)

        trainer_kwargs["logger"] = wandb_logger

    if (optimization_config.gradient_accumulation is not None) and (
        optimization_config.gradient_accumulation > 1
    ):
        trainer_kwargs["accumulate_grad_batches"] = optimization_config.gradient_accumulation

    # Fitting model
    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(model=LM, train_dataloaders=train_dataloader, val_dataloaders=tuning_dataloader)

    LM.save_pretrained(cfg.save_dir)

    if cfg.do_final_validation_on_metrics:
        held_out_pyd = PytorchDataset(data_config, split="held_out")
        held_out_dataloader = torch.utils.data.DataLoader(
            held_out_pyd,
            batch_size=optimization_config.validation_batch_size,
            num_workers=optimization_config.num_dataloader_workers,
            collate_fn=held_out_pyd.collate,
            shuffle=False,
        )

        LM.metrics_config = cfg.final_validation_metrics_config
        LM.build_metrics()

        tuning_metrics = trainer.validate(model=LM, dataloaders=tuning_dataloader)
        held_out_metrics = trainer.test(model=LM, dataloaders=held_out_dataloader)

        if os.environ.get("LOCAL_RANK", "0") == "0":
            logger.info("Saving final metrics...")

            with open(cfg.save_dir / "tuning_metrics.json", mode="w") as f:
                json.dump(tuning_metrics, f)
            with open(cfg.save_dir / "held_out_metrics.json", mode="w") as f:
                json.dump(held_out_metrics, f)

        data_config.fixed_code_mode = True 
        data_config.fixed_time_mode = True 
        data_config.fixed_time = EVAL_TIME
        for c in EVAL_CODES:
            data_config.fixed_code = c
            held_out_pyd = PytorchDataset(data_config, split="held_out")
            held_out_dataloader = torch.utils.data.DataLoader(
                held_out_pyd,
                batch_size=optimization_config.validation_batch_size,
                num_workers=optimization_config.num_dataloader_workers,
                collate_fn=held_out_pyd.collate,
                shuffle=False,
            )
            LM.static_query_prefix = f"{c['name']} ({EVAL_TIME['offset']}–{EVAL_TIME['duration']+EVAL_TIME['offset']})"
            trainer.test(model=LM, dataloaders=held_out_dataloader)

        return tuning_metrics[0]["tuning/loss"], tuning_metrics, held_out_metrics

    return None