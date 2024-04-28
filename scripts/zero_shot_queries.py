#!/usr/bin/env python
"""Pre-trains a model from scartch."""

try:
    # This color-codes and prettifies error messages if the script fails.
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import copy
import os
import hydra
import torch
from omegaconf import OmegaConf
import wandb
from EventStream.logger import hydra_loguru_init
from EventStream.transformer.lightning_modules.generative_modeling import (
    PretrainConfig,
)
from EventStream.transformer.lightning_modules.zero_shot_query_evaluation import (
    dump_preditions, test,
)

torch.set_float32_matmul_precision("high")

@hydra.main(version_base=None, config_name="pretrain_config")
def main(cfg: PretrainConfig):
    WANDB_RUN_ID="vhjwx9p8"
    hydra_loguru_init()
    if type(cfg) is not PretrainConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")
    return dump_preditions(cfg, WANDB_RUN_ID=WANDB_RUN_ID, device=2)

if __name__ == "__main__":
    main()
