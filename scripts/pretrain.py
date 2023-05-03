import sys
sys.path.append('..')

import hydra

from EventStream.transformer.generative_sequence_modelling_lightning import PretrainConfig, train

@hydra.main(version_base=None, config_name="pretrain_config")
def main(cfg: PretrainConfig):
    if type(cfg) is not PretrainConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_='object')
    # TODO(mmd): This isn't the right return value for hyperparameter sweeps.
    return train(cfg)

if __name__ == "__main__":
    main()