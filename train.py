import random
from argparse import ArgumentParser

import numpy as np
import torch
from omegaconf import OmegaConf

from trainer import Trainer


def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--calc-stats", action="store_true")
    args = parser.parse_args()

    return args


def main():
    set_rng_seed(42)

    args = parse_args()
    cfg = OmegaConf.load(args.config)
    trainer = Trainer(cfg)

    if args.calc_stats:
        mean, cov = trainer.calc_inceptionv3_stats(trainer.dl_val)
        np.savez(f"{cfg.data.root}/{cfg.data.dataset_cls}/val_stats.npz",
            mean=mean, cov=cov)
    else:
        trainer.load_inceptionv3_stats()
        trainer.train()


if __name__ == "__main__":
    main()
