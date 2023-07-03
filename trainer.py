import os
from datetime import datetime

import cv2
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import inception_v3
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from dataset import get_eval_dataloader
from model import DDPM
from utils import calc_fid_score


class Trainer:

    def __init__(self, cfg):
        self.cfg = cfg

        transforms = Compose([
            ToTensor(),
            Resize(cfg.data.target_size),
            Normalize(cfg.data.mean, cfg.data.std, inplace=True)
        ])
        dataset_cls = eval(f"datasets.{self.cfg.data.dataset_cls}")
        ds_train = dataset_cls(cfg.data.root, True, transforms, download=True)
        ds_val = dataset_cls(cfg.data.root, False, transforms, download=True)
        self.dl_train = DataLoader(ds_train, cfg.training.batch_size, True,
            num_workers=cfg.training.n_workers)
        self.dl_val = DataLoader(ds_val, cfg.training.batch_size, False,
            num_workers=cfg.training.n_workers)

        self.device = cfg.device
        self.model = DDPM(cfg).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), cfg.training.lr)

        self._init_inceptionv3()

        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = f"logs/{cur_time}"
        os.makedirs(self.log_dir, exist_ok=True)

    def _init_inceptionv3(self):
        inceptionv3 = inception_v3(weights="IMAGENET1K_V1")
        self.inceptionv3 = create_feature_extractor(inceptionv3,
            {"avgpool": "feats"}).to(self.device)
        self.inceptionv3.eval()

    @torch.no_grad()
    def calc_inceptionv3_stats(self, dataloader):
        feats = []
        for i, sample in enumerate(tqdm(dataloader)):
            images = sample[0].expand(-1, 3, -1, -1).to(self.device)
            feats.append(self.inceptionv3(images)["feats"].cpu().numpy())

        feats = np.concatenate(feats).squeeze()
        mean = feats.mean(axis=0)
        cov = np.cov(feats.T)

        return mean, cov

    def load_inceptionv3_stats(self):
        data_dir = f"{self.cfg.data.root}/{self.cfg.data.dataset_cls}"
        stats_path = f"{data_dir}/val_stats.npz"
        stats = np.load(stats_path)
        self.mean_gt, self.cov_gt = stats["mean"], stats["cov"]

    def _train_epoch(self):
        self.model.train()
        avg_loss = 0

        for i, sample in enumerate(tqdm(self.dl_train)):
            self.optimizer.zero_grad()

            images = sample[0].to(self.device)
            ts = self.model.sample_t(images.size(0))
            noisy_images, noise_targets = self.model.forward_sample(images, ts)
            noise_outputs = self.model(noisy_images, ts)
            loss = self.criterion(noise_outputs, noise_targets)

            loss.backward()
            self.optimizer.step()

            avg_loss += loss.detach().cpu().item()

        avg_loss /= len(self.dl_train)

        return avg_loss

    @torch.no_grad()
    def _val_epoch(self):
        self.model.eval()
        avg_loss = 0

        for i, sample in enumerate(tqdm(self.dl_val)):
            images = sample[0].to(self.device)
            ts = self.model.sample_t(images.size(0))
            noisy_images, noise_targets = self.model.forward_sample(images, ts)
            noise_outputs = self.model(noisy_images, ts)
            loss = self.criterion(noise_outputs, noise_targets)

            avg_loss += loss.cpu().item()

        avg_loss /= len(self.dl_val)

        return avg_loss

    @torch.no_grad()
    def _eval_on_samples(self):
        batch_size = self.cfg.eval.batch_size
        height, width = self.cfg.data.target_size
        shape = (batch_size, 1, height, width)
        raw_images_all = []
        epoch_dir = os.path.join(self.log_dir, f"epoch_{self.epoch_idx}")
        os.makedirs(epoch_dir)

        for i in range(self.cfg.eval.n_batches):
            raw_images, images = self.model.backward_sample(shape)
            raw_images_all.append(raw_images)

            images = np.repeat(images.transpose((0, 2, 3, 1)), [3], axis=-1)
            for idx in range(images.shape[0]):
                cv2.imwrite(os.path.join(epoch_dir, f"sample_{i}_{idx}.png"),
                    images[idx])

        raw_images_all = np.concatenate(raw_images_all)
        eval_dl = get_eval_dataloader(raw_images_all, self.cfg.eval.batch_size,
            self.cfg.training.n_workers)
        mean_pred, cov_pred = self.calc_inceptionv3_stats(eval_dl)
        fid_score = calc_fid_score(self.mean_gt, self.cov_gt, mean_pred,
            cov_pred)

        return fid_score

    def train(self):
        self.start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        wandb_cfg = OmegaConf.load("wandb_cfg.yaml")
        wandb.login(key=wandb_cfg.key)
        wandb.init(
            project="my-diffusion",
            name=self.start_time,
            config=self.cfg
        )

        for i in range(self.cfg.training.n_epochs):
            self.epoch_idx = i
            loss_train = self._train_epoch()

            loss_val = self._val_epoch()

            fid_score = self._eval_on_samples()

            wandb.log({
                "loss/train": loss_train,
                "loss/val": loss_val,
                "FID": fid_score,
                "epoch": i
            })

        wandb.finish()


if __name__ == "__main__":
    # from omegaconf import OmegaConf


    # cfg = OmegaConf.load("configs/mnist.yaml")
    # trainer = Trainer(cfg)
    # print(1)
    feats = np.random.normal(size=(4, 3))
    print(np.cov(feats.T))
