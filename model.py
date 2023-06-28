import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


def gather(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t - 1)

    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class SinPosEmbedding(nn.Module):

    def __init__(self, n_dims):
        super().__init__()
        self.n_dims = n_dims
    
    def forward(self, x):
        multiplier = -np.log(10000) / (self.n_dims // 2 - 1)
        embeddings = torch.exp(torch.arange(self.n_dims // 2) * multiplier)
        embeddings = x[:, None] * embeddings[None, :].to(x.device)
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)

        return embeddings


class ConvBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        ]
        super().__init__(*layers)


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, in_channels * 2)
        )

        self.conv_0 = ConvBlock(in_channels, out_channels)
        self.conv_1 = ConvBlock(out_channels, out_channels)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, ts):
        time_scale_shift = self.time_mlp(ts)
        scale, shift = time_scale_shift.chunk(2, dim=1)
        out = x * scale[..., None, None] + shift[..., None, None]

        out = self.conv_0(out)
        out = self.conv_1(out)
        out = self.res_conv(x) + out

        return out


class DDPM(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.n_steps = self.cfg.model.n_steps
        self.register_buffer("ts",
            torch.linspace(1, self.n_steps, self.n_steps, dtype=torch.long))
        self.register_buffer("betas", self.get_betas())
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alphas_cumprod",
            torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", self.alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
            (1 - self.alphas_cumprod).sqrt())
        self.register_buffer("alphas_cumprod_prev",
            F.pad(self.alphas_cumprod[:-1], (1, 0), value=1))
        self.register_buffer("sqrt_recip_alphas", (1.0 / self.alphas).sqrt())
        self.register_buffer("posterior_vars", self.betas\
            * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))

        self.first_conv = nn.Conv2d(
            in_channels=self.cfg.model.in_dims,
            out_channels=self.cfg.model.first_dims,
            kernel_size=self.cfg.model.first_kernel,
            padding=self.cfg.model.first_pad
        )
        self.time_mlp = nn.Sequential(
            SinPosEmbedding(self.cfg.model.time_pos_embed_dims),
            nn.Linear(self.cfg.model.time_pos_embed_dims,
                self.cfg.model.time_mlp_dims),
            nn.GELU(),
            nn.Linear(self.cfg.model.time_mlp_dims,
                self.cfg.model.time_mlp_dims),
        )
        down_channels = self.cfg.model.down_channels
        time_mlp_dims = self.cfg.model.time_mlp_dims
        self.down_blocks = nn.ModuleList([
            ResBlock(in_ch, out_ch, time_mlp_dims) for in_ch, out_ch in
            zip(down_channels[:-1], down_channels[1:])
        ])

        self.mid_block = ConvBlock(down_channels[-1], down_channels[-1])

        up_channels = self.cfg.model.up_channels
        self.up_blocks = nn.ModuleList([
            ResBlock(in_ch * 2, out_ch, time_mlp_dims) for in_ch, out_ch in
            zip(up_channels[:-1], up_channels[1:])
        ])

        self.final_block = ResBlock(up_channels[-1] * 2, up_channels[-1],
            time_mlp_dims)
        self.out_conv = nn.Conv2d(up_channels[-1], self.cfg.model.out_dims, 1)

    def get_betas(self):
        min_beta, max_beta = self.cfg.model.min_beta, self.cfg.model.max_beta
        betas = torch.linspace(min_beta, max_beta, self.cfg.model.n_steps)

        return betas

    def sample_t(self, n):
        rnd_indices = torch.tensor(random.sample(range(self.n_steps), n))
        ts = self.ts[rnd_indices]

        return ts

    def forward_sample(self, x_0, ts):
        noise = torch.randn_like(x_0)
        x_t = (
            gather(self.sqrt_alphas_cumprod, ts, x_0.shape) * x_0 +
            gather(self.sqrt_one_minus_alphas_cumprod, ts, x_0.shape) * noise
        )

        return x_t, noise

    def unnormalize(self, x):
        return np.clip((x + 1) / 2 * 255, 0, 255).astype(np.uint8)

    def backward_step(self, x_t, ts):
        betas_t = gather(self.betas, ts, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = gather(
            self.sqrt_one_minus_alphas_cumprod, ts, x_t.shape)
        sqrt_recip_alphas_t = gather(self.sqrt_recip_alphas, ts, x_t.shape)

        model_mean = sqrt_recip_alphas_t * (x_t\
            - betas_t / sqrt_one_minus_alphas_cumprod_t * self(x_t, ts))

        # no noise at the last step
        if ts[0] == 0:
            output = model_mean
        else:
            posterior_vars_t = gather(self.posterior_vars, ts, x_t.shape)
            noise = torch.randn_like(x_t)
            output = model_mean + noise * posterior_vars_t.sqrt()

        return output

    @torch.no_grad()
    def backward_sample(self, shape):
        self.eval()

        images = []
        device = next(self.parameters()).device
        image = torch.randn(shape, device=device)
        batch_size = shape[0]

        n_steps = self.n_steps
        for i in tqdm(reversed(range(1, n_steps + 1)), total=n_steps):
            ts = torch.full((batch_size, ), i, device=device, dtype=torch.long)
            image = self.backward_step(image, ts)
            raw_image = image.cpu().numpy()
            images.append(raw_image)

        image = self.unnormalize(raw_image)

        return raw_image, image

    def forward(self, x, ts):
        x = self.first_conv(x)
        ts = self.time_mlp(ts)

        residuals = [x.clone()]
        for block in self.down_blocks:
            x = block(x, ts)
            x = F.max_pool2d(x, 2)
            residuals.append(x.clone())

        x = self.mid_block(x)

        for res, block in zip(residuals[::-1], self.up_blocks):
            x = block(torch.cat([x, res], dim=1), ts)
            x = F.interpolate(x, scale_factor=2, mode="bilinear")

        x = self.final_block(torch.cat([x, residuals[0]], dim=1), ts)
        out = self.out_conv(x)

        return out


if __name__ == "__main__":
    from omegaconf import OmegaConf


    cfg = OmegaConf.load("configs/mnist.yaml")
    model = DDPM(cfg)

    samples = model.backward_sample((4, 1, 28, 28))
    print(1)
