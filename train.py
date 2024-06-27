import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/user/denoising-diffusion-pytorch')
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
  dim = 16,
  dim_mults = (1, 2, 4, 8),
  flash_attn = True,
)

diffusion = GaussianDiffusion(
  model,
  image_size = 32,
  timesteps = 1000,   # number of steps
  sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    '/home/user/cifar-10-batches-py/data_batch_1',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                      # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()

torch.save(diffusion.state_dict(), 'my_model_weights.pt')
