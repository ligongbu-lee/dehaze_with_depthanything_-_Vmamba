import time
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader

import torch.backends.cudnn as cudnn
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
#import encoding
from torchvision import transforms
import argparse
import pytorch_ssim
import dataset
import utils
import options as option

from data import create_dataloader, create_dataset


#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
# print(parser.parse_args().opt)
opt = option.parse("ir-sde_train_gan.yml", is_train=False)
print(opt)
print("????????????????????????????????????????????????????????????")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### Create train gan dataset and dataloader
train_gan_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    train_gan_set = create_dataset(dataset_opt)
    train_gan_loader = create_dataloader(train_gan_set, dataset_opt, opt)
    train_gan_loaders.append(train_gan_loader)




train_gan_set = create_dataset(dataset_opt)
train_gan_loader = create_dataloader(train_gan_set, dataset_opt, opt)
for a in enumerate(train_gan_loader):
    # import time
    # time.sleep(1)
    print(a)


"""
python ./train.py \
--opt "ir-sde_train_gan.yml" \
--load_name "" \
--multi_gpu "true"  \
--save_path "./models/models_rain100H" \
--sample_path "./samples" \
--save_mode 'epoch' \
--save_by_epoch 250 \
--save_by_iter 10000 \
--lr_g 0.0002 \
--b1 0.5 \
--b2 0.999 \
--weight_decay 0.0 \
--train_batch_size 16 \
--epochs 5000 \
--lr_decrease_epoch 2000 \
--num_workers 1 \
--crop_size 256 \
--no_gpu "false" \
--rainaug "false" \


"""

