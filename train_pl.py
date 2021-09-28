#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install tqdm
# !pip install torch
# !pip install torchvision torchaudio
# !pip install tensorboardX
# !pip install scikit-learn
# !pip install pytorch-lightning
# !pip install git+https://github.com/ncullen93/torchsample
# !pip install nibabel
# !pip install wget
# !pip install ipywidgets
# !pip install widgetsnbextension
# !pip install tensorflow

# jupyter labextension install @jupyter-widgets/jupyterlab-manager > /dev/null
# jupyter nbextension enable --py widgetsnbextension


# In[2]:


import shutil
import os
import time
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing

import torch
import torch.nn as nn
import torchmetrics
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import model
from dataset import MRDatasetMerged
from torch.utils.data import DataLoader 

import pytorch_lightning as pl
from sklearn import metrics
from ipywidgets import IntProgress


# In[3]:


# !jupyter nbextension enable --py widgetsnbextension
#%load_ext tensorboard
#%tensorboard --logdir lightning_logs/


# In[4]:


class Args:
    def __init__(self):
        self.task = "abnormal" #['abnormal', 'acl', 'meniscus']
        self.plane = "sagittal" #['sagittal', 'coronal', 'axial']
        self.prefix_name = "Test"
        self.augment = 1 #[0, 1]
        self.lr_scheduler = "plateau" #['plateau', 'step']
        self.gamma = 0.5
        self.epochs = 1
        self.lr = 1e-5
        self.flush_history = 0 #[0, 1]
        self.save_model = 1 #[0, 1]
        self.patience = 5
        self.log_every = 100
        
args = Args()


# In[5]:


def to_tensor(x):
    return torch.Tensor(x)

num_workers = multiprocessing.cpu_count() - 1

log_root_folder = "./logs/{0}/{1}/".format(args.task, args.plane)
if args.flush_history == 1:
    objects = os.listdir(log_root_folder)
    for f in objects:
        if os.path.isdir(log_root_folder + f):
            shutil.rmtree(log_root_folder + f)

now = datetime.now()
logdir = log_root_folder + now.strftime("%Y%m%d-%H%M%S") + "/"
os.makedirs(logdir)

writer = SummaryWriter(logdir)

# augmentor = Compose([
#     transforms.Lambda(to_tensor),
#     RandomRotate(25),
#     RandomTranslate([0.11, 0.11]),
#     RandomFlip(),
# #     transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
# ])
data_dir = "/home/jovyan/mrnet_dataset/"

train_dataset = MRDatasetMerged(data_dir, transform=None, train=True)
validation_dataset = MRDatasetMerged(data_dir, train=False)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers, drop_last=False)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False)

mrnet = model.MRNet()


# In[6]:


monitor = "val_f1"

callback = pl.callbacks.ModelCheckpoint(
            monitor=f'{monitor}',
            dirpath=f'/notebooks/checkpoints_{monitor}/',
            filename='checkpoint-{epoch:02d}-{' + f'{monitor}' + ':.2f}',
            save_top_k=3,
            mode='min',
        )


# In[7]:


trainer = pl.Trainer(max_epochs=1, gpus=0, callbacks=[callback]) #1


# In[8]:


trainer.fit(mrnet, train_loader, validation_loader)


# In[ ]:


m = MRNet.load_from_checkpoint(callback.best_model_path)


# In[ ]:


m(validation_dataset[0])


# In[ ]:


#export model
filepath = 'model_v2.onnx'
model = mrnet
input_sample = torch.randn((64, 3, 227, 227))
model.to_onnx(filepath, input_sample, export_params=True)

