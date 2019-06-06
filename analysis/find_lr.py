from __future__ import print_function

import matplotlib
matplotlib.use('agg')

import argparse
import os
import shutil
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import models.wideresnet as models
import dataset.freesound_X as dataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, lwlrap_accumulator, load_checkpoint
from tensorboardX import SummaryWriter
from fastai.basic_data import *
from fastai.basic_train import *
from fastai.train import *
from train import SemiLoss

model =  models.WideResNet(num_classes=80)
train_labeled_set, train_unlabeled_set, val_set, test_set, train_unlabeled_warmstart_set, num_classes, pos_weights = dataset.get_freesound()

labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
val_loader = data.DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)

train_criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

bunch = DataBunch(labeled_trainloader, val_loader, collate_fn=dataset.collate_fn, device=torch.device('cpu'))
learner = Learner(data=bunch, model=model, loss_func=train_criterion)
lr_find(learner)
fig = learner.recorder.plot(return_fig=True, suggestion=True)
fig.save('lr.png')