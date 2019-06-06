'''
rm -rf result/* && mkdir -p result/checkpoints && ./sync.sh && python3.6 train.py
'''

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

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=1024, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0,1,2', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--n-labeled', type=int, default=4467,
                        help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=1024,
                        help='Number of labeled data')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=20, type=float)
parser.add_argument('--rampup-length', default=50, type=float)
parser.add_argument('--T', default=10.0, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0  # best test accuracy
train_labeled_set, train_unlabeled_set, val_set, test_set, train_unlabeled_warmstart_set, num_classes, pos_weights = dataset.get_freesound(only_valid=True)
if use_cuda:
    pos_weights = pos_weights.cuda()
bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing freesound')

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count() - 2, collate_fn=dataset.collate_fn)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    # Model
    print("==> creating WRN-28-2")

    def create_model():
        model = nn.DataParallel(models.WideResNet(num_classes=num_classes))
        if use_cuda:
            model.cuda()
        return model

    model = create_model()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=10, verbose=True, threshold=1e-3, cooldown=10, min_lr=2e-6)
    start_epoch = 0
    # Resume
    title = 'simple_resnet'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join('/tts_data/kaggle/mixmatch/MixMatch-pytorch/result', 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Valid Loss', 'Valid Acc.', 'Test Loss', 'Test Acc.'])

    writer = SummaryWriter(args.out)
    step = 0
    test_accs = []
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        train_loader = labeled_trainloader
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        train_loss = train(train_loader, model, optimizer, bce_loss, epoch, use_cuda)
        _, train_acc = validate(labeled_trainloader, model, criterion, epoch, use_cuda, mode='Train Stats')
        val_loss, val_acc = validate(val_loader, model, criterion, epoch, use_cuda, mode='Valid Stats')
        test_loss, test_acc = validate(test_loader, model, criterion, epoch, use_cuda, mode='Test Stats ')
        step = args.batch_size * args.val_iteration * (epoch + 1)

        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/valid_loss', val_loss, step)
        writer.add_scalar('losses/test_loss', test_loss, step)

        writer.add_scalar('accuracy/train_acc', train_acc, step)
        writer.add_scalar('accuracy/val_acc', val_acc, step)
        writer.add_scalar('accuracy/test_acc', test_acc, step)
        
        scheduler.step(val_acc)

        # append logger file
        logger.append([train_loss, val_loss, val_acc, test_loss, test_acc])

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, val_acc)
        test_accs.append(test_acc)
    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)

    print('Mean acc:')
    print(np.mean(test_accs[-20:]))


def train(labeled_trainloader, model, optimizer, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    size = args.val_iteration
    bar = Bar('Training', max=size)
    labeled_train_iter = iter(labeled_trainloader)
    model.train()

    for batch_idx in range(size):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)

        logits_x = model(inputs_x)
        loss = criterion(logits_x, targets_x)
        losses.update(loss.item(), inputs_x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
                    batch=batch_idx + 1,
                    size=size,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg
                    )
        bar.next()
    bar.finish()
    return losses.avg

def validate(valloader, model, criterion, epoch, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lrap = AverageMeter()
    lwlrap_acc = lwlrap_accumulator()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            lwlrap_acc.accumulate_samples(targets, outputs)
            losses.update(loss.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | LRAP: {lrap: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        lrap=lwlrap_acc.overall_lwlrap()
                        )
            bar.next()
        bar.finish()
    return (losses.avg, lwlrap_acc.overall_lwlrap())

def save_checkpoint(state, is_best, val_acc, checkpoint=args.out, filename='checkpoint'):
    filepath = os.path.join(checkpoint, f"checkpoints/{filename}_{state['epoch']}.pth.tar")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
        print(f"Best accuracy model saved to {os.path.abspath('result/model_best.pth.tar')} with accuracy {val_acc:.4f}")

if __name__ == '__main__':
    main()