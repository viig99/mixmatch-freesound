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
from pytorch_toolbelt import losses as L
import models.wideresnet as models
from models.adamw import AdamW
import dataset.freesound_X as dataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, lwlrap_accumulator, load_checkpoint
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=1024, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
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
parser.add_argument('--alpha', default=0.6, type=float)
parser.add_argument('--rampup-length', default=0, type=float)
parser.add_argument('--T', default=10.0, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--num_cpu', default=os.cpu_count() - 2, type=int)
parser.add_argument('--lambda_bc', default=40.0, type=float)
parser.add_argument('--lambda_m', default=10.0, type=float)
parser.add_argument('--lambda_n', default=1.0, type=float)


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

def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing freesound')

    train_labeled_set, train_unlabeled_set, val_set, test_set, train_unlabeled_warmstart_set, num_classes, pos_weights = dataset.get_freesound()
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_cpu, drop_last=True, collate_fn=dataset.collate_fn, pin_memory=True)
    noisy_train_loader = data.DataLoader(train_unlabeled_warmstart_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_cpu, drop_last=True, collate_fn=dataset.collate_fn)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_cpu, drop_last=True, collate_fn=dataset.collate_fn_unlabbelled)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_cpu, collate_fn=dataset.collate_fn, pin_memory=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_cpu, collate_fn=dataset.collate_fn, pin_memory=True)

    # Model
    print("==> creating WRN-28-4")

    def create_model(ema=False):
        model = nn.DataParallel(models.WideResNet(num_classes=num_classes))
        if use_cuda:
            model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    bce_loss = L.BinaryFocalLoss()
    train_criterion = SemiLoss(bce_loss)
    noisy_criterion = NoisyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, verbose=True, threshold=1e-3, cooldown=10, min_lr=2e-6)

    # load_checkpoint(model, ema_model, optimizer)

    ema_optimizer= WeightEMA(model, ema_model, num_classes, alpha=args.ema_decay)
    start_epoch = 0

    # Resume
    title = 'noisy-cifar-10'
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
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U',  'Train Loss N', 'Train Acc.', 'Valid Loss', 'Valid Acc.', 'Test Loss', 'Test Acc.'])

    writer = SummaryWriter(args.out)
    step = 0
    test_accs = []
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        train_loss, train_loss_x, train_loss_u, train_loss_n = train(labeled_trainloader, 
            unlabeled_trainloader, noisy_train_loader, model, optimizer, 
            ema_optimizer, train_criterion, noisy_criterion, epoch, use_cuda)
        _, train_acc = validate(labeled_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats')
        val_loss, val_acc = validate(val_loader, ema_model, criterion, epoch, use_cuda, mode='Valid Stats')
        test_loss, test_acc = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats ')

        step = args.batch_size * args.val_iteration * (epoch + 1)

        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/valid_loss', val_loss, step)
        writer.add_scalar('losses/test_loss', test_loss, step)

        writer.add_scalar('accuracy/train_acc', train_acc, step)
        writer.add_scalar('accuracy/val_acc', val_acc, step)
        writer.add_scalar('accuracy/test_acc', test_acc, step)
        
        scheduler.step(test_loss)

        # append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, train_loss_n, train_acc, val_loss, val_acc, test_loss, test_acc])

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
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


def train(labeled_trainloader, unlabeled_trainloader, noisy_train_loader, model, optimizer, ema_optimizer, criterion, noisy_criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    losses_p = AverageMeter()
    losses_n = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    size = args.val_iteration

    bar = Bar('Training', max=size)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    noisy_train_iter = iter(noisy_train_loader)

    
    model.train()
    for batch_idx in range(size):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()

        try:
            inputs_n, targets_n = noisy_train_iter.next()
        except:
            noisy_train_iter = iter(noisy_train_loader)
            inputs_n, targets_n = noisy_train_iter.next()

        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_n, targets_n = inputs_n.cuda(), targets_n.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()


        with torch.no_grad():
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.sigmoid(outputs_u) + torch.sigmoid(outputs_u2)) / 2
            targets_u = torch.sigmoid((p - 0.5) * args.T)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(args.alpha, args.alpha)

        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))
        outputs_n = model(inputs_n)

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/args.val_iteration)
        loss_noisy = noisy_criterion(outputs_n, targets_n)

        loss = args.lambda_bc * Lx + (args.lambda_m * w * Lu) + (args.lambda_n * w * loss_noisy)

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(args.lambda_bc * Lx.item(), inputs_x.size(0))
        losses_u.update(args.lambda_m * w * Lu.item(), inputs_x.size(0))
        losses_n.update(args.lambda_n * w * loss_noisy.item(), inputs_n.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | Loss_n: {loss_n:.4f} | W: {w:.4f}'.format(
                    batch=batch_idx + 1,
                    size=size,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    loss_n=losses_n.avg,
                    w=ws.avg,
                    )
        bar.next()
    bar.finish()

    ema_optimizer.step(bn=True)

    return (losses.avg, losses_x.avg, losses_u.avg, losses_n.avg)

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

def linear_rampup(current, rampup_length=args.rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip((current / rampup_length) - 0.5, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        Lx = self.criterion(outputs_x, targets_x)
        Lu = torch.mean((torch.sigmoid(outputs_u) - targets_u)**2)
        return Lx, Lu, linear_rampup(epoch)

class NoisyLoss(object):
    def __init__(self):
        self.ls = torch.nn.LogSigmoid()

    def __call__(self, outputs_n, targets_n):
        Ln = -torch.mean(torch.sum((targets_n * self.ls(outputs_n)), dim=1, dtype=torch.float32))
        return Ln

class WeightEMA(object):
    def __init__(self, model, ema_model, num_classes, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.tmp_model = models.WideResNet(num_classes=num_classes).cuda()
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.copy_(param.data)

    def step(self, bn=False):
        if bn:
            # copy batchnorm stats to ema model
            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                tmp_param.data.copy_(ema_param.data.detach())

            self.ema_model.load_state_dict(self.model.state_dict())

            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                ema_param.data.copy_(tmp_param.data.detach())
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)
                # customized weight decay
                param.data.mul_(1 - self.wd)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

if __name__ == '__main__':
    main()