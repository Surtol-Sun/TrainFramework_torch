#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import argparse

import torch
import torchvision

import torch.nn as nn
import torchvision.transforms as transforms

import models

parser = argparse.ArgumentParser(description='Evaluate a model')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18_quantize',
                    help='model architecture, useless when input model has a structure saved inside')
parser.add_argument('--model_path',
                    help='path to checkpoint or model with a structure (.pt .pth, .pth.tar accepted)')
parser.add_argument('--print_freq', '-p', default=200, type=int, help='print frequency')
parser.add_argument('-b', '--batch_size', default=16, type=int, help='mini-batch size')
parser.add_argument('-j', '--workers', default=12, type=int, help='number of data loading workers')
parser.add_argument('--remove_prefix', action='store_true', default=False, help='remove prefix in checkpoint')
parser.add_argument('--no_cuda', action='store_true', default=False, help='whether using CUDA in evaluation')

args = parser.parse_args()


def main():
    print('Evaluating {:s}\n=>Model path: {:s}\n=>Dataset path: {:s}\nCUDA: {:}'.format(
        args.arch, args.model_path, args.data, not args.no_cuda
    ))

    if args.model_path.split('.')[-1] == 'pth':
        model = load_model(args.model_path)
    elif args.model_path.split('.')[-1] == 'tar':
        model = load_model_tar(args.model_path)
    elif args.model_path.split('.')[-1] == 'pt':
        model = torch.jit.load(args.model_path)

    if not args.no_cuda:
        model = model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        model = model.to('cpu')
        criterion = nn.CrossEntropyLoss()

    data_loader_val = prepare_val_data_loaders(args.data)

    print('Evaluating...')
    top1, top5, batch_forward_time = evaluate(model, criterion, data_loader_val)
    print('Top-1 acc {:3.2f}%, Top-5 acc {:3.2f}%, Avg forward time {:2.3f}s'.format(top1.avg, top5.avg, batch_forward_time.avg))

    # Size of parameters
    print_size_of_parameter(model)
    
    print('END', '-' * 50, args.model_path)  # end


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    losses = AverageMeter('losses', ':3.4f')
    batch_forward_time = AverageMeter('Batch forward time', ':2.4f')
    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            start_time = time.time()

            output = model(image)

            end_time = time.time()

            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            losses.update(loss, image.size(0))
            batch_forward_time.update(end_time - start_time)

            if i % args.print_freq == 0:
                print('Evaluate: [{0}/{1}]\t'
                      'Batch forward {batch_forward_time.val:.4f}({batch_forward_time.avg:.4f})s\t'
                      'Loss {loss.val:.4f}({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f}({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f}({top5.avg:.3f})'.format(
                    i, len(data_loader), batch_forward_time=batch_forward_time,
                    loss=losses, top1=top1, top5=top5
            ))

    return top1, top5, batch_forward_time


def load_model(checkpoint_file_path):
    model = models.__dict__[args.arch](pretrained=False)
    if not args.no_cuda:
        checkpoint = torch.load(checkpoint_file_path)
    else:
        checkpoint = torch.load(checkpoint_file_path, map_location=torch.device('cpu'))

    # Some checkpoints have inconsistent names with an additional prefix, just remove it
    if args.remove_prefix:
        new_state_dict = {}
        for k, v in checkpoint.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        checkpoint = new_state_dict

    model.load_state_dict(checkpoint)
    return model


def load_model_tar(checkpoint_file_path):
    model = models.__dict__[args.arch](pretrained=False)
    if not args.no_cuda:
        checkpoint = torch.load(checkpoint_file_path)
    else:
        checkpoint = torch.load(checkpoint_file_path, map_location=torch.device('cpu'))

    # Some checkpoints have inconsistent names with an additional prefix, just remove it
    if args.remove_prefix:
        new_state_dict = {}
        for k, v in checkpoint.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        checkpoint = new_state_dict

    model.load_state_dict(checkpoint['state_dict'])
    return model


def print_size_of_parameter(model):
    # torch.save(model.state_dict(), "temp.p")
    # print('Size of parameters:', os.path.getsize("temp.p") / 1e6, '(MB)')
    # os.remove('temp.p')

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    state_dict = {}
    for name, param in model.named_parameters():
        state_dict[name] = param
    torch.save(state_dict, "temp.p")
    param_size = os.path.getsize("temp.p")
    os.remove('temp.p')

    print('Size of parameters: {:.3f}MB, total param num: {:.3f}M, trainable param num: {:.3f}M'.format(
        param_size/2**20, total_num/1e6, trainable_num/1e6
    ))


def prepare_val_data_loaders(data_path):
    valdir = os.path.join(data_path, 'validation')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler)

    return data_loader_test


if __name__ == '__main__':
    main()
