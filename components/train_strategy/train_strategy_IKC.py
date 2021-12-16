import time
import torch

from utils.utils import print_log, AverageMeter, time_string
from utils.global_config import get_checkpoint_path, get_use_cuda

from useful_functions.metrics.MSE import mse


class TrainIKC:
    def __init__(self, model, train_loader, val_loader, train_config):
        '''
        :param model:
        :param train_config: Should contain the following keys:
             max_epoch,
        :return:
        '''
        # global use_cuda
        self.epoch = train_config.get('epoch', 0)
        self.use_cuda = get_use_cuda()
        self.checkpoint_path = get_checkpoint_path()

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.print_freq = train_config['print_freq']
        self.max_epoch = train_config['max_epoch']
        self.learning_rate = train_config['learning_rate']


        # A dict that contains concerned metrics, e.g. {IoU: IoUFunc, ...} ToDo !!!!
        self.evaluate_metric_dict = {'MSE': mse}

        # ToDo !!!
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, betas=(0.5, 0.999))
        self.schedule = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    @staticmethod
    def save_checkpoint(state, filename):
        torch.save(state, filename)

    def adjust_learning_rate(self, epoch_start=0, reduce_epoch=30, log=True):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        epoch_converted = max(self.epoch - epoch_start, 0)
        lr = self.learning_rate * (0.1 ** (epoch_converted // reduce_epoch))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        print_str = f"{'-' * 15} learning rate = {lr:.10f} {'-' * 15}"
        if log:
            print_log(print_str)
        else:
            print(print_str)

    def run_one_epoch(self, mode):
        '''
        :param mode: 'train' or 'evaluate'
        ;:param eval_metric: A list for all evaluate metrics
        :return:
        '''
        assert mode in ['train', 'evaluate']
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        metric_avg_dict = {}
        for metric_name in self.evaluate_metric_dict.keys():
            metric_avg_dict[metric_name] = AverageMeter()
        result_dict = {}

        # switch to train mode or evaluate mode
        if mode == 'train':
            self.model.train()
            data_loader = self.train_loader
        elif mode == 'evaluate':
            self.model.eval()
            data_loader = self.train_loader  # ToDo
        else:
            data_loader = None

        start_time = time.time()
        # Run for one epoch
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            time_middle_1 = time.time()

            if self.use_cuda:
                self.model = self.model.cuda()
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

            # compute output
            self.optimizer.zero_grad()
            output = self.model(input_var)
            loss = mse(output, target_var)

            # compute gradient and do SGD step
            if mode == 'train':
                loss.backward()
                self.optimizer.step()

            # Evaluate model
            if mode == 'evaluate':
                for metric_name, metric_func in self.evaluate_metric_dict.items():
                    result = metric_func(output.data, target)
                    metric_avg_dict[metric_name].update(result)

            # Record results and time
            losses.update(loss.data.item(), input.size(0))
            data_time.update(time_middle_1 - start_time)
            batch_time.update(time.time() - start_time)
            start_time = time.time()

            # Print log
            if i % self.print_freq == 0:
                print_str = f'=={mode}== '
                print_str += f'Epoch[{self.epoch}]: [{i}/{len(self.train_loader)}]\t'
                print_str += f'Loss {losses.avg:.4f}\t'
                print_str += f'Time {batch_time.avg:.3f}\t'
                print_str += f'Data {data_time.avg:.3f}\t'
                if mode == 'evaluate':
                    for metric_name, avg_class in metric_avg_dict.items():
                        print_str += f'{metric_name} {avg_class.avg:.3f}\t'
                print_log(print_str)

        # Record result
        result_dict['loss'] = losses.avg
        if mode == 'evaluate':
            for metric_name, avg_class in metric_avg_dict.items():
                result_dict[metric_name] = avg_class.avg
        return result_dict

    def train(self):
        best_result = None
        for epoch in range(self.max_epoch):
            self.epoch += 1
            self.adjust_learning_rate(epoch_start=0, reduce_epoch=30)
            print_log(f'Epoch {self.epoch :3d}/{ self.max_epoch :3d} ----- [{time_string():s}]')

            # train for one epoch and evaluate
            train_result_dict = self.run_one_epoch(mode='train')
            print('Evaluating...')
            evaluate_result_dict = self.run_one_epoch(mode='evaluate')

            # remember result history and save checkpoint
            crucial_metric = evaluate_result_dict['loss']
            is_best = crucial_metric >= (best_result if best_result is not None else crucial_metric)
            if is_best:
                best_result = crucial_metric
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, self.checkpoint_path)


####################
# blur kernel and PCA
####################
import math
import numpy as np
from scipy.ndimage import zoom  # Reference https://blog.csdn.net/u013066730/article/details/101073505
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def cal_sigma(sig_x, sig_y, radians):
    D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    U = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), 1 * np.cos(radians)]])
    sigma = np.dot(U, np.dot(D, U.T))
    return sigma


def anisotropic_gaussian_kernel(l, sigma_matrix, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((l * l, 1)), yy.reshape(l * l, 1))).reshape(l, l, 2)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(xy, inverse_sigma) * xy, 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def isotropic_gaussian_kernel(l, sigma, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def random_anisotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, scaling=3, l=21, tensor=False):
    pi = np.random.random() * math.pi * 2 - math.pi
    x = np.random.random() * (sig_max - sig_min) + sig_min
    y = np.clip(np.random.random() * scaling * x, sig_min, sig_max)
    sig = cal_sigma(x, y, pi)
    k = anisotropic_gaussian_kernel(l, sig, tensor=tensor)
    return k


def random_isotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, l=21, tensor=False):
    x = np.random.random() * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k


def stable_isotropic_gaussian_kernel(sig=2.6, l=21, tensor=False):
    x = sig
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k


def random_gaussian_kernel(l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, tensor=False):
    if np.random.random() < rate_iso:
        return random_isotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, tensor=tensor)
    else:
        return random_anisotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, scaling=scaling, tensor=tensor)


def stable_gaussian_kernel(l=21, sig=2.6, tensor=False):
    return stable_isotropic_gaussian_kernel(sig=sig, l=l, tensor=tensor)


def random_batch_kernel(batch, l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, tensor=True):
    batch_kernel = np.zeros((batch, l, l))
    for i in range(batch):
        batch_kernel[i] = random_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling, tensor=False)
    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel


def stable_batch_kernel(batch, l=21, sig=2.6, tensor=True):
    batch_kernel = np.zeros((batch, l, l))
    for i in range(batch):
        batch_kernel[i] = stable_gaussian_kernel(l=l, sig=sig, tensor=False)
    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel


def downsample_Bicubic(variable, scale):
    tensor = variable.data
    B, C, H, W = tensor.size()
    tensor_v = tensor.view((B*C, 1, H, W))
    new_tensor_list = []
    for i in range(B*C):
        img = tensor_v[i].numpy()
        new_tensor_list.append(zoom(img, 1.0/scale, order=2))  # cube interpolation
    re_tensor_v = torch.from_numpy(np.array(new_tensor_list))
    re_tensor_v = re_tensor_v.view((B, C, H/scale, W/scale))
    return re_tensor_v


def random_batch_noise(batch, high, rate_cln=1.0):
    noise_level = np.random.uniform(size=(batch, 1)) * high
    noise_mask = np.random.uniform(size=(batch, 1))
    noise_mask[noise_mask < rate_cln] = 0
    noise_mask[noise_mask >= rate_cln] = 1
    return noise_level * noise_mask


def b_GaussianNoising(tensor, sigma, mean=0.0, noise_size=None, min=0.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.mul(torch.FloatTensor(np.random.normal(loc=mean, scale=1.0, size=size)), sigma.view(sigma.size() + (1, 1)))
    return torch.clamp(noise + tensor, min=min, max=max)


class BatchSRKernel(object):
    def __init__(self, l=21, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3):
        self.l = l
        self.sig = sig
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.rate = rate_iso
        self.scaling = scaling

    def __call__(self, random, batch, tensor=False):
        if random == True: #random kernel
            return random_batch_kernel(batch, l=self.l, sig_min=self.sig_min, sig_max=self.sig_max, rate_iso=self.rate,
                                       scaling=self.scaling, tensor=tensor)
        else: #stable kernel
            return stable_batch_kernel(batch, l=self.l, sig=self.sig, tensor=tensor)


class PCAEncoder(object):
    def __init__(self, weight):
        self.weight = weight #[l^2, k]
        self.size = self.weight.size()
        self.weight = Variable(self.weight)

    def __call__(self, batch_kernel):
        B, H, W = batch_kernel.size() #[B, l, l]
        return torch.bmm(batch_kernel.view((B, 1, H * W)), self.weight.expand((B, ) + self.size)).view((B, -1))


class BatchBlur(nn.Module):
    def __init__(self, l=15):
        super(BatchBlur, self).__init__()
        self.l = l
        if l % 2 == 1:
            self.pad = nn.ReflectionPad2d(l // 2)
        else:
            self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        # self.pad = nn.ZeroPad2d(l // 2)

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = pad.view((C * B, 1, H_p, W_p))
            kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        else:
            input_CBHW = pad.view((1, C * B, H_p, W_p))
            kernel_var = kernel.contiguous().view((B, 1, self.l, self.l)).repeat(1, C, 1, 1).view((B * C, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, groups=B*C).view((B, C, H, W))


class SRMDPreprocessing(object):
    def __init__(self, scale, pca, random=21, para_input=10, kernel=21, noise=True, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, rate_cln=0.2, noise_high=0.08):
        self.encoder = PCAEncoder(pca)
        self.kernel_gen = BatchSRKernel(l=kernel, sig=sig, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling)
        self.blur = BatchBlur(l=kernel)
        self.para_in = para_input
        self.l = kernel
        self.noise = noise
        self.scale = scale
        self.rate_cln = rate_cln
        self.noise_high = noise_high
        self.random = random

    def __call__(self, hr_tensor, kernel=False):
        ### hr_tensor is tensor, not cuda tensor
        B, C, H, W = hr_tensor.size()
        b_kernels = Variable(self.kernel_gen(self.random, B, tensor=True)).cuda() if self.cuda else Variable(self.kernel_gen(self.random, B, tensor=True))
        # blur
        hr_blured_var = self.blur(Variable(hr_tensor), b_kernels)
        # kernel encode
        kernel_code = self.encoder(b_kernels) # B x self.para_input
        # Down sample
        lr_blured_t = downsample_Bicubic(hr_blured_var, self.scale)

        # Noisy
        if self.noise:
            Noise_level = torch.FloatTensor(random_batch_noise(B, self.noise_high, self.rate_cln))
            lr_noised_t = b_GaussianNoising(lr_blured_t, Noise_level)
        else:
            Noise_level = torch.zeros((B, 1))
            lr_noised_t = lr_blured_t

        Noise_level = Variable(Noise_level)
        re_code = torch.cat([kernel_code, Noise_level * 10], dim=1) if self.noise else kernel_code
        lr_re = Variable(lr_noised_t)
        return (lr_re, re_code, b_kernels) if kernel else (lr_re, re_code)




