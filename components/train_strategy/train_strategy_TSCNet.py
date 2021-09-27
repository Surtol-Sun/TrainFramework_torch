import time
import torch

from utils.utils import print_log, AverageMeter, time_string
from utils.global_config import get_checkpoint_path, get_use_cuda


class TrainTSCNet:
    def __init__(self, model, criterion, optimizer, train_loader, train_config):
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
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.print_freq = train_config['print_freq']
        self.max_epoch = train_config['max_epoch']
        self.learning_rate = train_config['learning_rate']

        from components.metrics.MSE import mse
        self.evaluate_metric_dict = {'MSE': mse}  # A dict that contains concerned metrics, e.g. {IoU: IoUFunc, ...} ToDo !!!!

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
        lambda_gd = 0.2  # Hyper parameter for generator and discriminator loss
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_dis = AverageMeter()
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
        for i, (img1, img2, img3) in enumerate(data_loader):
            # measure data loading time
            time_middle_1 = time.time()

            if self.use_cuda:
                self.model = self.model.cuda()
                img1 = img1.cuda(non_blocking=True)
                img2 = img2.cuda(non_blocking=True)
                img3 = img3.cuda(non_blocking=True)
            with torch.no_grad():
                img1_var = torch.autograd.Variable(img1)
                img2_var = torch.autograd.Variable(img2)
                img3_var = torch.autograd.Variable(img3)

            # compute output
            self.optimizer.zero_grad()
            img1_5 = self.model(torch.cat([img1_var, img2_var], 1))  # Contact 2 images
            img2_5 = self.model(torch.cat([img2_var, img3_var], 1))  # Contact 2 images
            output = self.model(torch.cat([img1_5, img2_5], 1))  # Contact 2 images
            real_img = self.model.discriminator(img2)
            fake_img = self.model.discriminator(output)

            loss_discriminator = 1 - real_img.mean() + fake_img.mean()
            loss = self.criterion(output, img2_var) - lambda_gd * fake_img.mean()

            # compute gradient and do SGD step
            if mode == 'train':
                loss.backward(retain_graph=True)
                loss_discriminator.backward(retain_graph=True)
                self.optimizer.step()

            # Evaluate model
            if mode == 'evaluate':
                for metric_name, metric_func in self.evaluate_metric_dict.items():
                    result = metric_func(output.data, img2)
                    metric_avg_dict[metric_name].update(result)

            # Record results and time
            losses.update(loss.data.item(), output.size(0))
            losses_dis.update(loss_discriminator.data.item(), output.size(0))
            data_time.update(time_middle_1 - start_time)
            batch_time.update(time.time() - start_time)
            start_time = time.time()

            # Print log
            if i % self.print_freq == 0:
                print_str = f'=={mode}== '
                print_str += f'Epoch[{self.epoch}]: [{i}/{len(self.train_loader)}]\t'
                print_str += f'Loss {losses.avg:.4f}\t'
                print_str += f'LossDis {losses_dis.avg:.4f}\t'
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
            if self.epoch > self.max_epoch:
                print_log('Max epoch reached')
                break
            self.adjust_learning_rate(epoch_start=0, reduce_epoch=10)
            print_log(f'Epoch {self.epoch :3d}/{ self.max_epoch :3d} ----- [{time_string():s}]')

            # train for one epoch and evaluate
            train_result_dict = self.run_one_epoch(mode='train')
            print('Evaluating...')
            with torch.no_grad():
                evaluate_result_dict = self.run_one_epoch(mode='evaluate')

            # remember result history and save checkpoint
            crucial_metric = evaluate_result_dict['loss']
            is_best = crucial_metric >= (best_result if best_result is not None else crucial_metric)
            if is_best:
                best_result = crucial_metric
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, self.checkpoint_path)



