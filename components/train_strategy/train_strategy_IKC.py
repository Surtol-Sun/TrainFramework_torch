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
        self.loss_dict = {'PLoss': mse, 'SFTMDLoss': mse, 'CLoss': mse}
        self.evaluate_metric_dict = {'MSE': mse}

        # Optimizer and schedule here
        self.optimizer_predictor = torch.optim.Adam(self.model.Predictor.parameters(), lr=0.01, betas=(0.5, 0.999))
        self.optimizer_sftmd = torch.optim.Adam(self.model.SFTMD.parameters(), lr=0.01, betas=(0.5, 0.999))
        self.optimizer_corrector = torch.optim.Adam(self.model.Corrector.parameters(), lr=0.01, betas=(0.5, 0.999))
        self.schedule_predictor = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_predictor, gamma=0.9)
        self.schedule_sftmd = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_sftmd, gamma=0.9)
        self.schedule_corrector = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_corrector, gamma=0.9)

    @staticmethod
    def save_checkpoint(state, filename):
        torch.save(state, filename)

    def adjust_learning_rate(self, epoch_start=0, reduce_epoch=30, log=True):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        epoch_converted = max(self.epoch - epoch_start, 0)
        lr = self.learning_rate * (0.1 ** (epoch_converted // reduce_epoch))
        for param_group in self.optimizer_predictor.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_sftmd.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_corrector.param_groups:
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
        losses_avg_dict = {}
        metric_avg_dict = {}
        for metric_name in self.evaluate_metric_dict.keys():
            metric_avg_dict[metric_name] = AverageMeter()
        for loss_name in self.loss_dict.keys():
            losses_avg_dict[loss_name] = AverageMeter()
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
        for i, data_dict in enumerate(data_loader):
            # measure data loading time
            time_middle_1 = time.time()

            input = data_dict['LQ']
            kernel_code = data_dict['Kernel_code']
            target = data_dict['GT']

            if self.use_cuda:
                self.model = self.model.cuda()
                input = input.cuda(non_blocking=True)
                kernel_code = kernel_code.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
                kernel_code_var = torch.autograd.Variable(kernel_code)
                target_var = torch.autograd.Variable(target)

            # Prepare optimizer for training
            self.optimizer_predictor.zero_grad()
            self.optimizer_sftmd.zero_grad()
            self.optimizer_corrector.zero_grad()

            # Train Predictor and SFTMD
            kernel_code_predict = self.model.Predictor(input_var)
            output = self.model.SFTMD(input_var, kernel_code_var)
            # print(f'input size = {input.shape}')
            # print(f'kernel_code size = {kernel_code.shape}')
            # print(f'target size = {target.shape}')
            # print(f'kernel_code_predict size = {kernel_code_predict.shape}')
            # print(f'output size = {output.shape}')
            loss_predictor = self.loss_dict['PLoss'](kernel_code_predict, kernel_code)
            loss_sftmd = self.loss_dict['SFTMDLoss'](target_var, output)

            if mode == 'train':
                self.optimizer_predictor.step()
                self.optimizer_sftmd.step()

            # Train Corrector
            # corrected_kernel_code = self.model.Corrector(input_var, kernel_code_var)

            # Evaluate model
            if mode == 'evaluate':
                for metric_name, metric_func in self.evaluate_metric_dict.items():
                    result = metric_func(output.data, target)
                    metric_avg_dict[metric_name].update(result)

            # Record results and time
            losses_avg_dict['PLoss'].update(loss_predictor.data.item(), input.size(0))
            losses_avg_dict['SFTMDLoss'].update(loss_sftmd.data.item(), input.size(0))
            losses_avg_dict['CLoss'].update(0, input.size(0))
            data_time.update(time_middle_1 - start_time)
            batch_time.update(time.time() - start_time)
            start_time = time.time()

            # Print log
            if i % self.print_freq == 0:
                print_str = f'=={mode}== '
                print_str += f'Epoch[{self.epoch}]: [{i}/{len(self.train_loader)}]\t'
                for loss_name in losses_avg_dict.keys():
                    print_str += f'{loss_name} {losses_avg_dict[loss_name].avg:.4f}\t'
                print_str += f'Time {batch_time.avg:.3f}\t'
                print_str += f'Data {data_time.avg:.3f}\t'
                if mode == 'evaluate':
                    for metric_name, avg_class in metric_avg_dict.items():
                        print_str += f'{metric_name} {avg_class.avg:.3f}\t'
                print_log(print_str)

        # Record result
        result_dict['loss'] = losses_avg_dict['SFTMDLoss'].avg
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
                    # 'optimizer': self.optimizer.state_dict(),
                }, self.checkpoint_path)


