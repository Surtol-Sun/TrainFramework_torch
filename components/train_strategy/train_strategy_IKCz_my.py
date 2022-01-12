import time
import torch

from utils.utils import print_log, AverageMeter, time_string
from utils.global_config import get_checkpoint_path, get_use_cuda

from useful_functions.losses.mse import MSE_Loss


class TrainIKCz_my:
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
        from torch.nn import CrossEntropyLoss
        from useful_functions.losses.xz_3d_my import XZ_3D_MY_Loss
        self.loss_dict = {'GLoss': XZ_3D_MY_Loss(), 'DLoss': CrossEntropyLoss(), 'GDLoss': CrossEntropyLoss()}
        self.evaluate_metric_dict = {'MSE': MSE_Loss()}

        # Optimizer and schedule here
        self.optimizer_generator = torch.optim.Adam(self.model.generator.parameters(), lr=0.01, betas=(0.5, 0.999))
        self.optimizer_discriminator = torch.optim.Adam(self.model.discriminator.parameters(), lr=0.01, betas=(0.5, 0.999))
        self.schedule_generator = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_generator, gamma=0.9)
        self.schedule_discriminator = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_discriminator, gamma=0.9)

    @staticmethod
    def save_checkpoint(state, filename):
        torch.save(state, filename)

    def adjust_learning_rate(self, epoch_start=0, reduce_epoch=30, log=True):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        epoch_converted = max(self.epoch - epoch_start, 0)
        lr = self.learning_rate * (0.1 ** (epoch_converted // reduce_epoch))
        for param_group in self.optimizer_generator.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_discriminator.param_groups:
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
            data_loader = self.val_loader
        else:
            data_loader = None

        start_time = time.time()
        # Run for one epoch
        for i, data_dict in enumerate(data_loader):
            # measure data loading time
            time_middle_1 = time.time()

            input = data_dict['LQ']
            target = data_dict['GT']

            if self.use_cuda:
                self.model = self.model.cuda()
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

            # Prepare optimizer for training
            self.optimizer_generator.zero_grad()
            self.optimizer_discriminator.zero_grad()

            # Train Generator
            output = self.model.generator(input_var)
            loss_g = self.loss_dict['GLoss'](output, target_var)

            # Train Discriminator
            import numpy as np
            axis_len, cut_len = input_var.size(2), input_var.size(1)
            cut_start = np.random.randint(0, axis_len-cut_len)

            input_d_true = input_var[:, :, cut_start:cut_start+cut_len, :]  # [B, C, cut to C, W]
            # input_d_true = torch.transpose(input_d_true, 1, 2)  # [B, C, cut to C, W] => [B, cut to C, C, W]

            input_d_false = output[:, :, cut_start:cut_start+cut_len, :]  # [B, C, cut to C, W]
            input_d_false = torch.transpose(input_d_false, 1, 2)  # [B, C, cut to C, W] => [B, cut to C, C, W]

            output_d_true = self.model.discriminator(input_d_true)
            output_d_false = self.model.discriminator(input_d_false)
            if self.use_cuda:
                loss_d_true = self.loss_dict['DLoss'](output_d_true, torch.tensor([1] * input_var.size(0)).cuda())
                loss_d_false = self.loss_dict['DLoss'](output_d_false, torch.tensor([0] * input_var.size(0)).cuda())
            else:
                loss_d_true = self.loss_dict['DLoss'](output_d_true, torch.tensor([1] * input_var.size(0)))
                loss_d_false = self.loss_dict['DLoss'](output_d_false, torch.tensor([0] * input_var.size(0)))

            # Train Generator and Discriminator
            output_g_d = self.model(input_var)
            if self.use_cuda:
                loss_g_d = self.loss_dict['GDLoss'](output_g_d, torch.tensor([1] * input_var.size(0)).cuda())
            else:
                loss_g_d = self.loss_dict['GDLoss'](output_g_d, torch.tensor([1] * input_var.size(0)))

            if mode == 'train':
                loss_g.backward()

                # if losses_avg_dict['DLoss'].avg < 0.8:
                #     loss_g_d.backward(retain_graph=True)
                # else:
                #     ((loss_d_true + loss_d_false) / 2).backward(retain_graph=True)
                #     self.optimizer_discriminator.step()
                # self.optimizer_generator.step()
                # loss_g_d.backward()  # Clear Graph

                # # View change of parameters
                # for name, param in self.model.named_parameters():
                #     # print(name, param[0])
                #     if name == 'generator.inc.double_conv.0.weight':
                #         print(f'Name={name}, Require Grad={param.requires_grad}, Grad={param.grad}')
                #     if name == 'generator.outc.weight':
                #         print(f'Name={name}, Require Grad={param.requires_grad}, Grad={param.grad}')
                #     if name == 'discriminator.net.0.weight':
                #         print(f'Name={name}, Require Grad={param.requires_grad}, Grad={param.grad}')
                # # print(f'Name=output, Require Grad={output.requires_grad}, Grad={output.grad}')
                # print(f'Name=output_d_false, Require Grad={output_d_false.requires_grad}, Grad={output_d_false.grad}')
                # print(f'Name=input_d_false, Require Grad={input_d_false.requires_grad}, Grad={input_d_false.grad}')

                self.optimizer_generator.step()
                self.optimizer_discriminator.step()

            # Evaluate model
            if mode == 'evaluate':
                for metric_name, metric_func in self.evaluate_metric_dict.items():
                    result = metric_func(output.data, target)
                    metric_avg_dict[metric_name].update(result)

            # Record results and time
            losses_avg_dict['GLoss'].update(loss_g.data.item(), input.size(0))
            losses_avg_dict['DLoss'].update(loss_d_true.data.item() + loss_d_false.data.item(), input.size(0))
            losses_avg_dict['GDLoss'].update(loss_g_d.data.item(), input.size(0))
            data_time.update(time_middle_1 - start_time)
            batch_time.update(time.time() - start_time)
            start_time = time.time()

            # Print log
            if i % self.print_freq == 0:
                if mode == 'train':
                    from skimage import io
                    io.imsave(f'results/{self.epoch}-input.tif', input.cpu().detach().numpy()[0])
                    io.imsave(f'results/{self.epoch}-output.tif', output.cpu().detach().numpy()[0])

                print_str = f'=={mode}== '
                print_str += f'Epoch[{self.epoch}]: [{i}/{len(data_loader)}]\t'
                for loss_name in losses_avg_dict.keys():
                    print_str += f'{loss_name} {losses_avg_dict[loss_name].avg:.4f}\t'
                print_str += f'Time {batch_time.avg:.3f}\t'
                print_str += f'Data {data_time.avg:.3f}\t'
                if mode == 'evaluate':
                    for metric_name, avg_class in metric_avg_dict.items():
                        print_str += f'{metric_name} {avg_class.avg:.3f}\t'
                print_log(print_str)

        # Record result
        result_dict['loss'] = losses_avg_dict['GLoss'].avg
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
            # crucial_metric = evaluate_result_dict['loss']
            crucial_metric = self.epoch
            is_best = crucial_metric >= (best_result if best_result is not None else crucial_metric)
            if is_best:
                best_result = crucial_metric
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    # 'optimizer': self.optimizer.state_dict(),
                }, self.checkpoint_path)


