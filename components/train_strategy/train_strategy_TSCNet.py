import time
import torch

from utils.utils import print_log, AverageMeter, time_string
from utils.global_config import get_checkpoint_path, get_use_cuda
from utils.supported_items import supported_loss_dict


class TrainTSCNet:
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

        from components.metrics.MSE import mse
        self.evaluate_metric_dict = {'MSE': mse}  # A dict that contains concerned metrics, e.g. {IoU: IoUFunc, ...} ToDo !!!!

        # ToDo !!!
        self.optimizer = torch.optim.Adam(self.model.generator.parameters(), lr=0.01, betas=(0.5, 0.999))
        self.schedule = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.optimizer_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.schedule_d = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    @staticmethod
    def save_checkpoint(state, filename):
        torch.save(state, filename)

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
        losses_d = AverageMeter()
        metric_avg_dict = {}
        for metric_name in self.evaluate_metric_dict.keys():
            metric_avg_dict[metric_name] = AverageMeter()
        result_dict = {}

        # switch to train mode or evaluate mode
        self.model.zero_grad()
        if mode == 'train':
            self.model.train()
            data_loader = 'self.train_loader'
        elif mode == 'evaluate':
            # self.model.eval()  # ToDo !!!
            self.model.train()
            data_loader = 'self.val_loader'
        else:
            data_loader = None

        start_time = time.time()
        # Run for one epoch
        # for i, (img1, img2, img3) in enumerate(data_loader):
        # for i, (img1, img2, img3) in enumerate(eval(data_loader)): ToDo!!!
        for i, (img1, img2, img3) in enumerate(self.train_loader):
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
            out_discriminator = self.model.discriminator(output)
            out_discriminator_img2 = self.model.discriminator(img2_var)
            output = torch.autograd.Variable(output)

            if i == 0:
                from skimage.io import imsave
                # imsave(f'results/images/{self.epoch}-1.tif', img1.detach().to('cpu').numpy())
                # imsave(f'results/images/{self.epoch}-2.tif', img2.detach().to('cpu').numpy())
                # imsave(f'results/images/{self.epoch}-3.tif', img3.detach().to('cpu').numpy())
                # imsave(f'results/images/{self.epoch}-1-5.tif', img1_5.detach().to('cpu').numpy())
                # imsave(f'results/images/{self.epoch}-2-5.tif', img2_5.detach().to('cpu').numpy())
                imsave(f'results/images/{self.epoch}-2out.tif', output.detach().to('cpu').numpy())

            # Train Generator
            loss_mse = supported_loss_dict['MSE'](output, img2_var)
            loss_cyc = torch.pow(output - img2_var, 2).mean(dim=[0, 1, 2, 3]) - torch.log(out_discriminator_img2.mean())
            loss = loss_mse + loss_cyc
            if mode == 'train' and losses_d.avg < 0.5:
                loss.backward()
                self.optimizer.step()
                self.schedule.step()

            # Train Discriminator
            self.optimizer_d.zero_grad()
            real_img = self.model.discriminator(img2_var)
            fake_img = self.model.discriminator(output)
            loss_discriminator = -torch.log(real_img.mean()) - torch.log(1.0 - fake_img.mean())
            if mode == 'train' and losses_d.avg > 0.2:
                loss_discriminator.backward()
                self.optimizer_d.step()
                self.schedule_d.step()

            # Evaluate model
            if mode == 'evaluate':
                for metric_name, metric_func in self.evaluate_metric_dict.items():
                    result = metric_func(output.data, img2)
                    metric_avg_dict[metric_name].update(result)

            # Record results and time
            losses.update(loss.data.item(), output.size(0))
            losses_d.update(loss_discriminator.data.item(), output.size(0))
            data_time.update(time_middle_1 - start_time)
            batch_time.update(time.time() - start_time)
            start_time = time.time()

            # Print log
            if i % self.print_freq == 0:
                print_str = f'=={mode}== '
                print_str += f'Epoch[{self.epoch}]: [{i}/{len(self.train_loader)}]\t'
                print_str += f'Loss {losses.avg:.4f}\t'
                print_str += f'LossD {losses_d.avg:.4f}\t'
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



