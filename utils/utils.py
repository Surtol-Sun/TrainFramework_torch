import yaml

from utils.global_config import *

import time
import torch
import numpy as np


def load_config(config_file_path=''):
    # Record configure information into log file
    print_log(f'=========Configuration file: {config_file_path}')
    for line_text in open(config_file_path, 'r', encoding='utf-8'):
        line_text = line_text.strip('\n\r')
        print_log(line_text)
    print_log(f'=========')

    # Fetch configuration
    config = yaml.safe_load(open(config_file_path, 'r', encoding='utf-8'))
    # Get a device to train on
    device_str = config.get('device', None)

    if device_str is not None:
        print_log(f'Device specified in config: {device_str}')
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            print_log('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        print_log(f'Using {device_str} device')

    device = torch.device(device_str)
    config['device'] = device
    return config


def time_string():
    ISOTIMEFORMAT = '%Y%m%d-%H%M%S'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string


def print_log(print_string):
    log_file_name = get_log_file_name()
    with open(log_file_name, 'a+') as log_file:
        print("{:}".format(print_string))
        log_file.write('{:}\n'.format(print_string))
        log_file.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, queue_num=1000):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.store_queue = []

        self.queue_num = queue_num

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.store_queue = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        if self.queue_num is not None:
            for _ in range(n):
                self.count += 1
                self.store_queue.append(val)
                if self.count > self.queue_num:
                    self.count -= 1
                    self.sum -= self.store_queue.pop(0)
        else:
            self.count += n
        self.avg = self.sum / self.count


def _normalization(img_in, dtype=None, max_val=None, min_val=None):
    img_in_type = img_in.dtype if dtype is None else dtype
    img_in = img_in.astype(np.float32)

    max_val = np.max(img_in) if max_val is None else max_val
    min_val = np.min(img_in) if min_val is None else min_val
    img_in = np.clip(img_in, a_min=min_val, a_max=max_val)

    if max_val - min_val > 0:
        img_out = (img_in - min_val) / (max_val - min_val)
    else:
        img_out = img_in - min_val

    # print(f'Normalized image saved as type {img_in_type}, range 0-1')
    if img_in_type == np.float or img_in_type == np.float16 or img_in_type == np.float32 or img_in_type == np.float64:
        return img_out.astype(img_in_type)
    elif img_in_type == np.uint8:
        return (img_out * 255).astype(img_in_type)
    elif img_in_type == np.uint16:
        return (img_out * 65535).astype(img_in_type)
    else:
        return img_out.astype(img_in_type)

