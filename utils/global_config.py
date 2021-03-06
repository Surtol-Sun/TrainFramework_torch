import os
import time

use_cuda = False
time_now = time.strftime('%Y%m%d-%H%M%S')
log_file_name = f'results/logs/log-{time_now}.log'
checkpoint_path = f'results/checkpoints/checkpoint-{time_now}.pth'

if not os.path.exists(os.path.split(log_file_name)[0]):
    os.makedirs(os.path.split(log_file_name)[0])
if not os.path.exists(os.path.split(checkpoint_path)[0]):
    os.makedirs(os.path.split(checkpoint_path)[0])


def get_use_cuda():
    '''
    For global using of use_cuda
    :return:
    '''
    global use_cuda
    return use_cuda


def set_use_cuda(value):
    '''
    For global using of use_cuda
    :return:
    '''
    global use_cuda
    use_cuda = value
    return use_cuda


def get_log_file_name():
    global log_file_name
    return log_file_name


def get_checkpoint_path():
    global checkpoint_path
    return checkpoint_path

