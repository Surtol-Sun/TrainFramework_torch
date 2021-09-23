import time

use_cuda = False
time_now = time.strftime('%Y%m%d-%H%M%S')
log_file = open(f'results/logs/log-{time_now}.log', 'w')
checkpoint_path = f'results/checkpoints/checkpoint-{time_now}.pth'


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

