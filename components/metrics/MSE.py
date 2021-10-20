import torch


def mse(output, target):
    '''
    :param output: Result generated by neural network
    :param target: Ground truth
    :return: Mean squared error between
    '''
    return torch.std(output-target).float()




