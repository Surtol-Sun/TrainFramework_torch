import os
import re
import torch
import numpy as np


def main():
    deal_file = 'all'  # 'all' or list of file names
    # deal_file = ['best.resnet50.2018-07-16-4310.pth.tar']
    checkpoint_file_dir = './'

    if deal_file == 'all':
        file_list = os.listdir(checkpoint_file_dir)
    else:
        file_list = deal_file

    for file_name in file_list:
        checkpoint_file_path = os.path.join(checkpoint_file_dir, file_name)
        get_pruned_structure(checkpoint_file_path)


def get_pruned_structure(checkpoint_file_path):
    '''
    Print pruned parameters for evaluation_flops.py
    :param checkpoint_file_path: path to checkpoint(.pth and .tar suppported)
    :return:
    '''
    if checkpoint_file_path.split('.')[-1] != 'pth' and checkpoint_file_path.split('.')[-1] != 'tar':
        return -1
    if not os.path.exists(checkpoint_file_path):
        print('File {:s} not exists!'.format(checkpoint_file_path))
        return -2
    print('-' * 25, checkpoint_file_path.split('/')[-1], '-' * 25)

    # Load checkpoint
    remove_prefix = False
    checkpoint = torch.load(checkpoint_file_path, map_location='cpu')
    if checkpoint_file_path.split('.')[-1] == 'tar':
        remove_prefix = True
        checkpoint = checkpoint['state_dict']
    if remove_prefix:
        from _collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        checkpoint = new_state_dict

    # # DEBUG
    # for i, (name, param) in enumerate(checkpoint.items()):
    #     if len(param.shape) == 4:  # conv layer
    #         print('\033[0;42m' + '{:-3d}:{:35s}=>{:s}'.format(i, name, str(param.shape)) + '\033[0m')
    #     # else:
    #     #     print('{:-3d}:{:35s}=>{:s}'.format(i, name, str(param.shape)))
    # # END debug

    # DEBUG
    pruned_index = []
    for i, (name, param) in enumerate(checkpoint.items()):
        if len(param.shape) == 4:  # conv layer
            pruned_index.append(i)
    print(pruned_index)
    # END debug

    # Count zeros in checkpoint
    filter_sumup = layer_if_zero(checkpoint)

    # Sum up pruning
    checkpoint_file_name = checkpoint_file_path.split('/')[-1]
    assign_channel_num_index = []
    if 'vgg16_cifar10' in checkpoint_file_name:
        assign_channel_num_index = [
            [0, 7],
            [14, 21],
            [28, 35, 42],
            [49, 56, 63],
            [70, 77, 84]
        ]
    elif 'vgg16' in checkpoint_file_name:
        assign_channel_num_index = [
            [0, 7],
            [14, 21],
            [28, 35, 42],
            [49, 56, 63],
            [70, 77, 84]
        ]
    elif 'resnet20_cifar10' in checkpoint_file_name:
        assign_channel_num_index = [
            [6, 18, 30],
            [42, 60, 72],
            [84, 102, 114]

        ]
    elif 'resnet32_cifar10' in checkpoint_file_name:
        assign_channel_num_index = [
            [6, 18, 30, 42, 54],
            [66, 84, 96, 108, 120],
            [132, 150, 162, 174, 186, ]
        ]
    elif 'resnet56_cifar10' in checkpoint_file_name:
        assign_channel_num_index = [
            [6, 12, 30, 36, 48, 54, 66, 72, 84, 90, 102, 108, 120, 126, 138, 144, 156, 162],
            [174, 180, 198, 204, 216, 222, 234, 240, 252, 258, 270, 276, 288, 294, 306, 312, 324, 330],
            [342, 348, 366, 372, 384, 390, 402, 408, 420, 426, 438, 444, 456, 462, 474, 480, 492, 498]
        ]
    elif 'resnet110_cifar' in checkpoint_file_name:
        assign_channel_num_index = [
            [6, 12, 30, 36, 48, 54, 66, 72, 84, 90, 102, 108, 120, 126, 138, 144, 156, 162, 174, 180, 192, 198, 210,
             216, 228, 234, 246, 252, 264, 270, 282, 288, 300, 306, 318, 324],
            [336, 342, 360, 366, 378, 384, 396, 402, 414, 420, 432, 438, 450, 456, 468, 474, 486, 492, 504, 510, 522,
             528, 540, 546, 558, 564, 576, 582, 594, 600, 612, 618, 630, 636, 648, 654],
            [666, 672, 690, 696, 708, 714, 726, 732, 744, 750, 762, 768, 780, 786, 798, 804, 816, 822, 834, 840, 852,
             858, 870, 876, 888, 894, 906, 912, 924, 930, 942, 948, 960, 966, 978, 984]
        ]
    elif 'resnet18' in checkpoint_file_name:
        assign_channel_num_index = [
            [6, 18],
            [30, 48],
            [60, 78],
            [90, 108]
        ]
    elif 'resnet34' in checkpoint_file_name:
        assign_channel_num_index = [
        ]
    elif 'resnet50' in checkpoint_file_name:
        assign_channel_num_index = [
            ]
    elif 'mobilenet_v2' in checkpoint_file_name:
        assign_channel_num_index = [
            [6],
            [18, 36],
            [54, 72, 90],
            [108, 126, 144, 162],
            [180, 198, 216],
            [234, 252, 270],
            [288]
        ]

    assign_channel_num = []
    for index_list in assign_channel_num_index:
        channel_num_remained = []
        for index in index_list:
            zero_filter_num, all_filter_num = filter_sumup[index]
            channel_num_remained.append(all_filter_num - zero_filter_num)
        assign_channel_num.append(channel_num_remained)

    print('assign_channel_num =', assign_channel_num)
    return 0


def layer_if_zero(checkpoint, display_zero_type='filter'):
    filter_sumup = {}  # {index: [num zero filter, num all filter]}
    for index, (name, value) in enumerate(checkpoint.items()):
        if len(value.shape) != 4:  # not conv layer
            continue
        if display_zero_type == 'filter':
            zero_filter_num = 0
            for item_i in value:
                a = item_i.data.view(-1)
                b = a.cpu().numpy()
                if np.count_nonzero(b) == 0:
                    zero_filter_num += 1
            filter_sumup[index] = [zero_filter_num, value.shape[0]]
            print("layer: {:<3d}: zero filter={:>4d}/{:<4d}(nonzero={:4d}), prune rate={:>6.2f}%"
                  .format(index, zero_filter_num, value.shape[0], value.shape[0] - zero_filter_num,
                          100 * zero_filter_num / value.shape[0]))

        elif display_zero_type == 'num':
            a = value.data.view(-1)
            b = a.cpu().numpy()
            print("layer: %d, number of nonzero weight is %d, zero is %d" % (
                index, np.count_nonzero(b), len(b) - np.count_nonzero(b)))

    return filter_sumup


if __name__ == '__main__':
    main()
