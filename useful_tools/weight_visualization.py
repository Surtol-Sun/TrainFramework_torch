import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

checkpoint_prefix = '/home/edai/work_files/PretrainedModels/PyTorchModels'
checkpoint_path = os.path.join(checkpoint_prefix, 'resnet18-5c106cde.pth')


def main():
    if checkpoint_path.split('.')[-1] == 'pth':
        state_dict = load_statedict(checkpoint_path)
    elif checkpoint_path.split('.')[-1] == 'tar':
        state_dict = load_statedict_tar(checkpoint_path)
    else:
        print('Unsupported model or checkpoint')
        exit(0)
    print('=> Checkpoint loaded checkpoint from {}'.format(checkpoint_path))

    for key_i, value_i in state_dict.items():
        weights_i = value_i.detach().numpy()  # shape: (out_channel, in_channel, kernel_1, kernel_2)
        for kernel_per_channel in weights_i:
            heatmap_c = int(np.sqrt(kernel_per_channel.shape[0])) + 1
            for out_kernel_index, out_per_kernal in enumerate(kernel_per_channel):
                plt.subplot(heatmap_c, heatmap_c, out_kernel_index + 1)
                sns.heatmap(out_per_kernal, annot=True, cmap=plt.cm.Blues)

            plt.get_current_fig_manager().full_screen_toggle()
            plt.show()


def load_statedict(checkpoint_file_path, remove_prefix=False):
    checkpoint = torch.load(checkpoint_file_path, map_location=torch.device('cpu'))

    # Some checkpoints have inconsistent names with an additional prefix, just remove it
    if remove_prefix:
        new_state_dict = {}
        for k, v in checkpoint.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        checkpoint = new_state_dict

    return checkpoint


def load_statedict_tar(checkpoint_file_path, remove_prefix=False):
    checkpoint = torch.load(checkpoint_file_path, map_location=torch.device('cpu'))
    checkpoint = checkpoint['state_dict']

    # Some checkpoints have inconsistent names with an additional prefix, just remove it
    if remove_prefix:
        new_state_dict = {}
        for k, v in checkpoint.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        checkpoint = new_state_dict

    return checkpoint


if __name__ == '__main__':
    main()
