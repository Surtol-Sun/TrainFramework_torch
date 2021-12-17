import os
import json
import torch
import torch.utils.data as data
import numpy as np
from skimage import io
from utils.utils import _normalization


def dataset_mydatabase_IKC(dataloader_config):
    # Ref -- https://blog.csdn.net/Teeyohuang/article/details/79587125

    dataset_path = dataloader_config['dataset_path']
    train_batch_size = dataloader_config['train_batch_size']
    val_batch_size = dataloader_config['val_batch_size']
    num_workers = os.cpu_count()
    # num_workers = 1

    train_data = _DatasetLD(data_path=dataset_path)
    test_data = _DatasetLD(data_path=dataset_path)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=val_batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True)

    # return train_data, test_data
    return train_loader, val_loader


class _DatasetLD(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.dataset_path = data_path
        self.img_info_list = []  # [{'path':'path to image file',  'index': index of this image in stack}, ...]
        self.img_index_list = []  # [[index1, index2, index3], ...], indexes of images in self.image_files
        self.image_file_list = []  # This loader load all images in !! RAM !!, which requires large RAM

        # Generate required data
        stack_img_num = 3
        for img_path, json_dict in self.read_tif_file(self.dataset_path):
            img_file = io.imread(img_path)
            for i in range(img_file.shape[0]):
                if not (2 < json_dict['ZStep'][i] < 40):
                    continue
                img_f = _normalization(img_file[i], dtype=np.float32)
                self.image_file_list.append(np.ascontiguousarray(img_f))
                if i >= stack_img_num - 1:
                    index_base = len(self.image_file_list)
                    self.img_index_list.append(list(range(index_base - stack_img_num, index_base)))
                    self.img_info_list.append({'path': img_path, 'index': i + 1})

    @staticmethod
    def read_tif_file(img_root):
        tif_image_list = []  # [image_path, json_dict]
        for root, dirs, files in os.walk(img_root, topdown=True):
            if not len(files):
                continue
            print('-' * 20)

            img_path, json_dict = '', {}
            # Read TIF file path
            if any('.tif' in x for x in files):
                img_name = [x for x in files if '.tif' in x]
                if len(img_name) > 1:
                    print(f'Folder {root} has more than one image: {img_name}')
                    continue
                img_name = img_name[0]
                img_path = os.path.join(root, img_name)

            # Read JSON file content
            if any('.json' in x for x in files):
                json_name = [x for x in files if '.json' in x]
                if len(json_name) > 1:
                    print(f'Folder {root} has more than one json file: {json_name}')
                    continue
                json_name = json_name[0]
                json_path = os.path.join(root, json_name)

                with open(json_path, 'r', encoding='utf-8') as f:
                    json_dict = json.load(f)

            # Write result to list
            if img_path and json_dict:
                tif_image_list.append([img_path, json_dict])
                print(f'Folder {root} already read')
            else:
                print(f'!! One or more content is empty')
                print(f'Folder {root}: {files}')
                print(f'Image path = {img_path}')
                print(f'Json content = {json_dict}')

        return tif_image_list

    @staticmethod
    def _inner_rand_cut(img_in, cut_shape):
        shape_len = len(img_in.shape)
        for i, axis_len in enumerate(img_in.shape):
            loc = locals()
            if axis_len - cut_shape[i] <= 0:
                continue
            cut_start = np.random.randint(0, axis_len - cut_shape[i])

            # img_in = img_in[cut_start:cut_start+cut_len]
            exe_script = f'img_in=img_in[{":," * i}cut_start:cut_start+cut_len,{":," * (shape_len - i - 1)}]'
            exec(exe_script, {'img_in': img_in, 'cut_start': cut_start, 'cut_len': cut_shape[i]}, loc)
            img_in = loc['img_in']
        return img_in

    @staticmethod
    def _sort_num(name_string):
        '''
            Separate numbers in a name, in order to sort.
            Extract the first number in string
        '''
        import re
        num = re.findall('\d+\.?\d*', name_string)
        try:
            num = float(num[0])
        except:
            num = -1.0
        return num

    def __getitem__(self, index):  # Read data once
        img_list = []
        image_index_list = self.img_index_list[index]
        for image_index in image_index_list:
            img = self.image_file_list[image_index]  # [H, W]
            img_list.append(img)

        img_array = np.stack(img_list, axis=0)  # [Stack, H, W]
        img_array = self._inner_rand_cut(img_array, (3, 128, 128))

        img_GT = torch.from_numpy(np.ascontiguousarray(img_array)).float()

        return {'LQ': 'LQ with no data',
                'GT': img_GT,
                'LQ_path': 'LQ_path with no data',
                'GT_path': self.img_info_list[index]['path'],
                }

    def __len__(self):
        return len(self.img_index_list)


####################
# blur kernel and PCA
####################
import cv2
import math
import numpy as np
from scipy.ndimage import zoom  # Reference https://blog.csdn.net/u013066730/article/details/101073505
import torch.nn as nn
from torch.autograd import Variable


def cal_sigma(sig_x, sig_y, radians):
    D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    U = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), 1 * np.cos(radians)]])
    sigma = np.dot(U, np.dot(D, U.T))
    return sigma


def anisotropic_gaussian_kernel(l, sigma_matrix, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((l * l, 1)), yy.reshape(l * l, 1))).reshape(l, l, 2)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(xy, inverse_sigma) * xy, 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def isotropic_gaussian_kernel(l, sigma, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def random_anisotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, scaling=3, l=21, tensor=False):
    pi = np.random.random() * math.pi * 2 - math.pi
    x = np.random.random() * (sig_max - sig_min) + sig_min
    y = np.clip(np.random.random() * scaling * x, sig_min, sig_max)
    sig = cal_sigma(x, y, pi)
    k = anisotropic_gaussian_kernel(l, sig, tensor=tensor)
    return k


def random_isotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, l=21, tensor=False):
    x = np.random.random() * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k


def stable_isotropic_gaussian_kernel(sig=2.6, l=21, tensor=False):
    x = sig
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k


def random_gaussian_kernel(l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, tensor=False):
    if np.random.random() < rate_iso:
        return random_isotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, tensor=tensor)
    else:
        return random_anisotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, scaling=scaling, tensor=tensor)


def stable_gaussian_kernel(l=21, sig=2.6, tensor=False):
    return stable_isotropic_gaussian_kernel(sig=sig, l=l, tensor=tensor)


def random_noise(high, rate_cln=1.0):
    noise_level = np.random.random() * high
    noise_mask = 0 if np.random.random() < rate_cln else 1
    return noise_level * noise_mask


def GaussianNoising(img_in, sigma, mean=0.0, noise_size=None, min=0.0, max=1.0):
    if noise_size is None:
        size = img_in.shape
    else:
        size = noise_size
    noise = np.random.normal(loc=mean, scale=1.0, size=size) * sigma
    return np.clip(noise + img_in, min=min, max=max)


class SRKernel(object):
    def __init__(self, l=21, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3):
        self.l = l
        self.sig = sig
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.rate = rate_iso
        self.scaling = scaling

    def __call__(self, random):
        if random:  # random kernel
            return random_gaussian_kernel(l=self.l, sig_min=self.sig_min, sig_max=self.sig_max, rate_iso=self.rate,
                                          scaling=self.scaling, tensor=False)
        else:  # stable kernel
            return stable_gaussian_kernel(l=self.l, sig=self.sig, tensor=False)


class PCAEncoder(object):
    def __init__(self, weight):
        self.weight = weight  # [l^2, k]
        self.size = self.weight.shape

    def __call__(self, kernel):
        H, W = kernel.size()  # [l, l]
        return torch.bmm(kernel.view((B, 1, H * W)), self.weight.expand((B,) + self.size)).view((B, -1))


class Blur(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, img_in, kernel):
        return cv2.filter2D(img_in, -1, kernel=kernel)


class SRMDPreprocessing(object):
    def __init__(self, scale, pca, random=21, kernel=21, noise=True, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0,
                 scaling=3, rate_cln=0.2, noise_high=0.08):
        self.encoder = PCAEncoder(pca)
        # pca_matrix = torch.load('./pca_matrix.pth',map_location=lambda storage, loc: storage)
        #     print('PCA matrix shape: {}'.format(pca_matrix.shape))
        self.kernel_gen = SRKernel(l=kernel, sig=sig, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling)
        self.blur = Blur()
        self.l = kernel
        self.noise = noise
        self.scale = scale
        self.rate_cln = rate_cln
        self.noise_high = noise_high
        self.random = random

    def __call__(self, img_in, kernel=False):
        # Generate kernel
        kernel = self.kernel_gen(self.random)
        # blur
        hr_blured = self.blur(img_in, kernel)
        # kernel encode
        kernel_code = self.encoder(kernel)
        # Down sample
        lr_blured_t = zoom(hr_blured, 1.0 / self.scale, order=2)  # cube interpolation

        # Noisy
        if self.noise:
            Noise_level = random_noise(self.noise_high, self.rate_cln)
            lr_noised_t = GaussianNoising(lr_blured_t, Noise_level)
        else:
            Noise_level = 0
            lr_noised_t = lr_blured_t

        re_code = torch.cat([kernel_code, Noise_level * 10], dim=1) if self.noise else kernel_code
        lr_re = Variable(lr_noised_t)
        return (lr_re, re_code, kernel) if kernel else (lr_re, re_code)


if __name__ == '__main__':
    opt = {'datasets': {
        'train': {
            'name': 'TPM_Skin', 'mode': 'TPM_MongoDB', 'dataroot_GT': '/home/yxsun/win_data/20211209Skin',
            'dataroot_LQ': None, 'use_shuffle': True, 'n_workers': 8, 'batch_size': 32, 'GT_size': 256, 'LR_size': 64,
            'use_flip': True, 'use_rot': True, 'color': 'RGB', 'phase': 'train', 'scale': 4, 'data_type': 'img'},
        'val': {
            'name': 'TPM_Skin', 'mode': 'TPM_MongoDB', 'dataroot_GT': '/home/yxsun/win_data/20211209Skin',
            'dataroot_LQ': None, 'phase': 'val', 'scale': 4, 'data_type': 'img'}},
        'network_G': {'which_model_G': 'SFTMD', 'in_nc': 3, 'out_nc': 3, 'nf': 64, 'nb': 16, 'upscale': 4,
                      'code_length': 10, 'scale': 4},
    }
    dataset = _DatasetLD(opt['datasets']['train'])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True,
                                              num_workers=os.cpu_count(), drop_last=True,
                                              pin_memory=False)
    for data in data_loader:
        pass
