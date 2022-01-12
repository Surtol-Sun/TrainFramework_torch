import os
import json
import torch
import torch.utils.data as data
import numpy as np
from skimage import io
import microscPSF.microscPSF as msPSF
from scipy.signal import convolve
from utils.utils import _normalization

import cv2


def dataset_mydatabase_IKCz(dataloader_config):
    # Ref -- https://blog.csdn.net/Teeyohuang/article/details/79587125

    dataset_path = dataloader_config['dataset_path']
    train_batch_size = dataloader_config['train_batch_size']
    val_batch_size = dataloader_config['val_batch_size']
    data_out_shape = dataloader_config['data_out_shape']
    num_workers = os.cpu_count()
    # num_workers = 1

    train_data = _DatasetLD(data_path=dataset_path, data_out_shape=data_out_shape, mode='train')
    test_data = _DatasetLD(data_path=dataset_path, data_out_shape=data_out_shape, mode='test')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=val_batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True)

    # return train_data, test_data
    return train_loader, val_loader


class _DatasetLD(torch.utils.data.Dataset):
    def __init__(self, data_path, data_out_shape, mode='train'):
        super().__init__()
        self.mode = mode  # 'train' or 'test'
        self.dataset_path = data_path
        self.img_info_list = []  # [{'path':'path to image file',  'index': index of this image in stack}, ...]
        self.img_index_list = []  # [[index1, index2, index3], ...], indexes of images in self.image_files
        self.image_file_list = []  # This loader load all images in !! RAM !!, which requires large RAM
        self.image_lowre_file_list = []  # Low resolution files

        self.LowResolutionPreprocess = BlurPreprocessing()

        self.data_out_shape = data_out_shape

        # Generate required data
        stack_img_num = self.data_out_shape[0]
        for img_path, json_dict in self.read_tif_file(self.dataset_path):
            img_file = io.imread(img_path)
            img_file_lowre = self.LowResolutionPreprocess(img_file, psf_dim='zxy', return_kernel=False)
            # img_file_lowre = img_file
            for i in range(img_file.shape[0]):
                img_f = img_file[i]
                img_f = self.LowResolutionPreprocess.convolve(img_f, self.LowResolutionPreprocess.GaussianKernel2D(l=11, sigma=2))
                img_f = _normalization(img_f, dtype=np.float32)
                self.image_file_list.append(np.ascontiguousarray(img_f))

                img_f_lowre = img_file_lowre[i]
                img_f_lowre = self.LowResolutionPreprocess.convolve(img_f_lowre, self.LowResolutionPreprocess.GaussianKernel2D(l=11, sigma=2))
                img_f_lowre = _normalization(img_f_lowre, dtype=np.float32)
                self.image_lowre_file_list.append(np.ascontiguousarray(img_f_lowre))
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
        data_out_z, data_out_x, data_out_y = self.data_out_shape

        index = index % len(self.img_index_list)
        img_list = []
        img_lowre_list = []
        image_index_list = self.img_index_list[index]
        for image_index in image_index_list:
            img = self.image_file_list[image_index]  # [H, W]
            img_list.append(img)

            img_lowre = self.image_lowre_file_list[image_index]  # [H, W]
            img_lowre_list.append(img_lowre)

        img_array = np.stack(img_list+img_lowre_list, axis=0)  # [Stack, H, W]
        img_array = self._inner_rand_cut(img_array, (data_out_z*2, data_out_x, data_out_y))

        img_GT = np.array(img_array[:data_out_z], dtype=np.float32)
        img_LQ = np.array(img_array[data_out_z:], dtype=np.float32)

        # print(f'img_GT size = {img_GT.shape}')
        # print(f'img_LQ size = {img_LQ.shape}')
        # print(f'kernel_code size = {kernel_code.shape}')
        # print(f'img_kernel size = {img_kernel.shape}')
        # from skimage import io
        # for i, img in enumerate(img_GT):
        #     io.imsave(f'GT{i}.tif', img)
        # for i, img in enumerate(img_LQ):
        #     io.imsave(f'LQ{i}.tif', img)
        # io.imsave(f'K.tif', img_kernel)


        return {'LQ': img_LQ,
                'Kernel_code': np.array([0.5] * 10, dtype=np.float32),
                'GT': img_GT,
                'GT_path': self.img_info_list[index]['path'],
                'LQ_path': 'LQ_path with no data',
                }

    def __len__(self):
        dataset_length = len(self.img_index_list) * 2 if self.mode == 'train' else len(self.img_index_list)
        return dataset_length


class BlurPreprocessing(object):
    def __init__(self):
        self.mp = {"M": 9.0,  # magnification
                   "NA": 0.9,  # numerical aperture
                   "ng0": 1.48,  # coverslip RI design value
                   "ng": 1.48,  # coverslip RI experimental value
                   "ni0": 1.42,  # immersion medium RI design value
                   "ni": 1.42,  # immersion medium RI experimental value
                   "ns": 1.42,  # specimen refractive index (RI)
                   "ti0": 1000,  # microns, working distance (immersion medium thickness) design value
                   "tg": 300,  # microns, coverslip thickness experimental value
                   "tg0": 300,  # microns, coverslip thickness design value
                   "zd0": 6}  # microscope tube length (in microns).
        self.psf_kernel_size = [21, 31, 31]  # Z, X, Y

    @staticmethod
    def random_noise(high, rate_cln=1.0):
        noise_level = np.random.random() * high
        noise_mask = 0 if np.random.random() < rate_cln else 1
        return noise_level * noise_mask

    @staticmethod
    def GaussianNoising(img_in, sigma, mean=0.0, noise_size=None, min=0.0, max=1.0):
        if noise_size is None:
            size = img_in.shape
        else:
            size = noise_size
        noise = np.random.normal(loc=mean, scale=1.0, size=size) * sigma
        return np.clip(noise + img_in, a_min=min, a_max=max)

    @staticmethod
    def GaussianKernel2D(l, mu=0, alpha=1, sigma=1):
        axis_line = np.arange(-l // 2 + 1, l // 2 + 1)
        xx, yy = np.meshgrid(axis_line, axis_line)

        kernel = mu + alpha * np.exp(- (xx ** 2 + yy ** 2) / sigma ** 2)
        return kernel / np.sum(kernel)

    @staticmethod
    def convolve(img_in, kernel, mode='same'):
        return convolve(img_in, kernel, mode=mode)

    def gen_psf_kernel(self, psf_dim):
        psf_z, psf_x, psf_y = self.psf_kernel_size
        zv = np.arange(-psf_z//2, psf_z//2) + 1
        psf_xyz = msPSF.gLXYZFocalScan(self.mp, psf_x, psf_y, zv, pz=0.1)

        if psf_dim == 'zxy':
            _psf = psf_xyz
        elif psf_dim == 'xy':
            _psf = None
        elif psf_dim == 'zx':
            _psf = psf_xyz[:, :, psf_y // 2]
        else:
            _psf = None
        return _psf

    def __call__(self, img_in, psf_dim='zxy', return_kernel=False):
        psf_kernel = self.gen_psf_kernel(psf_dim=psf_dim)
        img_result = self.convolve(img_in, psf_kernel)

        return (img_result, psf_kernel) if return_kernel else img_result


if __name__ == '__main__':
    pass



