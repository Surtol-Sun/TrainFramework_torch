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
                    self.img_index_list.append(list(range(index_base-stack_img_num, index_base)))
                    self.img_info_list.append({'path': img_path, 'index': i+1})

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
            if axis_len-cut_shape[i] <= 0:
                continue
            cut_start = np.random.randint(0, axis_len-cut_shape[i])

            # img_in = img_in[cut_start:cut_start+cut_len]
            exe_script = f'img_in=img_in[{":,"*i}cut_start:cut_start+cut_len,{":,"*(shape_len-i-1)}]'
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


if __name__ == '__main__':
    opt = {'datasets': {
                'train': {
                    'name': 'TPM_Skin', 'mode': 'TPM_MongoDB', 'dataroot_GT': '/home/yxsun/win_data/20211209Skin', 'dataroot_LQ': None, 'use_shuffle': True, 'n_workers': 8, 'batch_size': 32, 'GT_size': 256, 'LR_size': 64, 'use_flip': True, 'use_rot': True, 'color': 'RGB', 'phase': 'train', 'scale': 4, 'data_type': 'img'},
                'val': {
                    'name': 'TPM_Skin', 'mode': 'TPM_MongoDB', 'dataroot_GT': '/home/yxsun/win_data/20211209Skin', 'dataroot_LQ': None, 'phase': 'val', 'scale': 4, 'data_type': 'img'}}, 'network_G': {'which_model_G': 'SFTMD', 'in_nc': 3, 'out_nc': 3, 'nf': 64, 'nb': 16, 'upscale': 4, 'code_length': 10, 'scale': 4},
    }
    dataset = _DatasetLD(opt['datasets']['train'])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True,
                                num_workers=os.cpu_count(), drop_last=True,
                                pin_memory=False)
    for data in data_loader:
        pass

