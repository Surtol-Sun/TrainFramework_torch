import os
import torch
import numpy as np
from skimage import io
from utils.utils import _normalization

# This dataset comes form my self-collection
# One folder contains some 3D tif files

# This dataloader load dataset for unsupervised learning method Step2 proposed in
# Two-Stage Self-supervised Cycle-Consistency Network for Reconstruction of Thin-Slice MR Images


def collagen_dataset_TSCNet_mytif(dataloader_config):
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
    def __init__(self, data_path, transform=None, target_transform=None):
        super().__init__()
        self.dataset_path = data_path
        self.img_path = []  # [[index1, index2, index3], ...], indexes of images in self.image_files
        self.image_files = []  # This loader load all images in RAM, which requires large RAM

        for file_name in os.listdir(data_path):
            if '.tif' not in file_name:
                continue
            img_file = io.imread(os.path.join(data_path, file_name))
            for i in range(img_file.shape[0]):
                self.image_files.append(_normalization(img_file[i], dtype=np.float32))
                if i >= 2:
                    index_base = len(self.image_files)
                    self.img_path.append([index_base-3, index_base-2, index_base-1])

        self.transform = transform
        self.target_transform = target_transform

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
        image_index_list = self.img_path[index]
        for image_index in image_index_list:
            img = self.image_files[image_index]
            img = np.reshape(img, (1,) + img.shape)  # Convert gray image into [C, H, W] mode
            img_list.append(img)
        img_array = self._inner_rand_cut(np.array(img_list), (len(img_list),) + (1, 128, 128))  # [img, C, H, W]

        if self.transform is not None:
            img_array = self.transform(img_array)
        img1, img2, img3 = img_array
        return img1, img2, img3

    def __len__(self):
        return len(self.img_path)



