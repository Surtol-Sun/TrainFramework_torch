import os
import torch
import numpy as np
from PIL import Image

# This dataset comes form paper:
# [2D and 3D Segmentation of Uncertain Local Collagen Fiber Orientations in SHG Microscopy]
# https://github.com/Emprime/uncertain-fiber-segmentation

# This dataloader load dataset for unsupervised learning method Step2 proposed in
# Two-Stage Self-supervised Cycle-Consistency Network for Reconstruction of Thin-Slice MR Images


def collagen_dataset_TSCNet(dataloader_config):
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
        self.img_path = []

        for subfolder in os.listdir(data_path):
            subfolder_path = os.path.join(self.dataset_path, subfolder)
            # Sub-folders:
            # shg-ce-de: SHG image
            # shg-masks: SHG mask, do not care
            if subfolder == 'shg-ce-de':
                for root, dirs, files in os.walk(subfolder_path):
                    if not len(dirs) and len(files):
                        # print(root, dirs, files)
                        file_list = list(files)
                        file_list.sort(key=self._sort_num)
                        for i, file_name in enumerate(file_list):
                            try:
                                self.img_path.append([
                                    os.path.join(root, file_list[i]),
                                    os.path.join(root, file_list[i+1]),
                                    os.path.join(root, file_list[i+2]),
                                ])
                            except:
                                pass

        self.transform = transform
        self.target_transform = target_transform

    @staticmethod
    def _inner_rand_cut(img_in, cut_shape):
        shape_len = len(img_in.shape)
        for i, axis_len in enumerate(img_in.shape):
            cut_len = axis_len-cut_shape[i]
            if cut_len <= 0:
                continue
            cut_start = np.random.randint(0, cut_len)

            # img_in = img_in[cut_start:cut_start+cut_len]
            exe_script = f'img_in['
            for _ in range(i):
                exe_script += ':,'
            exe_script += 'cut_start:cut_start+cut_len,'
            for _ in range(shape_len-i-1):
                exe_script += ':,'
            exe_script += ']'

            exec(exe_script, {'img_in': img_in, 'cut_start': cut_start, 'cut_len': cut_len})
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
        image_path_list = self.img_path[index]
        for image_path in image_path_list:
            img = np.array(Image.open(image_path).convert('L'))  # [H, W]
            img = np.reshape(img, (1,) + img.shape)  # Convert gray image into [C, H, W] mode
            img_list.append(img)
        img_array = self._inner_rand_cut(np.array(img_list), (len(img_list),) + (1, 512, 512))  # [img, C. H, W]

        img_array = np.array(img_array/255., dtype=np.float32)
        if self.transform is not None:
            img_array = self.transform(img_array)
        img1, img2, img3 = img_array
        return img1, img2, img3

    def __len__(self):
        return len(self.img_path)



