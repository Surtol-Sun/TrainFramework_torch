import os
import torch
import numpy as np
from PIL import Image

# This dataset comes form paper:
# [2D and 3D Segmentation of Uncertain Local Collagen Fiber Orientations in SHG Microscopy]
# https://github.com/Emprime/uncertain-fiber-segmentation


def collagen3d_dataset(dataloader_config, label_type='mask'):
    # Ref -- https://blog.csdn.net/Teeyohuang/article/details/79587125
    # label_type: 'classlabel' or 'mask'

    dataset_path = dataloader_config['dataset_path']
    train_batch_size = dataloader_config['train_batch_size']
    val_batch_size = dataloader_config['val_batch_size']
    num_workers = os.cpu_count()
    # num_workers = 1

    train_data = _DatasetLD(data_path=dataset_path, dataset_return=label_type, read_in_ram_mode=True)
    test_data = _DatasetLD(data_path=dataset_path, dataset_return=label_type, read_in_ram_mode=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=val_batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True)

    # return train_data, test_data
    return train_loader, val_loader


class _DatasetLD(torch.utils.data.Dataset):
    def __init__(self, data_path, dataset_return, transform=None, target_transform=None, read_in_ram_mode=False):
        super().__init__()
        self.dataset_path = data_path
        self.dataset_return = dataset_return

        self.read_in_ram_mode = read_in_ram_mode

        self.img_name = []
        self.num_label = []

        for subfolder in os.listdir(data_path):
            subfolder_path = os.path.join(self.dataset_path, subfolder)
            # Sub-folders:
            # shg-ce-de: SHG image
            # shg-masks: SHG mask
            if subfolder == 'shg-ce-de':
                for root, dirs, files in os.walk(subfolder_path):
                    if not len(dirs) and len(files):
                        # print(root, dirs, files)
                        self.img_name.append(root)

        # Read all images in RAM, which requires a large RAM
        if self.read_in_ram_mode:
            img_all_list, mask_all_list = [], []
            for i, index in enumerate(range(len(self.img_name))):
                print(f'Reading image [{i+1}]/[{len(self.img_name)}]')
                image_folder = self.img_name[index]
                mask_folder = self.img_name[index].replace('shg-ce-de', 'shg-masks')
                img_list, mask_list = [], []
                img_file_list, mask_file_list = list(os.listdir(image_folder)), list(os.listdir(mask_folder))

                img_file_list.sort(key=self._sort_num)
                for img_name in img_file_list:
                    img = np.array(Image.open(os.path.join(image_folder, img_name)).convert('L'))  # [H, W]
                    img = np.reshape(img, img.shape + (1,))  # Convert gray image into [H, W, C] mode
                    # img = np.array(Image.open(os.path.join(image_folder, img_name)))
                    img_list.append(img)

                mask_file_list.sort(key=self._sort_num)
                for mask_name in mask_file_list:
                    mask_list.append(np.array(Image.open(os.path.join(mask_folder, mask_name))))  # [H, W, C]

                img_all_list.append(img_list)
                mask_all_list.append(mask_list)
            self.img_name, self.num_label = img_all_list, mask_all_list
        self.transform = transform
        self.target_transform = target_transform

    @staticmethod
    def _inner_rand_cut(img_in, cut_start):
        h, w = img_in.shape
        if h > w:
            return img_in[cut_start:cut_start+w, :, :]
        else:
            return img_in[:, cut_start:cut_start+h, :]

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
        if self.dataset_return == 'mask':
            return self._getitem_mask(index)
        elif self.dataset_return == 'classlabel':
            return self._getitem_label(index)
        else:
            return

    def _getitem_label(self, index):  # Read data once
        # Todo !!!!!!!!! Not written
        file_name = self.img_name[index]
        label = self.num_label[index]
        img = Image.open(os.path.join(self.dataset_path, 'image', file_name))
        img = np.array(img)

        # Random cut
        h, w = img.shape
        cut_start = np.random.randint(0, abs(h-w))
        img = self._inner_rand_cut(img, cut_start)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def _getitem_mask(self, index):  # Read data once
        if self.read_in_ram_mode:
            img_list = self.img_name[index]
            mask_list = self.num_label[index]
        else:
            image_folder = self.img_name[index]
            mask_folder = self.img_name[index].replace('shg-ce-de', 'shg-masks')
            img_list, mask_list = [], []
            img_file_list, mask_file_list = list(os.listdir(image_folder)), list(os.listdir(mask_folder))

            img_file_list.sort(key=self._sort_num)
            for img_name in img_file_list:
                img = np.array(Image.open(os.path.join(image_folder, img_name)).convert('L'))  # [H, W]
                img = np.reshape(img, img.shape + (1,))  # Convert gray image into [H, W, C] mode
                # img = np.array(Image.open(os.path.join(image_folder, img_name)))
                img_list.append(img)

            mask_file_list.sort(key=self._sort_num)
            for mask_name in mask_file_list:
                mask_list.append(np.array(Image.open(os.path.join(mask_folder, mask_name))))  # [H, W, C]
        img = np.array(img_list).transpose([3, 1, 2, 0])  # Convert from [D, H, W, C] into [C, H, W, D] mode
        mask = np.array(mask_list)
        mask = np.max(mask, axis=3)  # Convert mask to label
        mask = np.transpose(mask, [1, 2, 0])  # Convert from [D, H, W] into [H, W, D] mode

        # ToDo Temp
        _, h, w, d = img.shape
        new_size = 64
        new_depth = 32
        h_random, w_random, d_random = np.random.randint(0, h-new_size), np.random.randint(0, w-new_size), np.random.randint(0, d-new_depth)
        img = img[:, h_random:h_random+new_size, w_random:w_random+new_size, d_random:d_random+new_depth]
        mask = mask[h_random:h_random+new_size, w_random:w_random+new_size, d_random:d_random+new_depth]


        ######################################
        # Should cut with mask here
        ######################################
        # # Random cut
        # h, w = img.shape
        # if np.abs(h - w):
        #     cut_start = np.random.randint(0, abs(h - w))
        #     img = self._inner_rand_cut(img, cut_start)
        #     mask = self._inner_rand_cut(mask, cut_start)

        if self.transform is not None:
            img = self.transform(img)
        return np.array(img, dtype=np.float32), np.array(mask/255., dtype=np.int64)

    def __len__(self):
        return len(self.img_name)



