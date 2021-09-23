import os
import torch
import numpy as np
from PIL import Image


def collagen_dataset(dataloader_config, label_type='mask'):
    # Ref -- https://blog.csdn.net/Teeyohuang/article/details/79587125
    # label_type: 'classlabel' or 'mask'

    dataset_path = dataloader_config['dataset_path']
    train_batch_size = dataloader_config['train_batch_size']
    val_batch_size = dataloader_config['val_batch_size']
    # num_workers = os.cpu_count()
    num_workers = 1

    train_data = _DatasetLD(data_path=dataset_path, dataset_return=label_type)
    test_data = _DatasetLD(data_path=dataset_path, dataset_return=label_type)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=val_batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True)

    # return train_data, test_data
    return train_loader, val_loader


class _DatasetLD(torch.utils.data.Dataset):
    def __init__(self, data_path, dataset_return, transform=None, target_transform=None):
        super().__init__()
        self.dataset_path = data_path
        self.dataset_return = dataset_return

        self.img_name = []
        self.num_label = []

        for subfolder in os.listdir(data_path):
            subfolder_path = os.path.join(self.dataset_path, subfolder)
            if subfolder == 'image':
                for img_name in os.listdir(subfolder_path):
                    self.img_name.append(img_name)

        # with open(os.path.join(self.dataset_path, 'train.csv'), 'r') as f:
        #     csv_r = csv.reader(f)
        #     for file_name, label in csv_r:
        #         if 'png' not in file_name and 'PNG' not in file_name:
        #             continue
        #         self.img_name.append(file_name)
        #         self.num_label.append(label)

        self.transform = transform
        self.target_transform = target_transform

    @staticmethod
    def _inner_rand_cut(img_in, cut_start):
        h, w = img_in.shape
        if h > w:
            return img_in[cut_start:cut_start+w, :, :]
        else:
            return img_in[:, cut_start:cut_start+h, :]

    def __getitem__(self, index):  # Read data once
        if self.dataset_return == 'mask':
            return self._getitem_mask(index)
        elif self.dataset_return == 'classlabel':
            return self._getitem_label(index)
        else:
            return

    def _getitem_label(self, index):  # Read data once
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
        file_name = self.img_name[index]
        img = Image.open(os.path.join(self.dataset_path, 'image', file_name)).convert('L')
        mask = Image.open(os.path.join(self.dataset_path, 'mask', file_name))
        img = np.array(img)
        mask = np.array(mask)
        mask = np.transpose(mask, [2, 0, 1])  # Convert original [h, w, d] mask to [d, h, w]
        mask = np.max(mask, axis=0)  # Convert mask to label

        ######################################
        # Should cut with mask here
        ######################################
        # Random cut
        h, w = img.shape
        if np.abs(h - w):
            cut_start = np.random.randint(0, abs(h - w))
            img = self._inner_rand_cut(img, cut_start)
            mask = self._inner_rand_cut(mask, cut_start)

        if self.transform is not None:
            img = self.transform(img)
        return np.array([img], dtype=np.float32), np.array(mask/255., dtype=np.int64)

    def __len__(self):
        return len(self.img_name)



