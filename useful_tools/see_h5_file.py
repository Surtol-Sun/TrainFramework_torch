import h5py
import numpy as np
from skimage import io

'''
This tool is used to extract images in h5 file to some specific folder.
'''

file_path = r'imgs.hdf5'

f = h5py.File(file_path, 'r')
print(f'Keys in this file: {list(f.keys())}')

data_dict = {}
for key, item in f.items():
    print(f'Reading data {item} ...')
    data_dict[key] = item[()]
    print(f'Data [{key}] with shape {data_dict[key].shape}')

for i in range(362):
    image = data_dict['volData'][:, :, :, i]  # [x, x, z]
    image = np.transpose(image, [2, 0, 1])
    io.imsave(f'{i}.tif', image)



