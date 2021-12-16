import h5py
from skimage import io

'''
This tool is used to extract images in h5 file to some specific folder.
'''

file_path = r'/home/yxsun/win_data/20210720Arabidopsis-thaliana-lateral-root/Light-sheet/test/Movie1_t00006_crop_gt.h5'

f = h5py.File(file_path, 'r')
print(f'Keys in this file: {list(f.keys())}')
for key in list(f.keys()):
    file_name = f'{key}.tif'
    io.imsave(file_name, f[key])


