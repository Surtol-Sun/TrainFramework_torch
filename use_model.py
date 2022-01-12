import torch
import argparse
import numpy as np

from utils.utils import load_config, set_use_cuda, print_log, _normalization
from components.supported_components import supported_model_dict


parser = argparse.ArgumentParser(description='3D Segmentation')
parser.add_argument('--config', type=str, help='Path to the YAML config file', default=r'config_scripts/IKC_use.yml')
args = parser.parse_args()


def main():
    config = load_config(args.config)
    use_cuda = False if str(config['device']) == 'cpu' else True
    set_use_cuda(use_cuda)
    print(f'Use cuda: {str(use_cuda)}')

    # Load model ----------------------------------------------------------------------------------
    model_config = config['model_config']
    model = supported_model_dict[model_config['model_name']]
    print_log(model)

    # Load model parameters ------------------------------------------------------------------------
    model_parameter_path = model_config.get('checkpoint_path', None)
    if model_parameter_path:
        checkpoint = torch.load(model_parameter_path, map_location=torch.device(config['device']))
        model.load_state_dict(checkpoint['state_dict'])
        config['train_config']['epoch'] = checkpoint['epoch']
    else:
        print_log('No checkpoint loaded!')
        exit(-1)

    if use_cuda:
        model = model.cuda()

    # Generate result -------------------------------------------------------------------------------
    img_hr_list = []
    from skimage import io
    # data_tif = io.imread(r'/home/yxsun/win_data/00000000Scan200nm/183058&183844.tif')  # ZXY
    data_tif = io.imread(r'/home/yxsun/win_data/00000000Scan200nm/160645.tif')  # ZXY

    # Blur
    def GaussianKernel2D(l, mu=0, alpha=1, sigma=1):
        axis_line = np.arange(-l // 2 + 1, l // 2 + 1)
        xx, yy = np.meshgrid(axis_line, axis_line)

        kernel = mu + alpha * np.exp(- (xx ** 2 + yy ** 2) / sigma ** 2)
        return kernel / np.sum(kernel)
    from scipy.signal import convolve
    for i, img_i in enumerate(data_tif):
        data_tif[i] = convolve(img_i, GaussianKernel2D(l=11, sigma=2), mode='same')
    io.imsave('data_use.tif', data_tif)

    # Use model
    for i in range(data_tif.shape[0]-4):
        print(f'Handling with image {i}')
        data_use = data_tif[i:i+4, :, :]
        # data_use = data_tif[6:9, :256, :256]
        data_use = _normalization(data_use, dtype=np.float32)

        data_use = np.array([data_use])  # (B, S, H, W)

        if use_cuda:
            data_use = torch.Tensor(data_use).cuda()
        else:
            data_use = torch.Tensor(data_use)
        img_hr = model.generator(data_use)
        img_hr = img_hr.detach().cpu().numpy()[0]
        img_hr_list.append(np.average(img_hr, axis=0))
        # io.imsave('img_hr.tif', img_hr)
    io.imsave('img_hr_list.tif', np.stack(img_hr_list, axis=0))
    exit()

    # result_list = []
    # with torch.no_grad():
    #     for i in range(data_tif.shape[0]-1):
    #         # img1 = np.reshape(data_tif[i], data_tif[i].shape + (1,))
    #         # img2 = np.reshape(data_tif[i+1], data_tif[i+1].shape + (1,))
    #
    #         img1 = _normalization(data_tif[i], dtype=np.float32)
    #         img2 = _normalization(data_tif[i+1], dtype=np.float32)
    #
    #         img = torch.Tensor([[img1, img2]])
    #         if use_cuda:
    #             img = img.cuda()
    #         output = model(img)
    #
    #         result_list.append(img1)
    #         result_list.append(np.squeeze(output.cpu().numpy()))
    #         result_list.append(img2)
    # io.imsave('aaa.tif', np.array(result_list))

    result_list = []
    with torch.no_grad():
        for i in range(data_tif.shape[0] - 2):
            # img1 = np.reshape(data_tif[i], data_tif[i].shape + (1,))
            # img2 = np.reshape(data_tif[i+1], data_tif[i+1].shape + (1,))

            img1 = _normalization(data_tif[i], dtype=np.float32)
            img2 = _normalization(data_tif[i + 1], dtype=np.float32)
            img3 = _normalization(data_tif[i + 2], dtype=np.float32)

            img1_5 = torch.Tensor([[img1, img2]])
            img2_5 = torch.Tensor([[img2, img3]])
            if use_cuda:
                img1_5 = img1_5.cuda()
                img2_5 = img2_5.cuda()
            output1 = model(img1_5)
            output2 = model(img2_5)
            output = model(torch.cat([output1, output2], 1))

            mse = np.std(np.squeeze(output.cpu().numpy()) - img2)
            print_log(f'mse={mse}')



if __name__ == '__main__':
    main()


