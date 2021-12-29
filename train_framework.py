import torch
import argparse

from utils.utils import load_config, set_use_cuda, print_log
from components.supported_components import supported_model_dict, supported_dataloader_dict, supported_training_strategy_dict


parser = argparse.ArgumentParser(description='3D Segmentation')
parser.add_argument('--config', type=str, help='Path to the YAML config file', default=r'config_scripts/IKCz.yml')
args = parser.parse_args()


def main():
    config = load_config(args.config)
    use_cuda = False if str(config['device']) == 'cpu' else True
    set_use_cuda(use_cuda)
    print_log(f'Use cuda: {str(use_cuda)}')

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

    # Define loss function (criterion) --------------------------------------------------------------
    if use_cuda:
        model = model.cuda()

    while True:
        # Data loading code -------------------------------------------------------------------------
        dataset_config = config['dataset_config']
        dataloader = supported_dataloader_dict[dataset_config['loader_name']]
        train_loader, val_loader = dataloader(dataset_config)

        # Train strategy ----------------------------------------------------------------------------
        train_config = config['train_config']
        train_strategy = supported_training_strategy_dict[train_config['strategy_name']](
            model, train_loader, val_loader, train_config)

        # Train -------------------------------------------------------------------------------------
        train_successfully = True
        try:
            train_strategy.train()
        except RuntimeError as e:
            train_successfully = False
            if 'CUDA out of memory' in str(e):
                train_batch_size = config['dataset_config']['train_batch_size']
                print_log(f'Train batch = {train_batch_size}, {str(e)}')
                if train_batch_size > 1:
                    config['dataset_config']['train_batch_size'] -= 1
                    print_log(f'Change train batch size to {train_batch_size-1}')
                else:
                    print_log('Failed, no enough CUDA memory')
                    break
            else:
                print_log(f'Unexpected error:')
                print_log(f'{str(e)}')
                break

        if train_successfully:
            break
    print_log('------------------------ END ------------------------')



if __name__ == '__main__':
    main()
