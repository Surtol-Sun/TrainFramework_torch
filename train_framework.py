import torch
import argparse

from utils.global_config import *
from utils.utils import load_config, set_use_cuda, print_log
from utils.supported_items import supported_loss_dict, supported_model_dict, supported_dataloader_dict, supported_training_strategy_dict


parser = argparse.ArgumentParser(description='3D Segmentation')
parser.add_argument('--config', type=str, help='Path to the YAML config file', default=r'config_scripts/TSCNet_step2.yml')
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

    # Define loss function (criterion) and optimizer ------------------------------------------------
    if use_cuda:
        model = model.cuda()
        criterion = supported_loss_dict[model_config['loss_name']]
        criterion = criterion.cuda()
    else:
        criterion = supported_loss_dict[model_config['loss_name']]

    optimizer = torch.optim.SGD(model.parameters(), 0.01,  # ToDo!!!!!!!!!!!!!
                                momentum=0.9,  # ToDo!!!!!!!!!!!!!
                                weight_decay=1e-4,  # ToDo!!!!!!!!!!!!
                                nesterov=True)

    # Data loading code -----------------------------------------------------------------------------
    dataset_config = config['dataset_config']
    dataloader = supported_dataloader_dict[dataset_config['loader_name']]
    train_loader, val_loader = dataloader(dataset_config)

    # Train strategy --------------------------------------------------------------------------------
    train_config = config['train_config']
    train_strategy = supported_training_strategy_dict[train_config['strategy_name']](
        model, criterion, optimizer, train_loader, train_config)

    # Train -----------------------------------------------------------------------------------------
    train_strategy.train()


if __name__ == '__main__':
    main()
