# Supported model
import components.models.smp as smp
from components.models.unet3d.model import UNet3D
supported_model_dict = {
    'unet2d': smp.Unet(
        encoder_name='resnet18',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,           # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,                      # model output channels (number of classes in your dataset)
    ),
    'unet3d': UNet3D(
        in_channels=1,
        out_channels=1,
    ),
    'unet2d_TSCNet': smp.Unet(
        encoder_name='resnet18',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,           # use `imagenet` pre-trained weights for encoder initialization
        in_channels=2,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    ),
}

# Supported loss
import torch
from components.losses.dice import DiceLoss
supported_loss_dict = {
    'Dice': DiceLoss(mode='multilabel'),
    'CrossEntropy': torch.nn.CrossEntropyLoss(),
    'MSE': torch.nn.MSELoss(reduce=True, size_average=True),
}

# Supported data loader
from components.dataset_loader.dataset_loader import cifar10_dataloader, cifar100_dataloader
from components.dataset_loader.dataset_loader_3dcollagen import collagen3d_dataset
from components.dataset_loader.dataset_loader_collagen_TSCNet import collagen_dataset_TSCNet
from components.dataset_loader.dataset_loader_collagen_TSCNet_mytif import collagen_dataset_TSCNet_mytif
supported_dataloader_dict = {
    'cifar10': cifar10_dataloader,
    'cifar100': cifar100_dataloader,
    'collagen3d_dataset': collagen3d_dataset,
    'collagen_dataset_TSCNet': collagen_dataset_TSCNet,
    'collagen_dataset_TSCNet_mytif': collagen_dataset_TSCNet_mytif,
}

# Supported training strategy
from components.train_strategy.train_strategy import RegularTrain
from components.train_strategy.train_strategy_TSCNet import TrainTSCNet
supported_training_strategy_dict = {
    'regular': RegularTrain,
    'TrainTSCNet': TrainTSCNet,
}


