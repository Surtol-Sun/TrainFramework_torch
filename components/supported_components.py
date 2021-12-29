# Supported model
from components.models.unet3d.model import UNet3D
from components.models.unet.model import UNet
from components.models.unet.model_gan import UNet_Gan
from components.models.tscnet.model_step2 import TSCNetStep2
from components.models.ikc.IKC_model import IKC
supported_model_dict = {
    'unet2d': UNet(
        n_channels=1,
        n_classes=1,
    ),
    'unet2d_gan': UNet_Gan(
        in_channels=2,
        out_channels=1,
    ),
    'unet3d': UNet3D(
        in_channels=1,
        out_channels=1,
        is_segmentation=False,
    ),
    'TSCNetStep2': TSCNetStep2(
        in_channels=2,
        out_channels=1,
    ),
    'IKC': IKC(
        scale=1,
        input_channel=3,
        output_channel=3,
    ),
}

# Supported data loader
from components.dataset_loader.dataset_loader import cifar10_dataloader, cifar100_dataloader
from components.dataset_loader.dataset_loader_3dcollagen import collagen3d_dataset
from components.dataset_loader.dataset_loader_collagen_TSCNet import collagen_dataset_TSCNet
from components.dataset_loader.dataset_loader_collagen_TSCNet_mytif import collagen_dataset_TSCNet_mytif
from components.dataset_loader.dataset_loader_mydatabase import dataset_mydatabase
from components.dataset_loader.dataset_loader_mydatabase_IKC import dataset_mydatabase_IKC
from components.dataset_loader.dataset_loader_mydatabase_IKCz import dataset_mydatabase_IKCz
supported_dataloader_dict = {
    'cifar10': cifar10_dataloader,
    'cifar100': cifar100_dataloader,
    'collagen3d_dataset': collagen3d_dataset,
    'collagen_dataset_TSCNet': collagen_dataset_TSCNet,
    'collagen_dataset_TSCNet_mytif': collagen_dataset_TSCNet_mytif,
    'dataset_mydatabase': dataset_mydatabase,
    'dataset_mydatabase_IKC': dataset_mydatabase_IKC,
    'dataset_mydatabase_IKCz': dataset_mydatabase_IKCz,
}

# Supported training strategy
from components.train_strategy.train_strategy import RegularTrain
from components.train_strategy.train_strategy_TSCNet import TrainTSCNet
from components.train_strategy.train_strategy_TSCNet_pre import TrainTSCNet_pre
from components.train_strategy.train_strategy_IKC import TrainIKC
from components.train_strategy.train_strategy_IKCz import TrainIKCz
supported_training_strategy_dict = {
    'regular': RegularTrain,
    'TrainTSCNet': TrainTSCNet,
    'TrainTSCNet_pre': TrainTSCNet_pre,
    'TrainIKC': TrainIKC,
    'TrainIKCz': TrainIKCz,
}


