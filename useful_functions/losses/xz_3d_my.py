import torch
from torch import Tensor
from torch.nn import MSELoss
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class XZ_3D_MY_Loss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        alpha, beta, gamma = 1.0, 1.0, 1.0
        alpha = alpha / (alpha + beta + gamma)
        beta = alpha / (alpha + beta + gamma)
        gamma = alpha / (alpha + beta + gamma)

        # input and target: (B, S, H, W)
        mse_loss = F.mse_loss(input, target, reduction=self.reduction)
        layer_loss = torch.std(torch.mean(input, dim=[2, 3], keepdim=False), dim=-1, keepdim=False)
        simi_loss = torch.mean(torch.std(input, dim=1, keepdim=False), dim=[1, 2], keepdim=False)
        return torch.mean(alpha * mse_loss + beta * layer_loss + gamma * simi_loss)



