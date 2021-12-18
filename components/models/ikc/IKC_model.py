import torch.nn as nn
from components.models.ikc.sftmd_arch import SFTMD, SFTMD_DEMO, Predictor, Corrector


class IKC(nn.Module):
    def __init__(self, input_channel, output_channel, code_length=10, nf=64, nb=16, scale=4, input_para=10):
        super().__init__()
        self.SFTMD = SFTMD(in_nc=input_channel, out_nc=output_channel, nf=nf, nb=nb, scale=scale, input_para=input_para)
        self.Predictor = Predictor(in_nc=input_channel, code_len=code_length)
        self.Corrector = Corrector(in_nc=input_channel, code_len=code_length)

    def forward(self, x):
        kernel_code = self.Predictor(x)
        sr_img = self.SFTMD(x, kernel_code)
        return sr_img, kernel_code

