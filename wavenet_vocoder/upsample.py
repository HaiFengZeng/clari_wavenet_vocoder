import torch
from torch import nn
from hparams import hparams
from wavenet_vocoder.modules import ConvTranspose2d
import os
from hparams import hparams

class UpSampleConv(nn.Module):
    def __init__(self,
                 path=None,
                 share_condition=True,
                 weight_normalization=True):
        super(UpSampleConv, self).__init__()
        self.path = path
        self.upsample_conv = nn.ModuleList()
        for s in hparams.upsample_scales:
            freq_axis_padding = (hparams.freq_axis_kernel_size - 1) // 2
            convt = ConvTranspose2d(1, 1, (hparams.freq_axis_kernel_size, s),
                                    padding=(freq_axis_padding, 0),
                                    dilation=1, stride=[1, s],
                                    weight_normalization=weight_normalization)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(inplace=True,negative_slope=0.2))
        # load condition form teacher wavenet
        if path and share_condition:
            self.load()

    def forward(self, c):
        for f in self.upsample_conv:
            c = f(c)
        return c

    def load(self):
        if self.path and os.path.exists(self.path):
            self.upsample_conv.load_state_dict(torch.load(self.path))
        else:
            raise Exception("can't load state dict, check path, see get_model in train_student.py !")


class ClariUpsampleConv(nn.Module):
    def __init__(self, weight_normalization=True):
        super(ClariUpsampleConv, self).__init__()
        self.upsample_conv = nn.ModuleList()
        for s in hparams.upsample_size:
            convt = ConvTranspose2d(1, 1, kernel_size=s, stride=(1,s[0] / 2), weight_normalization=weight_normalization)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(inplace=True, negative_slope=0.4))

    def forward(self, c):
        for f in self.upsample_conv:
            c = f(c)
        return c

if __name__ == '__main__':
    checkpoint = torch.load('/home/jinqiangzeng/work/mypycharm/wavenet/clari_wavenet_vocoder/checkpoints/checkpoint_step000430000_ema.pth')
    preset = '/home/jinqiangzeng/work/mypycharm/wavenet/clari_wavenet_vocoder/presets/ljspeech_gaussian.json'
    with open(preset) as f:
        hparams.parse_json(f.read())
    from train_student import build_model
    teacher = build_model(hparams,'teacher')
    teacher.load_state_dict(checkpoint['state_dict'])
    upsample_state_dict = teacher.upsample_conv.state_dict()
    upsample_conv = UpSampleConv()
    upsample_conv.load_state_dict(upsample_state_dict)
    for para in upsample_conv.parameters():
        para.requires_grad=False