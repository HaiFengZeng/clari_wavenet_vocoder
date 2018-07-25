# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import math
import librosa
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from .modules import Embedding, Conv1d1x1, ResidualConv1dGLU, ConvTranspose2d
from train import build_model
from wavenet_vocoder import receptive_field_size
from wavenet_vocoder.wavenet import _expand_global_features, WaveNet
from .mixture import sample_from_discretized_mix_logistic


class StudentWaveNet(nn.Module):

    def __init__(self, out_channels=2, layers=20, stacks=2,
                 residual_channels=64,
                 iaf_layer_sizes=[10, 10, 10, 30],
                 gate_channels=64,
                 kernel_size=3, dropout=1 - 0.95,
                 cin_channels=-1, gin_channels=-1, n_speakers=None,
                 weight_normalization=True,
                 upsample_conditional_features=False,
                 upsample_scales=None,
                 freq_axis_kernel_size=3,
                 scalar_input=False,
                 use_speaker_embedding=True,
                 ):
        super(StudentWaveNet, self).__init__()
        self.scalar_input = scalar_input
        self.out_channels = out_channels
        self.cin_channels = cin_channels
        self.last_layers = []

        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        if scalar_input:
            self.first_conv = nn.ModuleList([Conv1d1x1(1, residual_channels)
                                             for _ in range(len(iaf_layer_sizes))])
        else:
            self.first_conv = nn.ModuleList([Conv1d1x1(out_channels, residual_channels)
                                             for _ in range(len(iaf_layer_sizes))])

        self.iaf_layers = nn.ModuleList()
        self.last_layers = nn.ModuleList()

        for iaf_layer_size in iaf_layer_sizes:
            iaf_layer = nn.ModuleList()
            for layer_index in range(iaf_layer_size):
                dilation = 2 ** (layer_index % layers_per_stack)
                conv = ResidualConv1dGLU(
                    residual_channels,
                    gate_channels,
                    kernel_size=kernel_size,
                    bias=True,
                    dilation=dilation,
                    dropout=dropout,
                    cin_channels=cin_channels,
                    gin_channels=gin_channels,
                    weight_normalization=weight_normalization
                )
                iaf_layer.append(conv)
            self.iaf_layers.append(iaf_layer)
            self.last_layers.append(nn.ModuleList([
                nn.ReLU(),
                Conv1d1x1(residual_channels, out_channels, weight_normalization=weight_normalization)
            ]))

        if gin_channels > 0 and use_speaker_embedding:
            assert n_speakers is not None
            self.embed_speakers = Embedding(
                n_speakers, gin_channels, padding_idx=None, std=0.1)
        else:
            self.embed_speakers = None

            # Upsample conv net
        if upsample_conditional_features:
            self.upsample_conv = nn.ModuleList()
            for s in upsample_scales:
                freq_axis_padding = (freq_axis_kernel_size - 1) // 2
                convt = ConvTranspose2d(1, 1, (freq_axis_kernel_size, s),
                                        padding=(freq_axis_padding, 0),
                                        dilation=1, stride=(1, s),
                                        weight_normalization=weight_normalization)
                self.upsample_conv.append(convt)
                # assuming we use [0, 1] scaled features
                # this should avoid non-negative upsampling output
                self.upsample_conv.append(nn.ReLU(inplace=True))
        else:
            self.upsample_conv = None

        self.receptive_field = receptive_field_size(layers, stacks, kernel_size)

    def has_speaker_embedding(self):
        return self.embed_speakers is not None

    def local_conditioning_enabled(self):
        return self.cin_channels > 0

    def forward(self, z, c=None, g=None, softmax=False, use_cuda=True, use_scale=False):

        if c is not None and self.upsample_conv is not None:
            # B x 1 x C x T
            c = c.unsqueeze(1)
            for f in self.upsample_conv:
                c = f(c)
            # B x C x T
            c = c.squeeze(1)

        if z is None:  # for inference
            z = np.random.logistic(0, 1, (1, 1, c.size(-1)))
            z = torch.from_numpy(z).float()
            if use_cuda:
                z = z.cuda()

        assert c.size(-1) == z.size(-1)

        B, _, T = z.size()

        if g is not None:
            if self.embed_speakers is not None:
                # (B x 1) -> (B x 1 x gin_channels)
                g = self.embed_speakers(g.view(B, -1))
                # (B x gin_channels x 1)
                g = g.transpose(1, 2)
                assert g.dim() == 3
        # Expand global conditioning features to all time steps
        g_bct = _expand_global_features(B, T, g, bct=True)

        mu_tot = torch.zeros(z.size(), requires_grad=True)
        scale_tot = torch.ones(z.size(), requires_grad=True)
        if use_cuda:
            mu_tot, scale_tot = mu_tot.cuda(), scale_tot.cuda()

        for first_conv, iaf_layer, last_layer in zip(self.first_conv, self.iaf_layers, self.last_layers):
            new_z = first_conv(z)
            for f in iaf_layer:
                new_z, _ = f(new_z, c, g_bct)
            for f in last_layer:
                new_z = f(new_z)
            if use_scale:
                mu_s_f, scale_s_f = new_z[:, :1, :], new_z[:, 1:, :]
            else:
                mu_s_f, scale_s_f = new_z[:, :1, :], torch.exp(new_z[:, 1:, :])
            mu_tot = mu_s_f + mu_tot * scale_s_f
            scale_tot = scale_tot * scale_s_f
            z = z*scale_s_f + mu_s_f

        return z, mu_tot, scale_tot

    def clear_buffer(self):
        self.first_conv.clear_buffer()
        for f in self.conv_layers:
            f.clear_buffer()
        for f in self.last_conv_layers:
            try:
                f.clear_buffer()
            except AttributeError:
                pass

    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(remove_weight_norm)













