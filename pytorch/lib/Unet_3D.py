import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Create the model


class _ConvBlock1(nn.Module):
    """
    First block - quarter scale.
    Four Conv3d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    Optional dropout before final Conv3d layer
    ReLU after first two Conv3d layers, not after last two - predictions can be +ve or -ve
    """

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock1, self).__init__()
        layers = [
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad3d(1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(),
        ]

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _ConvBlock2(nn.Module):
    """
    Second block - half scale.
    Six Conv3d layers. First one kernel size 5, padding 2, remainder kernel size 3 padding 1.
    Optional dropout before final Conv3d layer
    ReLU after first four Conv3d layers, not after last two - predictions can be +ve or -ve
    """

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock2, self).__init__()
        layers = [
            nn.MaxPool3d(2),
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad3d(1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _ConvBlock3(nn.Module):
    """
    Third block - full scale.
    Six Conv3d layers. First and last kernel size 5, padding 2, remainder kernel size 3 padding 1.
    Optional dropout before final Conv3d layer
    ReLU after first four Conv3d layers, not after last two - predictions can be +ve or -ve
    """

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock3, self).__init__()
        layers = [
            nn.MaxPool3d(2),
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad3d(1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _ConvBlock4(nn.Module):
    """
    Third block - full scale.
    Six Conv3d layers. First and last kernel size 5, padding 2, remainder kernel size 3 padding 1.
    Optional dropout before final Conv3d layer
    ReLU after first four Conv3d layers, not after last two - predictions can be +ve or -ve
    """

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock4, self).__init__()
        layers = [
            nn.MaxPool3d(2),
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad3d(1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _ConvBlock5(nn.Module):
    """
    Third block - full scale.
    Six Conv3d layers. First and last kernel size 5, padding 2, remainder kernel size 3 padding 1.
    Optional dropout before final Conv3d layer
    ReLU after first four Conv3d layers, not after last two - predictions can be +ve or -ve
    """

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock5, self).__init__()
        layers = [
            nn.MaxPool3d(2),
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad3d(1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _ConvBlock6(nn.Module):
    """
    Third block - full scale.
    Six Conv3d layers. First and last kernel size 5, padding 2, remainder kernel size 3 padding 1.
    Optional dropout before final Conv3d layer
    ReLU after first four Conv3d layers, not after last two - predictions can be +ve or -ve
    """

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock6, self).__init__()
        layers = [
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad3d(1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _ConvBlock7(nn.Module):
    """
    Third block - full scale.
    Six Conv3d layers. First and last kernel size 5, padding 2, remainder kernel size 3 padding 1.
    Optional dropout before final Conv3d layer
    ReLU after first four Conv3d layers, not after last two - predictions can be +ve or -ve
    """

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock7, self).__init__()
        layers = [
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad3d(1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _ConvBlock8(nn.Module):
    """
    Third block - full scale.
    Six Conv3d layers. First and last kernel size 5, padding 2, remainder kernel size 3 padding 1.
    Optional dropout before final Conv3d layer
    ReLU after first four Conv3d layers, not after last two - predictions can be +ve or -ve
    """

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock8, self).__init__()
        layers = [
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad3d(1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _ConvBlock9(nn.Module):
    """
    Third block - full scale.
    Six Conv3d layers. First and last kernel size 5, padding 2, remainder kernel size 3 padding 1.
    Optional dropout before final Conv3d layer
    ReLU after first four Conv3d layers, not after last two - predictions can be +ve or -ve
    """

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock9, self).__init__()
        layers = [
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad3d(1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _ConvBlock10(nn.Module):
    """
    Third block - full scale.
    Six Conv3d layers. First and last kernel size 5, padding 2, remainder kernel size 3 padding 1.
    Optional dropout before final Conv3d layer
    ReLU after first four Conv3d layers, not after last two - predictions can be +ve or -ve
    """

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock10, self).__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class UNet3D(nn.Module):
    """
    Define the network. Only input when called is number of data (input) channels.
        -Downsample input to quarter scale and use ConvBlock1.
        -Upsample output of ConvBlock1 to half scale.
        -Downsample input to half scale, concat to output of ConvBlock1; use ConvBLock2.
        -Upsample output of ConvBlock2 to full scale.
        -Concat input to output of ConvBlock2, use ConvBlock3. Output of ConvBlock3 has 8 channels
        -Use final Conv3d layer with kernel size of 1 to go from 8 channels to 1 output channel.
    """

    def __init__(self, data_channels):
        super(UNet3D, self).__init__()
        self.convN_1 = _ConvBlock1(data_channels, 32)
        self.convN_2 = _ConvBlock2(32, 32)
        self.convN_3 = _ConvBlock3(32, 32)
        self.convN_4 = _ConvBlock4(32, 64)
        self.convN_5 = _ConvBlock5(64, 64)
        self.convN_6 = _ConvBlock6(128, 64)
        self.convN_7 = _ConvBlock7(96, 64)
        self.convN_8 = _ConvBlock8(96, 32)
        self.convN_9 = _ConvBlock9(64, 32)
        self.final = _ConvBlock10(32, 1)

    def forward(self, x):

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        convN_1out = self.convN_1(x)
        convN_2out = self.convN_2(convN_1out)
        convN_3out = self.convN_3(convN_2out)
        convN_4out = self.convN_4(convN_3out)
        convN_5out = self.convN_5(convN_4out)
        convN_6out = self.convN_6(
            torch.cat(
                (F.interpolate(
                    convN_5out,
                    scale_factor=2,
                    mode='trilinear'),
                    convN_4out),
                dim=1))
        convN_7out = self.convN_7(
            torch.cat(
                (F.interpolate(
                    convN_6out,
                    scale_factor=2,
                    mode='trilinear'),
                    convN_3out),
                dim=1))
        convN_8out = self.convN_8(
            torch.cat(
                (F.interpolate(
                    convN_7out,
                    scale_factor=2,
                    mode='trilinear'),
                    convN_2out),
                dim=1))
        convN_9out = self.convN_9(
            torch.cat(
                (F.interpolate(
                    convN_8out,
                    scale_factor=2,
                    mode='trilinear'),
                    convN_1out),
                dim=1))
        final_out = self.final(convN_9out)

        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        elapsed_time_ms = start_event.elapsed_time(end_event)
        #print(f'Elapsed time {elapsed_time_ms}')

        return final_out
