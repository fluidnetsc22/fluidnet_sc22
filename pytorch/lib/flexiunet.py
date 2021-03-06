########################################################################################################################
#                                                                                                                      #
#                                  2.0 Simple flexible network  combining Unet and Msc                                 #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

import torch
import torch.nn as nn

from .scalesnet import ScalesNet

# Create the model


class _ConvBlock(nn.Module):
    """
    General convolution block for UNet. Depending on the location of the block
    in the architecture, the block can begin with a MaxPool3d (for bottom)
    or end with an UpSample or deconvolution layer (for up)
    """

    def __init__(self, fmaps, block_type, kernel_size, padding_mode='zeros',
                 upsample_mode='nearest', out_size=None):
        super(_ConvBlock, self).__init__()
        layers = list()
        # Apply pooling on down and bottom blocks
        if block_type == 'down' or block_type == 'bottom':
            layers.append(nn.MaxPool3d(2))

        # Append all the specified layers
        for i in range(len(fmaps) - 1):
            layers.append(nn.Conv3d(fmaps[i], fmaps[i + 1],
                                    kernel_size=kernel_size, padding=int((kernel_size - 1) / 2),
                                    padding_mode=padding_mode))
            # No ReLu at the very last layer
            if i != len(fmaps) - 2 or block_type != 'out':
                layers.append(nn.ReLU())

        # Apply either Upsample or deconvolution
        if block_type == 'up' or block_type == 'bottom':
            layers.append(nn.Upsample(out_size, mode=upsample_mode))

        # Build the sequence of layers
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class FlexiUNet(ScalesNet):
    """
    General UNet_new. All the layers are specified in the config file. Three different options are possible
    when going up the U: upsample, deconvolution or interpolation. Only interpolation
    allows the network to work on different resolutions
    """

    def __init__(self, scales, kernel_sizes, input_res,
                 padding_mode='zeros', upsample_mode='nearest'):
        super(FlexiUNet, self).__init__(scales, kernel_sizes)

        print('Input res ', input_res)

        if len(scales) == 1:
            self.monoblock = True
            self.OnlyBlock = nn.ModuleList()
            self.OnlyBlock.append(
                _ConvBlock(
                    self.scales['depth_0'],
                    'in',
                    self.kernel_sizes[0],
                    padding_mode=padding_mode))

        else:
            self.monoblock = False

            # create down_blocks, bottom_fmaps and up_blocks
            in_fmaps = self.scales['depth_0'][0]

            down_blocks = list()
            for local_depth in range(1, self.depth):
                down_blocks.append(self.scales[f'depth_{local_depth:d}'][0])

            bottom_fmaps = self.scales[f'depth_{self.depth:d}']

            up_blocks = list()
            for local_depth in range(self.depth - 1, 0, -1):
                up_blocks.append(self.scales[f'depth_{local_depth:d}'][1])

            out_fmaps = self.scales['depth_0'][1]

            # For upsample the list of resolution is needed when
            # the number of points is not a power of 2
            list_res = [(tuple(ti // 2**i for ti in input_res)) for i in range(self.depth)]

            # Entry layer
            self.ConvsDown = nn.ModuleList()
            self.ConvsDown.append(_ConvBlock(in_fmaps, 'in', self.kernel_sizes[0], padding_mode=padding_mode))

            # Intermediate down layers (with MaxPool at the beginning)
            for idown, down_fmaps in enumerate(down_blocks):
                self.ConvsDown.append(_ConvBlock(down_fmaps, 'down', self.kernel_sizes[idown + 1],
                                                 padding_mode=padding_mode))

            # Bottom layer (MaxPool at the beginning and Upsample/Deconv at the end)
            self.ConvBottom = _ConvBlock(bottom_fmaps,
                                         'bottom',
                                         self.kernel_sizes[-1],
                                         padding_mode=padding_mode,
                                         upsample_mode=upsample_mode,
                                         out_size=list_res.pop())

            # Intemediate layers up (UpSample/Deconv at the end)
            self.ConvsUp = nn.ModuleList()
            for iup, up_fmaps in enumerate(up_blocks):
                self.ConvsUp.append(_ConvBlock(up_fmaps,
                                               'up',
                                               self.kernel_sizes[-2 - iup],
                                               padding_mode=padding_mode,
                                               upsample_mode=upsample_mode,
                                               out_size=list_res.pop()))

            # Out layer
            self.ConvsUp.append(_ConvBlock(out_fmaps, 'out', self.kernel_sizes[0],
                                           padding_mode=padding_mode))

    def forward(self, x):
        if self.monoblock:
            Convblock = self.OnlyBlock[0]
            x = Convblock(x)
        else:
            # List of the temporary x that are used for linking with the up branch
            inputs_down = list()

            # Apply the down loop
            for ConvDown in self.ConvsDown:
                x = ConvDown(x)
                inputs_down.append(x)

            # Bottom part of the U
            x = self.ConvBottom(x)
            # Apply the up loop
            for ConvUp in self.ConvsUp:
                input_tmp = inputs_down.pop()
                x = ConvUp(torch.cat((x, input_tmp), dim=1))

        return x
