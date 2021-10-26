"""
modified from torchvision.models.densenet
2d layers are replaced with 3d layers,
and parameter [norm_layer] is added so you can choose the type of normalization layer ("bn"-BatchNorm3d / "in"-InstanceNorm3d)
"""

import torch
import torch.nn as nn

#"""Bottleneck layers. Although each layer only produces k
#output feature-maps, it typically has many more inputs. It
#has been noted in [37, 11] that a 1×1 convolution can be in-
#troduced as bottleneck layer before each 3×3 convolution
#to reduce the number of input feature-maps, and thus to
#improve computational efficiency."""
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout_rate, norm_layer):
        super().__init__()
        #"""In  our experiments, we let each 1×1 convolution
        #produce 4k feature-maps."""
        inner_channel = 4 * growth_rate

        #"""We find this design especially effective for DenseNet and
        #we refer to our network with such a bottleneck layer, i.e.,
        #to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` ,
        #as DenseNet-B."""
        self.bottle_neck = nn.Sequential(
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, inner_channel, kernel_size=1, bias=False),
            norm_layer(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

#"""We refer to layers between blocks as transition
#layers, which do convolution and pooling."""
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super().__init__()
        #"""The transition layers used in our experiments
        #consist of a batch normalization layer and an 1×1
        #convolutional layer followed by a 2×2 average pooling
        #layer""".
        self.down_sample = nn.Sequential(
            norm_layer(in_channels),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool3d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, in_channel=3, num_class=100, init_weights=False, dropout_rate=0, norm_layer="bn"):
        super().__init__()
        if norm_layer == 'bn':
            self.norm_layer = nn.BatchNorm3d
        elif norm_layer == 'in':
            self.norm_layer = nn.InstanceNorm3d
        
        self.growth_rate = growth_rate

        #"""Before entering the first dense block, a convolution
        #with 16 (or twice the growth rate for DenseNet-BC)
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        #For convolutional layers with kernel size 3×3, each
        #side of the inputs is zero-padded by one pixel to keep
        #the feature-map size fixed.
        self.conv1 = nn.Conv3d(in_channel, inner_channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index], dropout_rate, self.norm_layer))
            inner_channels += growth_rate * nblocks[index]

            #"""If a dense block contains m feature-maps, we let the
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels, self.norm_layer))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1], dropout_rate, self.norm_layer))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', self.norm_layer(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.linear = nn.Linear(inner_channels, num_class)

        # weights initialization
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output

    def get_features(self, x, return_feature='features'):
        output = self.conv1(x)
        features = self.features(output)
        avgpool = self.avgpool(features)
        view = avgpool.view(avgpool.size()[0], -1)
        out = self.linear(view)
        return_values = {
            'features': features, #1664*10*10
            'avgpool': avgpool, #1664*1*1
            'view': view, #1664
            'out': out #4
        }
        return return_values[return_feature]

    def _make_dense_layers(self, block, in_channels, nblocks, dropout_rate, norm_layer):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            if block == Bottleneck:
                dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate, dropout_rate, norm_layer))
            else:
                dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate, norm_layer))
            in_channels += self.growth_rate
        return dense_block

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def densenet3d121(**kwargs):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, **kwargs)

def densenet3d169(**kwargs):
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, **kwargs)

def densenet3d201(**kwargs):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32, **kwargs)

def densenet3d161(**kwargs):
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48, **kwargs)
