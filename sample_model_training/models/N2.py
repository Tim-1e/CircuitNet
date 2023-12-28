import torch
import torch.nn as nn

from collections import OrderedDict
from torchvision.models.resnet import BasicBlock

def generation_init_weights(module):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                    or classname.find('Linear') != -1):
            
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    module.apply(init_func)

def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    return missing_keys

class conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        super(conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return input+self.main(input)

class upconv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1),
                nn.InstanceNorm2d(dim_out, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class ResidualConv(nn.Module):
    def __init__(self, dim_in, dim_out, stride=1):
        super(ResidualConv, self).__init__()
        # 假设使用 BasicBlock，您也可以选择其他类型的块
        self.downsample = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(dim_out)
        )
        self.block = BasicBlock(dim_in, dim_out, stride=stride, downsample=self.downsample)

    def forward(self, x):
        return self.block(x)

class UpConvWithSkip(nn.Module):
    def __init__(self, dim_in, dim_out, dim_skip):
        super(UpConvWithSkip, self).__init__()
        self.upconv = nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU(inplace=True)
        # 合并跳跃连接的特征
        self.conv1x1 = nn.Conv2d(dim_out + dim_skip, dim_out, kernel_size=1)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # 将跳跃连接的特征合并到上采样的特征中
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv1x1(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=32,dropout_rate=0.2):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.c1 = ResidualConv(in_dim, 32)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.c2 = ResidualConv(32, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.c3 = nn.Sequential(
                nn.Conv2d(64, out_dim, 3, 1, 1),
                nn.BatchNorm2d(out_dim),
                nn.Tanh()
                )

    def init_weights(self):
        generation_init_weights(self)
        

    def forward(self, input):
        h1 = self.c1(input)
        h2=self.dropout1(h1)
        h2 = self.pool1(h2)
        h3 = self.c2(h2)
        h4=self.dropout2(h3)
        h4 = self.pool2(h4)
        h5 = self.c3(h4)
        return h5, h2  # shortpath from 2->7


from torch import nn

class Decoder(nn.Module):
    def __init__(self, out_dim=2, in_dim=32, dropout_rate=0.2):
        super(Decoder, self).__init__()
        self.conv1 = conv(in_dim, 32)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.upc1 = upconv(32, 16)
        self.conv2 = conv(16, 16)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.upc2 = upconv(32+16, 4)
        self.conv3 = nn.Sequential(
            nn.Conv2d(4, out_dim, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        feature, skip = input
        d1 = self.conv1(feature)
        d1 = self.dropout1(d1)
        d2 = self.upc1(d1)
        d3 = self.conv2(d2)
        d3 = self.dropout2(d3)
        d4 = self.upc2(torch.cat([d3, skip], dim=1))
        output = self.conv3(d4)
        return output



class N2(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=2,
                 **kwargs):
        super().__init__()

        self.encoder = Encoder(in_dim=in_channels)
        self.decoder = Decoder(out_dim=out_channels)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def init_weights(self, pretrained=None, pretrained_transfer=None, strict=False, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
            print('Load state dict form {}'.format(pretrained))
        elif pretrained is None:
            generation_init_weights(self)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')
