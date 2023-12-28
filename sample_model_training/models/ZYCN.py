import torch
import torch.nn as nn

from collections import OrderedDict
from torchvision.models.resnet import BasicBlock,ResNet50_Weights
from torchvision.models import resnet50

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
        return self.main(input)

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

class Encoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=32):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.c1 = conv(in_dim, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.c2 = conv(32, 64)
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
        h2 = self.pool1(h1)
        h3 = self.c2(h2)
        h4 = self.pool2(h3)
        h5 = self.c3(h4)
        return h5, h2  # shortpath from 2->7


class Decoder(nn.Module):
    def __init__(self, out_dim=2, in_dim=32):
        super(Decoder, self).__init__()
        self.conv1 = conv(in_dim, 32)
        self.upc1 = upconv(32, 16)
        self.conv2 = conv(16, 16)
        self.upc2 = upconv(32+16, 4)
        self.conv3 =  nn.Sequential(
                nn.Conv2d(4, out_dim, 3, 1, 1),
                nn.Sigmoid()
                )

    def init_weights(self):
        generation_init_weights(self)

    def forward(self, input):
        feature, skip = input
        d1 = self.conv1(feature)
        d2 = self.upc1(d1)
        d3 = self.conv2(d2)
        d4 = self.upc2(torch.cat([d3, skip], dim=1))
        output = self.conv3(d4)  # shortpath from 2->7
        return output

class Res_Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Res_Encoder, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # 替换resnet的第一个卷积层以匹配输入通道数
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 使用 resnet 的特征，除去最后的全连接层和平均池化层
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        for i, layer in enumerate(self.features):
            x = layer(x)
        return x


class Res_Decoder(nn.Module):
    def __init__(self, in_channels=2048, out_channels=2):
        super(Res_Decoder, self).__init__()
        # 第一层上采样和卷积
        self.upconv1 = nn.ConvTranspose2d(in_channels, in_channels // 4, 4,4)
        self.conv1 = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        # 第二层上采样和卷积
        self.upconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 16, 4,4)
        self.conv2 = nn.Conv2d(in_channels // 16, in_channels // 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        # 第二层上采样和卷积
        self.upconv3 = nn.ConvTranspose2d(in_channels // 16, in_channels // 32, 4,2,1)
        self.conv3 = nn.Conv2d(in_channels // 32, in_channels // 32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        # 最终卷积层
        self.final_conv = nn.Conv2d(in_channels // 32, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.upconv1(x)
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.upconv2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = self.upconv3(x)
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.final_conv(x)
        return self.activation(x)

class ZYCN(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=2,
                 **kwargs):
        super().__init__()

        # Encoder
        self.encoder = Res_Encoder(in_channels=in_channels)

        # Decoder
        self.decoder = Res_Decoder(out_channels=out_channels)

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
