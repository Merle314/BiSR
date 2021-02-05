# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
from model import common
import numpy as np
import torch
import torch.nn as nn
import time

def make_model(args, parent=False):
    return RBiRDN(args)

class BiUpsampler(nn.Module):
    """Kernel Predict Networks for image upsampler, predict the point-wise convolution kernels from the position maps.
    """
    def __init__(self, in_channels=64, out_channels=3, kernel_size=3):
        """Calculate the default poseMaps according the input args
        """
        super(BiUpsampler, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.register_buffer('buffer', torch.zeros((1, 1, 1, 1), dtype=torch.float32))
        self.sigmar = nn.Parameter(torch.zeros(1, in_channels, 1, 1, 1))
        self.dwconv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, in_channels*kernel_size**2, kernel_size=1)
        )
        self.pwconv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, in_channels*out_channels, kernel_size=1)
        )

    def forward(self, x, poseMap, interMapY, interMapX):
        """Calcualte the [bs, out_channel*ks*ks, h, w] kernel, from the [bs, input_channel, h, w]
        """
        bs, _, hi, wi = x.shape
        _, _, ho, wo = poseMap.shape
        p = self.buffer.resize_(poseMap.shape).copy_(torch.from_numpy(poseMap)) # [1, 4, H*r, W*r]
        dw = self.dwconv(p.detach()) # [1, Cin*K*K, H*r, W*r]
        dw = dw.contiguous().view(1, self.in_channels, self.kernel_size**2, ho, wo) # [1, Cin, K*K, H*r, W*r]
        pw = self.pwconv(p.detach()) # [1, Cin*Cout, H*r, W*r]
        pw = pw.contiguous().view(self.in_channels, self.out_channels, ho, wo) # [Cin, Cout, H*r, W*r]

        h = nn.functional.unfold(x, self.kernel_size, padding=self.kernel_size//2) # [B, Cin*K*K, H*W]
        h = h.contiguous().view(bs, self.in_channels, self.kernel_size*self.kernel_size, hi, wi) # [B, Cin, K*K, H, W]
        h = h[:, :, :, interMapY, interMapX] # [B, Cin, K*K, H*r, W*r]
        # mean = x[:, :, interMapY, interMapX].unsqueeze(2) # [B, Cin, 1, H*r, W*r]
        mean = torch.sum(h.mul(dw.softmax(dim=2)), dim=2, keepdim=True) # [B, Cin, 1, H*r, W*r]
        bw = dw+self.sigmar*(h-mean)**2 # [B, Cin, K*K, H*r, W*r]

        # for _ in range(5):
        #     mean = torch.sum(h.mul(bw.softmax(dim=2)), dim=2, keepdim=True) # [B, Cin, 1, H*r, W*r]
        #     bw = dw+self.sigmar*(h-mean)**2 # [B, Cin, K*K, H*r, W*r]

        x = torch.einsum('abcde->abde', h.mul(bw.softmax(dim=2))) # [B, Cin, H*r, W*r]
        # x = torch.einsum('abcde->abde', h.mul(dw.softmax(dim=2))) # [B, Cin, H*r, W*r]
        out = torch.einsum("abde,bcde->acde", x, pw) # [B, Cin, H*r, W*r]
        return out

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU(inplace=True)
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RBiRDN(nn.Module):
    def __init__(self, args):
        super(RBiRDN, self).__init__()
        G0 = args.G0
        kSize = args.RDNkSize
        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Kernel Prediction Network based Up-sampling net
        self.BiUP = BiUpsampler(in_channels=G0, out_channels=args.n_colors, kernel_size=3)

    def forward(self, x, poseMap, interMapY, interMapX):
        x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        RDBs = torch.cat(RDBs_out, 1)
        x = self.GFF(RDBs)
        x += f__1

        out = self.BiUP(x, poseMap, interMapY, interMapX)
        out = self.add_mean(out)
        return out


    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('buffer') < 0:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

