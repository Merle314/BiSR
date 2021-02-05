import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import MeanShift

def make_model(args, parent=False):
    return RBiCARN()

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
        # self.sigmar = nn.Parameter(torch.zeros(1, in_channels, 1, 1, 1))
        # self.sigmad = nn.Parameter(torch.ones(1, in_channels, 1, 1, 1))
        self.sigmaconv = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, 1, 1)
        )
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
        mean = torch.sum(h.mul(dw.softmax(dim=2)), dim=2, keepdim=False) # [B, Cin, 1, H*r, W*r]
        sigma = self.sigmaconv(mean).unsqueeze(2) # [B, 2, H*r, W*r]
        sigmad, sigmar = torch.split(sigma, 1, dim=1)
        bw = sigmad*dw+sigmar*torch.abs(h-mean.unsqueeze(2)) # [B, Cin, K*K, H*r, W*r]

        # for _ in range(5):
        #     mean = torch.sum(h.mul(bw.softmax(dim=2)), dim=2, keepdim=True) # [B, Cin, 1, H*r, W*r]
        #     bw = dw+self.sigmar*(h-mean)**2 # [B, Cin, K*K, H*r, W*r]

        x = torch.einsum('abcde->abde', h.mul(bw.softmax(dim=2))) # [B, Cin, H*r, W*r]
        # x = torch.einsum('abcde->abde', h.mul(dw.softmax(dim=2))) # [B, Cin, H*r, W*r]
        out = torch.einsum("abde,bcde->acde", x, pw) # [B, Cin, H*r, W*r]
        return out


class BilinearUpsampler(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super(BilinearUpsampler, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.register_buffer('buffer', torch.zeros((1, 1, 1, 1), dtype=torch.float32))
        self.wconv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, in_channels*kernel_size**2, kernel_size=1)
        )

    def forward(self, x, poseMap, interMapY, interMapX):
        """Calcualte the [bs, out_channel*ks*ks, h, w] kernel, from the [bs, input_channel, h, w]
        """
        bs, _, hi, wi = x.shape
        _, _, ho, wo = poseMap.shape
        p = self.buffer.resize_(poseMap.shape).copy_(torch.from_numpy(poseMap)) # [1, 4, H*r, W*r]
        w = self.wconv(p.detach()) # [1, Cin*Cout*K*K, H*r, W*r]
        w = w.contiguous().view( self.in_channels, self.kernel_size**2, ho, wo) # [Cin, K*K, H*r, W*r]

        h = nn.functional.unfold(x, self.kernel_size, padding=self.kernel_size//2) # [B, Cin*K*K, H*W]
        h = h.contiguous().view(bs, self.in_channels, self.kernel_size*self.kernel_size, hi, wi) # [B, Cin, K*K, H, W]
        h = h[:, :, :, interMapY, interMapX] # [B, Cin, K*K, H*r, W*r]

        x = torch.einsum('abcde, bcde->abde', h, w.softmax(dim=1)) # [B, Cin, H*r, W*r]
        return x


class Block(nn.Module):
    def __init__(self, num_fea):
        super(Block, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_fea, num_fea, 3, 1, 1)
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(num_fea * 2, num_fea, 1, 1, 0),
            nn.ReLU(True)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_fea, num_fea, 3, 1, 1)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(num_fea * 3, num_fea, 1, 1, 0),
            nn.ReLU(True)
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.ReLU(True), 
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(num_fea * 4, num_fea, 1, 1, 0),
            nn.ReLU(True)
        )

        self.act = nn.ReLU(True)

    def forward(self, x):
        b1 = self.act(self.b1(x) + x)
        c1 = torch.cat([x, b1], dim=1) # num_fea * 2
        o1 = self.c1(c1)

        b2 = self.act(self.b2(o1) + o1)
        c2 = torch.cat([c1, b2], dim=1) # num_fea * 3
        o2 = self.c2(c2)

        b3 = self.act(self.b3(o2) + o2)
        c3 = torch.cat([c2, b3], dim=1) # num_fea * 4
        o3 = self.c3(c3)

        return o3

class RBiCARN(nn.Module):
    def __init__(self, in_channels=3, num_fea=64, out_channels=3, use_skip=True):
        super(RBiCARN, self).__init__()
        self.use_skip = use_skip
        rgb_range = 255
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        # extract features
        self.res_in = nn.Conv2d(in_channels, num_fea, 3, 1, 1)
        self.fea_in = nn.Conv2d(num_fea, num_fea, 3, 1, 1)

        # CARN body
        self.b1 = Block(num_fea)
        self.c1 = nn.Sequential(
            nn.Conv2d(num_fea * 2, num_fea, 1, 1, 0),
            nn.ReLU(True)
        )

        self.b2 = Block(num_fea)
        self.c2 = nn.Sequential(
            nn.Conv2d(num_fea * 3, num_fea, 1, 1, 0),
            nn.ReLU(True)
        )

        self.b3 = Block(num_fea)
        self.c3 = nn.Sequential(
            nn.Conv2d(num_fea * 4, num_fea, 1, 1, 0),
            nn.ReLU(True)
        )

        # Reconstruct
        self.upsampler = BiUpsampler(in_channels=num_fea, out_channels=out_channels, kernel_size=3)
        self.biupsampler = BilinearUpsampler(in_channels, kernel_size=5)

    def forward(self, x, poseMap, interMapY, interMapX):
        x = self.sub_mean(x)

        # feature extraction
        inter_res = self.biupsampler(x, poseMap, interMapY, interMapX)    
        res = self.res_in(x)
        x = self.fea_in(res)

        # body
        b1 = self.b1(x)
        c1 = torch.cat([b1, x], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        # Reconstruct
        out = self.upsampler(o3+res, poseMap, interMapY, interMapX)+inter_res

        out = self.add_mean(out)

        return out
