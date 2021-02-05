import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range=255,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

def mean_channels(x):
    assert(x.dim() == 4)
    spatial_sum = x.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (x.shape[2] * x.shape[3])

def std(x):
    assert(x.dim() == 4)
    x_mean = mean_channels(x)
    x_var = (x - x_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (x.shape[2] * x.shape[3])
    return x_var.pow(0.5)

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class MetaUpsampler(nn.Module):
    """deepwise Kernel Predict Networks, predict the point-wise convolution kernels from the position maps.
    """
    def __init__(self, input_size, scale_factor, in_channels=64, out_channels=3, kernel_size=3, n_GPUs=1):
        """Calculate the default poseMaps according the input args
        """
        super(MetaUpsampler, self).__init__()
        self.n_GPUs = n_GPUs
        self.input_size = input_size
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.device = torch.device('cpu' if self.n_GPUs==0 else 'cuda')
        self.poseMap = torch.from_numpy(self.pose_map(input_size, scale_factor=scale_factor)).to(self.device)
        self.poseMaps = self.poseMap.unsqueeze(0)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, in_channels*kernel_size*kernel_size*out_channels, kernel_size=1)
        )
    
    def pose_map(self, input_size, output_size=None, scale_factor=None, mode='nearest', align_corners=False):
        if output_size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if output_size is not None and scale_factor is not None:
            raise ValueError('only one of size or scale_factor should be defined')
        src_h, src_w = input_size
        if output_size is not None:
            dst_h, dst_w = output_size
            if align_corners:
                scale_x, scale_y = float(src_w-1)/(dst_w-1), float(src_h-1)/(dst_h-1)
            else:
                scale_x, scale_y = float(src_w)/dst_w, float(src_h)/dst_h
        if scale_factor is not None:
            dst_h, dst_w = int(np.floor(src_h*scale_factor)), int(np.floor(src_w*scale_factor))
            if align_corners:
                scale_x, scale_y = float(src_w-1)/(dst_w-1), float(src_h-1)/(dst_h-1)
            else:
                scale_x, scale_y = float(src_w)/dst_w, float(src_h)/dst_h
        poseMap = np.zeros((3, dst_h, dst_w), dtype=np.float32)
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # find the origin x and y coordinates of dst image x and y
                if align_corners:
                    src_x = dst_x*scale_x
                    src_y = dst_y*scale_y
                    ner_x = np.min([np.floor(dst_x*scale_x), dst_w-1])
                    ner_y = np.min([np.floor(dst_y*scale_y), dst_h-1])
                else:
                    # src_x = (dst_x + 0.5) * scale_x - 0.5
                    # src_y = (dst_y + 0.5) * scale_y - 0.5
                    src_x = dst_x * scale_x
                    src_y = dst_y * scale_y
                    ner_x = np.min([np.floor(dst_x*scale_x), dst_w-1])
                    ner_y = np.min([np.floor(dst_y*scale_y), dst_h-1])

                # find the coordinates of the points which will be used to compute the interpolation
                poseMap[0, dst_y, dst_x] = src_x-ner_x
                poseMap[1, dst_y, dst_x] = src_y-ner_y
                poseMap[2, dst_y, dst_x] = scale_x
        return poseMap
    
    def forward(self, x):
        """Calcualte the [bs, out_channels*ks*ks, h, w] kernel, from the [bs, input_channel, h, w]
        """
        bs, cs, h, w = x.shape
        if self.input_size==(h, w):
            poseMaps = self.poseMaps
        else:
            poseMap = torch.from_numpy(self.pose_map((h, w), scale_factor=self.scale_factor)).to(self.device)
            poseMaps = poseMap.unsqueeze(0)
        weight = self.conv(poseMaps) # [1, Cin*K*K*Cout, W*r, H*r]
        weight = weight.squeeze(0) # [Cin*K*K*Cout, W*r, H*r]
        # weight = weight.repeat(bs, 1, 1, 1)

        _, ho, wo = weight.shape
        f = F.unfold(x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2) # [B, Cin*K*K, H*W]
        f = F.fold(f, output_size=(h, w), kernel_size=1) # [B, Cin*K*K, H, W]
        f = F.interpolate(f, scale_factor=self.scale_factor) # [B, Cin*K*K, H*r, W*r]
        # f = f.permute(0, 2, 3, 1).unsqueeze(3) # [B, H*r, W*r, 1, Cin*K*K]
        f = f.permute(2, 3, 0, 1) # [H*r, W*r, B, Cin*K*K]
        weight = weight.contiguous().view(self.kernel_size*self.kernel_size*self.in_channels, self.out_channels, ho, wo).permute(2, 3, 0, 1) # [H*r, W*r, Cin*K*K, Cout]
        out = torch.matmul(f, weight).permute(2, 3, 0, 1) # [B, Cout, H*r, W*r]

        return out


class contentKPN(nn.Module):
    """deepwise Kernel Predict Networks, predict the point-wise convolution kernels from the input batchs of content feature maps.
    """
    def __init__(self, in_channels=64, out_channels=3, kernel_size=1):
        """Definition the weight predict convolution layer
        """
        super(contentKPN, self).__init__()
        self.ks = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        ks = self.ks
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*ks*ks, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels*ks*ks, in_channels*ks*ks, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels*ks*ks, in_channels*ks*ks*out_channels, kernel_size=1),
        )
    
    def forward(self, x):
        """Calcualte the [bs, out_channel*ks*ks, h, w] kernel, from the [bs, input_channel, h, w]
        """
        ks = self.ks
        bs, cs, h, w = x.shape
        weight = self.conv(x) # [B, Cin*K*K*Cout, H, W]
        if ks != 1:
            x = nn.functional.unfold(x, kernel_size=self.ks, padding=self.ks//2) # [B, Cin*K*K, H*W]
            x = nn.functional.fold(x, output_size=(h, w), kernel_size=1) # [B, Cin*K*K, H, W]

        x = x.contiguous().permute(0, 2, 3, 1).unsqueeze(3) # [B, H*r, W*r, 1, Cin*K*K]
        weight = weight.contiguous().view(bs, self.in_channels*ks*ks, self.out_channels, h, w).permute(0, 3, 4, 1, 2) # [B, H*r, W*r, Cin*K*K, Cout]
        out = torch.matmul(x, weight).squeeze(3).permute(0, 3, 1, 2) # [B, Cout, H*r, W*r]
        return out


class positionKPN(nn.Module):
    """deepwise Kernel Predict Networks, predict the point-wise convolution kernels from the position maps.
    """
    def __init__(self, input_size, scale_list, in_channels=64, kernel_size=3):
        """Calculate the default poseMaps according the input args
        """
        super(positionKPN, self).__init__()

        self.input_size = input_size
        self.scale_list = scale_list
        self.kernel_size = kernel_size
        self.interMapX = []
        self.interMapY = []
        for idx, scale in enumerate(self.scale_list):
            poseMap, interMapY, interMapX = self.pose_map(self.input_size, scale_factor=scale)
            self.register_buffer('poseMap_{:d}'.format(idx), torch.from_numpy(poseMap))
            self.interMapY.append(interMapY)
            self.interMapX.append(interMapX)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, in_channels*kernel_size*kernel_size, kernel_size=1),
        )
    
    def pose_map(self, input_size, output_size=None, scale_factor=None, mode='bilinear', align_corners=False):
        if output_size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if output_size is not None and scale_factor is not None:
            raise ValueError('only one of size or scale_factor should be defined')
        src_h, src_w = input_size
        if output_size is not None:
            dst_h, dst_w = output_size
            scale_x, scale_y = float(src_w)/dst_w, float(src_h)/dst_h
        if scale_factor is not None:
            dst_h, dst_w = int(np.floor(src_h*scale_factor)), int(np.floor(src_w*scale_factor))
            scale_x, scale_y = float(src_w)/dst_w, float(src_h)/dst_h
        print('Generate poseMap for input size:{}x{}, scale:{:.2f}'.format(src_h, src_w, scale_x))
        poseMap = np.empty((1, 2, dst_h, dst_w), dtype=np.float32)
        interMapX = np.empty((dst_h, dst_w), dtype=np.int16)
        interMapY = np.empty((dst_h, dst_w), dtype=np.int16)
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # find the origin x and y coordinates of dst image x and y
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                # calculate the bilinear four point
                src_x_0 = np.floor(src_x)
                src_y_0 = np.floor(src_y)

                # find the interpolate Map
                interMapX[dst_y, dst_x] = src_x_0 + 1
                interMapY[dst_y, dst_x] = src_y_0 + 1

                # find the coordinates of the points which will be used to compute the interpolation
                poseMap[0, 0, dst_y, dst_x] = src_x-src_x_0
                poseMap[0, 1, dst_y, dst_x] = src_y-src_y_0
        return poseMap, interMapY, interMapX

    def forward(self, x, scale):
        """Calcualte the [bs, out_channel*ks*ks, h, w] kernel, from the [bs, input_channel, h, w]
        """
        bs, cs, h, w = x.shape
        if self.input_size==(h, w) and (scale in self.scale_list):
            scale_idx = self.scale_list.index(scale)
            poseMap = eval('self.poseMap_{:d}'.format(scale_idx))
            interMapY = self.interMapY[scale_idx]
            interMapX = self.interMapX[scale_idx]
        else:
            device = self.poseMap_0.device
            poseMap, interMapY, interMapX = self.pose_map((h, w), scale_factor=scale)
            poseMap = torch.from_numpy(poseMap).to(device)
        weight = self.conv(poseMap) # [1, C*K*K, W*r, H*r]

        _, _, ho, wo = weight.shape
        f = nn.functional.unfold(x, kernel_size=self.kernel_size, padding=self.kernel_size//2) # [B, C*K*K, (H+K//2)*(W+K//2)]
        f = nn.functional.fold(f, output_size=(h+self.kernel_size//2, w+self.kernel_size//2), kernel_size=1) # [B, C*K*K, H+K//2, W+K//2]
        f = f[:, :, interMapY, interMapX] # [B, C*K*K, H*r, W*r]
        # f = nn.functional.interpolate(f, (ho, wo), mode='nearest') # [B, C*K*K, H*r, W*r]
        # f = f.view(bs, cs, self.kernel_size*self.kernel_size, ho, wo).permute(0, 1, 3, 4, 2).unsqueeze(4) # [B, C, H*r, W*r, 1, K*K]
        f = f.contiguous().view(bs, cs, self.kernel_size*self.kernel_size, ho, wo).permute(1, 3, 4, 0, 2) # [C, H*r, W*r, B, K*K]
        weight = weight.contiguous().view(cs, self.kernel_size*self.kernel_size, ho, wo, 1).permute(0, 2, 3, 1, 4) # [C, H*r, W*r, K*K, 1]
        out = torch.matmul(f, weight).squeeze(4).permute(3, 0, 1, 2) # [B, C, H*r, W*r]
        return out


def pose_map(input_size, output_size=None, scale_factor=None, mode='bilinear', align_corners=False):
    if output_size is None and scale_factor is None:
        raise ValueError('either size or scale_factor should be defined')
    if output_size is not None and scale_factor is not None:
        raise ValueError('only one of size or scale_factor should be defined')
    src_h, src_w = input_size
    if output_size is not None:
        dst_h, dst_w = output_size
        scale_x, scale_y = float(src_w)/dst_w, float(src_h)/dst_h
    if scale_factor is not None:
        dst_h, dst_w = int(np.round(src_h*scale_factor)), int(np.round(src_w*scale_factor))
        scale_x, scale_y = float(src_w)/dst_w, float(src_h)/dst_h
    poseMap = np.empty((1, 3, dst_h, dst_w), dtype=np.float32)
    interMapX = np.empty((1, 1, 1, dst_h, dst_w), dtype=np.int16)
    interMapY = np.empty((1, 1, 1, dst_h, dst_w), dtype=np.int16)
    dst_xy = np.empty((2, dst_h, dst_w))
    dst_xy[0, :, :] = np.linspace(0, dst_w-1, dst_w).reshape(1, dst_w)
    dst_xy[1, :, :] = np.linspace(0, dst_h-1, dst_h).reshape(dst_h, 1)
    scale_xy = np.array([scale_x, scale_y]).reshape(2, 1, 1)
    src_xy = (dst_xy+0.5)*scale_xy-0.5
    src_xy_0 = np.floor(src_xy+0.5)
    interMapX = src_xy_0[0, :, :]
    interMapY = src_xy_0[1, :, :]
    poseMap[0, 0:2, :, :] = src_xy-src_xy_0
    poseMap[0, 2, :, :] = scale_x
    return poseMap, interMapY, interMapX

# def pose_map(input_size, output_size=None, scale_factor=None, mode='bicubic', align_corners=False):
#     if output_size is None and scale_factor is None:
#         raise ValueError('either size or scale_factor should be defined')
#     if output_size is not None and scale_factor is not None:
#         raise ValueError('only one of size or scale_factor should be defined')
#     src_h, src_w = input_size
#     if output_size is not None:
#         dst_h, dst_w = output_size
#         scale_x, scale_y = float(src_w)/dst_w, float(src_h)/dst_h
#     if scale_factor is not None:
#         dst_h, dst_w = int(np.round(src_h*scale_factor)), int(np.round(src_w*scale_factor))
#         scale_x, scale_y = float(src_w)/dst_w, float(src_h)/dst_h
#     poseMap = np.empty((1, 3, dst_h, dst_w), dtype=np.float32)
#     interMapX = np.empty((dst_h, dst_w), dtype=np.int16)
#     interMapY = np.empty((dst_h, dst_w), dtype=np.int16)
#     dst_xy = np.empty((2, dst_h, dst_w))
#     dst_xy[0, :, :] = np.linspace(0, dst_w-1, dst_w).reshape(1, dst_w)
#     dst_xy[1, :, :] = np.linspace(0, dst_h-1, dst_h).reshape(dst_h, 1)
#     scale_xy = np.array([scale_x, scale_y]).reshape(2, 1, 1)
#     src_xy = (dst_xy+0.5)*scale_xy-0.5
#     src_xy_0 = np.floor(src_xy)
#     interMapX[:, :] = src_xy_0[0, :, :]+1
#     interMapY[:, :] = src_xy_0[1, :, :]+1
#     poseMap[0, 0:2, :, :] = src_xy-src_xy_0
#     poseMap[0, 2, :, :] = scale_x
#     return poseMap, interMapY, interMapX


# def pose_map(input_size, output_size=None, scale_factor=None, mode='bilinear', align_corners=False):
#     if output_size is None and scale_factor is None:
#         raise ValueError('either size or scale_factor should be defined')
#     if output_size is not None and scale_factor is not None:
#         raise ValueError('only one of size or scale_factor should be defined')
#     src_h, src_w = input_size
#     if output_size is not None:
#         dst_h, dst_w = output_size
#         scale_x, scale_y = float(src_w)/dst_w, float(src_h)/dst_h
#     if scale_factor is not None:
#         dst_h, dst_w = int(np.round(src_h*scale_factor)), int(np.round(src_w*scale_factor))
#         scale_x, scale_y = float(src_w)/dst_w, float(src_h)/dst_h
#     poseMap = np.empty((1, 4, dst_h, dst_w), dtype=np.float32)
#     interMapX = np.empty((dst_h, dst_w), dtype=np.int16)
#     interMapY = np.empty((dst_h, dst_w), dtype=np.int16)
#     dst_xy = np.empty((2, dst_h, dst_w))
#     dst_xy[0, :, :] = np.linspace(0, dst_w-1, dst_w).reshape(1, dst_w)
#     dst_xy[1, :, :] = np.linspace(0, dst_h-1, dst_h).reshape(dst_h, 1)
#     scale_xy = np.array([scale_x, scale_y]).reshape(2, 1, 1)
#     src_xy = (dst_xy+0.5)*scale_xy-0.5
#     src_xy_0 = np.floor(src_xy+0.5)
#     interMapX[:, :] = src_xy_0[0, :, :]
#     interMapY[:, :] = src_xy_0[1, :, :]
#     poseMap[0, 0:2, :, :] = src_xy-src_xy_0

#     poseMapL = np.empty((1, 2, src_h, src_w), dtype=np.float32)
#     src_xy = np.empty((2, src_h, src_w))
#     src_xy[0, :, :] = np.linspace(0, src_w-1, src_w).reshape(1, src_w)
#     src_xy[1, :, :] = np.linspace(0, src_h-1, src_h).reshape(src_h, 1)
#     dst_xy = (src_xy+0.5)/scale_xy-0.5
#     dst_xy_0 = np.floor(dst_xy+0.5)
#     poseMapL[0, :, :, :] = dst_xy-dst_xy_0
#     poseMap[0, 2, :, :] = poseMapL[0, 0, interMapY, interMapX]
#     poseMap[0, 3, :, :] = poseMapL[0, 1, interMapY, interMapX]
#     return poseMap, interMapY, interMapX