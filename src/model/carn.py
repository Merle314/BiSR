import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import MeanShift

def make_model(args, parent=False):
    return CARN()


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

class CARN(nn.Module):
    def __init__(self, in_channels=3, num_fea=64, out_channels=3, use_skip=True):
        super(CARN, self).__init__()
        self.use_skip = use_skip
        r = 4
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
        # Up-sampling net
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(num_fea, num_fea * r * r, 3, 1, 1),
                nn.PixelShuffle(r),
                nn.Conv2d(num_fea, out_channels, 3, 1, 1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(num_fea, num_fea * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_fea, num_fea * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_fea, out_channels, 3, 1, 1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x, poseMap, interMapY, interMapX):
        x = self.sub_mean(x)

        # feature extraction
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
        out = self.UPNet(o3+res)

        out = self.add_mean(out)

        return out
