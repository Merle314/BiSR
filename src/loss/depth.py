from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Depth(nn.Module):
    def __init__(self, args):
        super(Depth, self).__init__()
        self.n_resblocks = args.n_resblocks

    def forward(self, sr, hr, depth):
        margin = torch.abs(hr-sr.detach())
        # print(torch.min(margin).cpu().numpy(), ';', torch.mean(margin).cpu().numpy(), ';', torch.max(margin).cpu().numpy())
        loss = torch.mean(torch.exp(-torch.abs(hr-sr.detach()))*depth/self.n_resblocks)

        return loss
