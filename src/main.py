import os

import torch

import data
import loss
import model
import utility
from option import args
from trainer import Trainer

os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()

if __name__ == '__main__':
    main()
