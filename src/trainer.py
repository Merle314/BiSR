import os
from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.utils as utils
from tqdm import tqdm

import utility
from model.common import pose_map

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, idx_scale,) in enumerate(self.loader_train):
            idx_scale = idx_scale[0]
            lr, hr = self.prepare(lr, hr)
            poseMap, interMapY, interMapX = pose_map(lr.shape[2:4], output_size=hr.shape[2:4])
            # poseMapL, poseMapH, interMapY, interMapX = pose_map(lr.shape[2:4], output_size=hr.shape[2:4])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, poseMap, interMapY, interMapX)
            # sr = self.model(lr, poseMapL, poseMapH, interMapY, interMapX)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        # torch.set_grad_enabled(False)
        with torch.no_grad():
            epoch = self.optimizer.get_last_epoch()
            self.ckp.write_log('\nEvaluation:')
            self.ckp.add_log(
                torch.zeros(1, 2, len(self.loader_test), len(self.args.test_scale))
            )
            self.model.eval()
            # self.model = self.model.to('cpu')

            timer_test = utility.timer()
            if self.args.save_results: self.ckp.begin_background()
            for idx_data, d in enumerate(self.loader_test):
                for idx_scale, scale in enumerate(self.args.test_scale):
                    d.dataset.set_scale(idx_scale)
                    for lr, hr, filename in tqdm(d, ncols=80):
                        lr = lr.to(self.device)
                        b, c, h, w = lr.size()
                        min_size = 360000
                        if((h*w*scale*scale)>min_size):
                            sr = self.forward_chop(lr, scale, min_size=min_size)
                        else:
                            poseMap, interMapY, interMapX = pose_map(lr.shape[2:4], output_size=hr.shape[2:4])
                            sr = self.model(lr, poseMap, interMapY, interMapX)
                        
                        # poseMapL, poseMapH, interMapY, interMapX = pose_map(lr.shape[2:4], output_size=hr.shape[2:4])
                        # sr = self.model(lr, poseMapL, poseMapH, interMapY, interMapX)
                        sr = sr.data.cpu()
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr]
                        self.ckp.log[-1, 0, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range, dataset=d
                        )
                        self.ckp.log[-1, 1, idx_data, idx_scale] += utility.calc_ssim(
                            sr, hr, scale, self.args.rgb_range, dataset=d
                        )
                        if self.args.save_gt:
                            save_list.extend([lr, hr])

                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, scale)
                        
                        del sr
                        del hr
                        del lr
                        del save_list
                        
                            # fname = self.ckp.get_path(
                            #     'results-{}'.format(d.dataset.name),
                            #     '{}_x{}_FL_'.format(filename[0], scale)
                            # )
                            # self.saveFeature(fl.data.cpu().numpy()[0], fname)
                            # fname = self.ckp.get_path(
                            #     'results-{}'.format(d.dataset.name),
                            #     '{}_x{}_FH_'.format(filename[0], scale)
                            # )
                            # self.saveFeature(fh.data.cpu().numpy()[0], fname)

                    self.ckp.log[-1, 0, idx_data, idx_scale] /= len(d)
                    self.ckp.log[-1, 1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} SSIM: {:.4f} (Best PSNR: {:.3f} @epoch {} Best SSIM: {:.4f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, 0, idx_data, idx_scale],
                            self.ckp.log[-1, 1, idx_data, idx_scale],
                            best[0][0, idx_data, idx_scale],
                            best[1][0, idx_data, idx_scale] + 1,
                            best[0][1, idx_data, idx_scale],
                            best[1][1, idx_data, idx_scale] + 1
                        )
                    )
            torch.cuda.empty_cache()
            self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
            self.ckp.write_log('Saving...')

            if self.args.save_results:
                self.ckp.end_background()

            if not self.args.test_only:
                # self.ckp.save(self, epoch, is_best=(best[1][0, 0, 0] + 1 == epoch))
                self.ckp.save(self, epoch, is_best=(torch.sum(best[1][0, :, :] + 1 == epoch)))

            self.ckp.write_log(
                'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
            )
        # torch.set_grad_enabled(True)


    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
    
    def saveFeature(self, feature, fname):
        num = feature.shape[0]
        width = feature.shape[1]
        height = feature.shape[2]
        for i in range(num):
            f = feature[i]
            f = (f-np.mean(f))/np.std(f)
            f = np.clip(f, -2.0, 2.0)
            plt.imshow(f)
            plt.axis('off')
            fig = plt.gcf()
            fig.set_size_inches(width/100.0,height/100.0) #输出width*height像素
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0)
            plt.margins(0,0)

            # cbar = plt.colorbar()
            # fig.savefig(filename+'Depth.png', dpi=500, bbox_inches='tight')
            # plt.margins(0,0)
            # fig.savefig(out_png_path, format='png', transparent=True, dpi=300, pad_inches = 0)
            plt.savefig(fname+str(i).zfill(2)+'.png')
            plt.close()

    def forward_chop(self, x, scale, shave=40, min_size=360000):
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        h_size = int(np.ceil(float(h_size)/10)*10)
        w_size = int(np.ceil(float(w_size)/10)*10)
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if (w_size*h_size*scale*scale) < min_size:
            poseMap, interMapY, interMapX = pose_map([h_size, w_size], scale_factor=scale)
            sr_list = []
            for i in range(4):
                lr_batch = lr_list[i]
                sr_list.append(self.model(lr_batch, poseMap, interMapY, interMapX))
        else:
            sr_list = [
                self.forward_chop(patch, scale, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = int(np.round(scale * h)), int(np.round(scale * w))
        h_half, w_half = int(np.round(scale * h_half)), int(np.round(scale * w_half))
        h_size, w_size = int(np.round(scale * h_size)), int(np.round(scale * w_size))
        shave = int(np.round(shave*scale))

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output