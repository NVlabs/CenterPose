# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from lib.models.data_parallel import DataParallel
from lib.utils.utils import AverageMeter
import cv2


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch, phase):
        pre_img = batch['pre_img'] if 'pre_img' in batch else None
        pre_hm = batch['pre_hm'] if 'pre_hm' in batch else None
        pre_hm_hp = batch['pre_hm_hp'] if 'pre_hm_hp' in batch else None

        outputs = self.model(batch['input'], pre_img, pre_hm, pre_hm_hp)

        loss, loss_stats, choice_list = self.loss(outputs, batch, phase)
        return outputs[-1], loss, loss_stats, choice_list


class BaseTrainer(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()

        writer_imgs = []  # Clear before each epoch # For tensorboard
        for iter_id, batch in enumerate(data_loader):

            # Skip the bad example
            if batch is None:
                continue

            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            output, loss, loss_stats, choice_list = model_with_loss(batch, phase)
            loss = loss.mean()  # No effect for our case
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()

                if isinstance(self.model_with_loss, torch.nn.DataParallel):
                    torch.nn.utils.clip_grad_norm_(self.model_with_loss.module.model.parameters(), 100.)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model_with_loss.model.parameters(), 100.)

                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                # Sometimes, some heads are not enabled
                if torch.is_tensor(loss_stats[l]) == True:
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            # Save everything for debug, including gt_hm/gt_hmhp/out_gt/out_pred/pred_hm/pred_hmhp/out_pred_gt_blend
            if phase == 'train':
                # Only save the first sample to save space

                # Debug only
                # if opt.debug > 0 :
                if opt.debug > 0 and iter_id == 0:
                    writer_imgs.append(self.debug(batch, output, iter_id, choice_list))

            elif opt.debug > 0:
                if opt.debug == 5:
                    # Todo: since validation dataset is not shuffled, we only care about 10+ images
                    if iter_id % (500 / opt.batch_size) == 0:
                        writer_imgs.append(self.debug(batch, output, iter_id, choice_list))
                else:
                    writer_imgs.append(self.debug(batch, output, iter_id, choice_list))

            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results, writer_imgs

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
