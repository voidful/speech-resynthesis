# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset import F0Dataset, get_dataset_filelist
from models import Quantizer
from utils import scan_checkpoint, load_checkpoint, save_checkpoint, build_env, \
    AttrDict

torch.backends.cudnn.benchmark = True


def train(a, h):
    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = Quantizer(h).to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        generator = torch.nn.DataParallel(generator)

    print(generator)
    os.makedirs(a.checkpoint_path, exist_ok=True)
    print("checkpoints directory : ", a.checkpoint_path)

    cp_g = None
    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')

    steps = 0
    if cp_g is None:
        last_epoch = -1
        state_dict_g = None
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        generator.load_state_dict(state_dict_g['generator'])
        steps = state_dict_g['steps'] + 1
        last_epoch = state_dict_g['epoch']

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_g is not None:
        optim_g.load_state_dict(state_dict_g['optim_g'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h)

    trainset = F0Dataset(training_filelist, h.segment_size, h.sampling_rate, n_cache_reuse=0, device=device,
                         multispkr=h.get('multispkr', None), f0_stats=h.get('f0_stats', None),
                         f0_normalize=h.get('f0_normalize', False), f0_feats=h.get('f0_feats', False),
                         f0_median=h.get('f0_median', False), f0_interp=h.get('f0_interp', False),
                         vqvae=h.get('code_vq_params', False))

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=True,
                              batch_size=h.batch_size, pin_memory=True, drop_last=True)

    validset = F0Dataset(validation_filelist, h.segment_size, h.sampling_rate, False, n_cache_reuse=0,
                         device=device, multispkr=h.get('multispkr', None), f0_stats=h.get('f0_stats', None),
                         f0_normalize=h.get('f0_normalize', False), f0_feats=h.get('f0_feats', False),
                         f0_median=h.get('f0_median', False), f0_interp=h.get('f0_interp', False),
                         vqvae=h.get('code_vq_params', False))
    validation_loader = DataLoader(validset, num_workers=h.num_workers, shuffle=False, sampler=None,
                                   batch_size=h.batch_size, pin_memory=True, drop_last=True)

    sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        start = time.time()
        print("Epoch: {}".format(epoch + 1))

        for i, batch in enumerate(train_loader):
            start_b = time.time()
            x, y, _ = batch
            y = torch.autograd.Variable(y.to(device, non_blocking=False))
            x = {k: torch.autograd.Variable(v.to(device, non_blocking=False)) for k, v in x.items()}

            y_g_hat, commit_loss, metrics = generator(**x)
            f0_commit_loss = commit_loss[0]
            f0_metrics = metrics[0]

            # Generator
            optim_g.zero_grad()

            # L2 Reconstruction Loss
            loss_recon = F.mse_loss(y_g_hat, y)

            # Replicate f0_commit_loss across the batch dimension (if necessary)
            #if f0_commit_loss.dim() == 0:
            #    f0_commit_loss = f0_commit_loss * torch.ones_like(loss_recon)

            loss_recon += f0_commit_loss.mean() * h.get('lambda_commit', None)

            loss_recon.backward()
            optim_g.step()

            # STDOUT logging
            if steps % a.stdout_interval == 0:
                print('Steps : {:d}, Gen Loss Total : {:4.3f}, s/b : {:4.3f}'.format(steps, loss_recon,
                                                                                     time.time() - start_b))

            # checkpointing
            if steps % a.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                                {'generator': generator.module.state_dict() if isinstance(generator, torch.nn.DataParallel) else generator.state_dict(),
                                 'optim_g': optim_g.state_dict(), 'steps': steps, 'epoch': epoch})

            # Tensorboard summary logging
            if steps % a.summary_interval == 0:
                sw.add_scalar("training/gen_loss_total", loss_recon, steps)
                sw.add_scalar("training/commit_error", f0_commit_loss.mean(), steps)

            # Validation
            if steps % a.validation_interval == 0:  # and steps != 0:
                generator.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0
                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):
                        x, y, _ = batch
                        x = {k: v.to(device, non_blocking=False) for k, v in x.items()}
                        y = torch.autograd.Variable(y.to(device, non_blocking=False))

                        y_g_hat, commit_loss, _ = generator(**x)
                        f0_commit_loss = commit_loss[0]

                        # Replicate f0_commit_loss across the batch dimension (if necessary)
                        if f0_commit_loss.dim() == 0:
                            f0_commit_loss = f0_commit_loss * torch.ones_like(y)

                        val_err_tot += f0_commit_loss * h.get('lambda_commit', None)
                        val_err_tot += F.mse_loss(y_g_hat, y).item()

                    val_err = val_err_tot / (j + 1)
                    sw.add_scalar("validation/mel_spec_error", val_err.mean(), steps)
                    sw.add_scalar("validation/commit_error", f0_commit_loss.mean(), steps)

                generator.train()

            steps += 1
            if steps >= a.training_steps:
                break

        scheduler_g.step()

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

    print('Finished training')


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=10000, type=int)
    parser.add_argument('--training_steps', default=400000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=10000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)

    train(a, h)


if __name__ == '__main__':
    main()