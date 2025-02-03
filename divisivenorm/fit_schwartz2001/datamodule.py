###############################################################################
# Copyright 2024 Ferenc Csikor
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
###############################################################################
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import from_numpy
import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

from config import config


class Schwartz2001DataModule(pl.LightningDataModule):
    def __init__(self, model_name):
        super().__init__()

        self.model_name = model_name
        self.train_ratio = config['train_ratio']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']

    def prepare_data(self):
        print(f'\n{self.model_name} data module:')
        if ('StandardVAE' in self.model_name
                and 'standardvae_ims_fn' in config):
            self.test_ims = Schwartz2001DataModule.load_test_images(
                f'{self.model_name}{config["standardvae_ims_fn"]}')
        elif ('StandardVAE' not in self.model_name
                and 'eavae_ims_fn' in config):
            self.test_ims = Schwartz2001DataModule.load_test_images(
                f'{self.model_name}{config["eavae_ims_fn"]}')
        else:
            self.test_ims = Schwartz2001DataModule.load_test_images(
                f'{self.model_name}_input_ims.pkl')
        if ('StandardVAE' in self.model_name
                and 'standardvae_recfs_fn' in config):
            self.recfs = Schwartz2001DataModule.load_recfs(
                f'{self.model_name}{config["standardvae_recfs_fn"]}',
                self.model_name,
                config[f'recf_{self.model_name}_sample_fn'])
        elif ('StandardVAE' not in self.model_name
                and 'eavae_recfs_fn' in config):
            self.recfs = Schwartz2001DataModule.load_recfs(
                f'{self.model_name}{config["eavae_recfs_fn"]}',
                self.model_name,
                config[f'recf_{self.model_name}_sample_fn'])
        else:
            self.recfs = Schwartz2001DataModule.load_recfs(
                f'{self.model_name}_receptive.pkl',
                self.model_name,
                config[f'recf_{self.model_name}_sample_fn'])
        if ('StandardVAE' in self.model_name
                and 'standardvae_postmns_fn' in config):
            self.postmns = Schwartz2001DataModule.load_postmns(
                f'{self.model_name}{config["standardvae_postmns_fn"]}',
                self.model_name)
        elif ('StandardVAE' not in self.model_name
                and 'eavae_postmns_fn' in config):
            self.postmns = Schwartz2001DataModule.load_postmns(
                f'{self.model_name}{config["eavae_postmns_fn"]}',
                self.model_name)
        else:
            self.postmns = Schwartz2001DataModule.load_postmns(
                f'{self.model_name}_posteriors_cpu.pkl', self.model_name)

        # harmonize data following the instructions from Domo
        if self.test_ims.shape[0] != self.postmns.shape[0]:
            print('the number of images and posteriors differ; '
                  'using the smaller number everywhere')
            n_ims = min(self.test_ims.shape[0], self.postmns.shape[0])
            self.test_ims = self.test_ims[:n_ims, :]
            self.postmns = self.postmns[:n_ims, :]
        if self.recfs.shape[0] != self.postmns.shape[1]:
            print('the number of z dims in receptive fields and posteriors '
                  'differ; using the smaller number everywhere')
            n_zdims = min(self.recfs.shape[0], self.postmns.shape[1])
            self.recfs = self.recfs[:n_zdims, :]
            self.postmns = self.postmns[:, :n_zdims]

        self.active_recfs, self.active_postmns, \
            self.inactive_recfs, self.inactive_postmns = \
            Schwartz2001DataModule.active_recfs_postmns(
                self.recfs, self.postmns, config['active_recfs_threshold'],
                config[f'recf_{self.model_name}_active_fn_pre'],
                self.model_name)

        abscorr_activepostmns = np.triu(np.abs(
            np.corrcoef(self.active_postmns.T)), 1)
        print('mean abs. correlation coefficient between posterior means: '
              f'{np.mean(abscorr_activepostmns):.5g}')

        self.linear_responses = np.matmul(self.test_ims, self.active_recfs.T)
        print(f'calculated linear responses for {self.model_name}')
        print(f'  shape: {self.linear_responses.shape}; '
              f'dtype: {self.linear_responses.dtype}')
        print(f'  mean: {np.mean(self.linear_responses):.5f}; '
              f'std: {np.std(self.linear_responses):.5f}')

        self.linear_responses /= np.std(self.linear_responses)
        print(f'normalized linear responses for {self.model_name}')
        print(f'  shape: {self.linear_responses.shape}; '
              f'dtype: {self.linear_responses.dtype}')
        print(f'  mean: {np.mean(self.linear_responses):.5f}; '
              f'std: {np.std(self.linear_responses):.5f}')

    def setup(self, stage):
        if stage == 'fit':
            tds = TensorDataset(from_numpy(self.linear_responses),
                                from_numpy(self.active_postmns))
            self.ds_train, self.ds_val = \
                random_split(tds, (self.train_ratio, 1.0 - self.train_ratio))

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size,
                          shuffle=True, drop_last=True,
                          num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size,
                          shuffle=False, drop_last=True,
                          num_workers=self.num_workers,
                          persistent_workers=True)

    @staticmethod
    def save_samples_png(samps, fn=None, vminmax=1.0):
        # Saves samps to fn
        n_rows = n_cols = math.ceil(math.sqrt(samps.shape[0]))

        if samps.ndim == 2:
            xdim = int(math.sqrt(samps.shape[1]))
            samps = samps.reshape((samps.shape[0], xdim, xdim))

        fig, axs = plt.subplots(n_rows, n_cols, constrained_layout=True,
                                figsize=(10, 10))
        for i_row in range(n_rows):
            for i_col in range(n_cols):
                i_lin = i_row * n_cols + i_col
                ax = axs[i_row, i_col]
                ax.set_xticks([])
                ax.set_yticks([])
                if i_lin < samps.shape[0]:
                    img = samps[i_lin, :, :]
                    ax.imshow(img, cmap='gray', vmin=-vminmax, vmax=vminmax)

        if fn:
            fig.savefig(fn)

    @staticmethod
    def load_test_images(ims_fn):
        # Load test images (natural, whitened)
        with open(config['smvae_data_dir'] + ims_fn, 'rb') as f:
            if ims_fn.endswith('.pt'):
                test_ims = torch.load(f, map_location='cpu').detach().numpy()
            else:
                test_ims = pickle.load(f).detach().numpy()

        if test_ims.ndim >= 3:
            test_ims = test_ims.reshape((test_ims.shape[0], -1))

        print('loaded test images')
        print(f'  shape: {test_ims.shape}; dtype: {test_ims.dtype}')
        print(f'  pixel mean: {np.mean(test_ims):.5f}; '
              f'std: {np.std(test_ims):.5f}')

        n_samps_sqrt = 5
        Schwartz2001DataModule.save_samples_png(
            test_ims[:n_samps_sqrt * n_samps_sqrt, :],
            config['image_sample_fn'], config['vminmax_imgs'])
        print(f'  saved {n_samps_sqrt}x{n_samps_sqrt} image samples '
              f'to {config["image_sample_fn"]}')

        return test_ims

    @staticmethod
    def load_recfs(recf_fn, model_name, sample_fn):
        # Load receptive fields
        with open(config['smvae_data_dir'] + recf_fn, 'rb') as f:
            if recf_fn.endswith('.pt'):
                recfs = torch.load(f, map_location='cpu')
            else:
                recfs = pickle.load(f)

        if recfs.ndim == 3:
            recfs = recfs.reshape((recfs.shape[0],
                                   recfs.shape[1] * recfs.shape[2]))

        print(f'loaded recfields for {model_name}')
        print(f'  shape: {recfs.shape}; dtype: {recfs.dtype}')
        print(f'  mean: {np.mean(recfs):.5f}; '
              f'std: {np.std(recfs):.5f}')

        n_samps_sqrt = 5
        Schwartz2001DataModule.save_samples_png(
            recfs[:n_samps_sqrt * n_samps_sqrt, :],
            sample_fn, config['vminmax_recfs'])
        print(f'  saved {n_samps_sqrt}x{n_samps_sqrt} image samples '
              f'to {sample_fn}')

        return recfs

    @staticmethod
    def load_postmns(postmn_fn, model_name):
        # Load posterior means
        with open(config['smvae_data_dir'] + postmn_fn, 'rb') as f:
            if postmn_fn.endswith('.pt'):
                postmns = torch.load(f, map_location='cpu').detach().numpy()
            else:
                postmns = pickle.load(f).detach().numpy()

        print(f'loaded postmns for {model_name}')
        print(f'  shape: {postmns.shape}; dtype: {postmns.dtype}')
        print(f'  mean: {np.mean(postmns):.5f}; '
              f'std: {np.std(postmns):.5f}')

        return postmns

    @staticmethod
    def active_recfs_postmns(recfs, postmns, threshold=1.0, fn_pre=None,
                             model_name=None):
        active_recfs = recfs[np.std(recfs, axis=1) >= threshold, :]
        active_postmns = postmns[:, np.std(recfs, axis=1) >= threshold]

        inactive_recfs = recfs[np.std(recfs, axis=1) < threshold, :]
        inactive_postmns = postmns[:, np.std(recfs, axis=1) < threshold]

        if fn_pre:
            Schwartz2001DataModule.save_samples_png(
                active_recfs, fn_pre + 'active.png',
                config['vminmax_recfs'])
            Schwartz2001DataModule.save_samples_png(
                inactive_recfs, fn_pre + 'inactive.png',
                config['vminmax_recfs'])

        if model_name:
            print(f'found {active_recfs.shape[0]} active and '
                  f'{inactive_recfs.shape[0]} inactive z dims '
                  f'for {model_name}')

        return active_recfs, active_postmns, inactive_recfs, inactive_postmns
