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
import os
import pytorch_lightning as pl
import numpy as np
import pickle

from config import config, results_dir
from datamodule import Schwartz2001DataModule
from linearmodel import LinearModel
from divisivenormmodel import DivisiveNormModel
import plots


def fit_schwartz2001(seed):
    pl.seed_everything(seed)

    sigma_square_unsign = getattr(np, config['sigma_square_unsign'])
    w_unsign = getattr(np, config['w_unsign'])

    dm_standardvae = Schwartz2001DataModule(config['standardvae_modelname'])
    dm_eavae = Schwartz2001DataModule(config['eavae_modelname'])

    print('\nTraining linearmodel_standardvae...')
    linearmodel_standardvae = LinearModel(**config)
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=config['devices'],
        default_root_dir=results_dir,
        max_epochs=config['epochs'],
        precision=config['precision'],
        callbacks=[pl.callbacks.EarlyStopping('val_loss')],
        enable_progress_bar=config['enable_progress_bar'],
    )
    trainer.fit(linearmodel_standardvae, datamodule=dm_standardvae)
    sigma_linearmodel_standardvae = np.sqrt(sigma_square_unsign(
        linearmodel_standardvae.signed_sigma_square.item()))
    print(f'sigma_linearmodel_standardvae = {sigma_linearmodel_standardvae:.5f}')

    print('\nTraining linearmodel_eavae...')
    linearmodel_eavae = LinearModel(**config)
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=config['devices'],
        default_root_dir=results_dir,
        max_epochs=config['epochs'],
        precision=config['precision'],
        callbacks=[pl.callbacks.EarlyStopping('val_loss')],
        enable_progress_bar=config['enable_progress_bar'],
    )
    trainer.fit(linearmodel_eavae, datamodule=dm_eavae)
    sigma_linearmodel_eavae = np.sqrt(sigma_square_unsign(
        linearmodel_eavae.signed_sigma_square.item()))
    print(f'sigma_linearmodel_eavae = '
          f'{sigma_linearmodel_eavae:.5f}')

    print('\nCreating divisivenormmodel_standardvae_untrained...')
    divisivenormmodel_standardvae_untrained = DivisiveNormModel(
        dm_standardvae.active_postmns.shape[1], **config)
    sigma_divisivenormmodel_standardvae_untrained = np.sqrt(sigma_square_unsign(
        divisivenormmodel_standardvae_untrained.signed_sigma_square.item()))
    print(f'sigma_divisivenormmodel_standardvae_untrained = '
          f'{sigma_divisivenormmodel_standardvae_untrained:.5f}')
    w_standardvae_untrained = w_unsign(
        divisivenormmodel_standardvae_untrained.signed_w.detach().numpy())
    print(f'w_standardvae_untrained: min: {np.min(w_standardvae_untrained):.5g}, '
          f'max: {np.max(w_standardvae_untrained):.5g}, '
          f'mean: {np.mean(w_standardvae_untrained):.5g}, '
          f'median: {np.median(w_standardvae_untrained):.5g}')
    pickle.dump(w_standardvae_untrained,
                open(results_dir +
                     f'w_standardvae_untrained_{config["standardvae_modelname"]}_'
                     f'seed{seed}.pkl', 'wb'))

    print('\nTraining divisivenormmodel_standardvae...')
    divisivenormmodel_standardvae = DivisiveNormModel(
        dm_standardvae.active_postmns.shape[1], **config)
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=config['devices'],
        default_root_dir=results_dir,
        max_epochs=config['epochs'],
        precision=config['precision'],
        callbacks=[pl.callbacks.EarlyStopping('val_loss')],
        enable_progress_bar=config['enable_progress_bar'],
    )
    trainer.fit(divisivenormmodel_standardvae, datamodule=dm_standardvae)
    sigma_divisivenormmodel_standardvae = np.sqrt(sigma_square_unsign(
        divisivenormmodel_standardvae.signed_sigma_square.item()))
    print(f'sigma_divisivenormmodel_standardvae = '
          f'{sigma_divisivenormmodel_standardvae:.5f}')
    w_standardvae = w_unsign(
        divisivenormmodel_standardvae.signed_w.detach().numpy())
    print(f'w_standardvae: min: {np.min(w_standardvae):.5g}, '
          f'max: {np.max(w_standardvae):.5g}, '
          f'mean: {np.mean(w_standardvae):.5g}, '
          f'median: {np.median(w_standardvae):.5g}')
    pickle.dump(w_standardvae,
                open(results_dir +
                     f'w_standardvae_{config["standardvae_modelname"]}_'
                     f'seed{seed}.pkl', 'wb'))

    print('\nCreating divisivenormmodel_eavae_untrained...')
    divisivenormmodel_eavae_untrained = DivisiveNormModel(
        dm_eavae.active_postmns.shape[1], **config)
    sigma_divisivenormmodel_eavae_untrained = \
        np.sqrt(sigma_square_unsign(
            divisivenormmodel_eavae_untrained.signed_sigma_square.item()
        ))
    print(f'sigma_divisivenormmodel_eavae_untrained = '
          f'{sigma_divisivenormmodel_eavae_untrained:.5f}')
    w_eavae_untrained = w_unsign(
        divisivenormmodel_eavae_untrained.signed_w.detach().numpy())
    print(f'w_eavae_untrained: min: '
          f'{np.min(w_eavae_untrained):.5g}, '
          f'max: {np.max(w_eavae_untrained):.5g}, '
          f'mean: {np.mean(w_eavae_untrained):.5g}, '
          f'median: {np.median(w_eavae_untrained):.5g}')
    pickle.dump(w_eavae_untrained,
                open(results_dir +
                     f'w_eavae_untrained_'
                     f'{config["eavae_modelname"]}_'
                     f'seed{seed}.pkl', 'wb'))

    print('\nTraining divisivenormmodel_eavae...')
    divisivenormmodel_eavae = DivisiveNormModel(
        dm_eavae.active_postmns.shape[1], **config)
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=config['devices'],
        default_root_dir=results_dir,
        max_epochs=config['epochs'],
        precision=config['precision'],
        callbacks=[pl.callbacks.EarlyStopping('val_loss')],
        enable_progress_bar=config['enable_progress_bar'],
    )
    trainer.fit(divisivenormmodel_eavae, datamodule=dm_eavae)
    sigma_divisivenormmodel_eavae = np.sqrt(sigma_square_unsign(
        divisivenormmodel_eavae.signed_sigma_square.item()))
    print(f'sigma_divisivenormmodel_eavae = '
          f'{sigma_divisivenormmodel_eavae:.5f}')
    w_eavae = w_unsign(
        divisivenormmodel_eavae.signed_w.detach().numpy())
    print(f'w_eavae: min: {np.min(w_eavae):.5g}, '
          f'max: {np.max(w_eavae):.5g}, '
          f'mean: {np.mean(w_eavae):.5g}, '
          f'median: {np.median(w_eavae):.5g}')
    pickle.dump(w_eavae,
                open(results_dir +
                     f'w_eavae_{config["eavae_modelname"]}_'
                     f'seed{seed}.pkl', 'wb'))

    ksps = plots.plot_w_histograms(w_standardvae_untrained, w_standardvae,
                                   w_eavae_untrained, w_eavae,
                                   seed)

    corrs_standardvae = plots.plot_recf_props_vs_w_corrs(
        dm_standardvae.active_recfs, w_standardvae,
        config['standardvae_modelname'], seed)
    corrs_eavae = plots.plot_recf_props_vs_w_corrs(
        dm_eavae.active_recfs, w_eavae,
        config['eavae_modelname'], seed)

    ws = {
        'w_standardvae_untrained': w_standardvae_untrained,
        'w_standardvae': w_standardvae,
        'w_eavae_untrained': w_eavae_untrained,
        'w_eavae': w_eavae,
    }

    return ws, ksps, corrs_standardvae, corrs_eavae


if __name__ == '__main__':
    os.makedirs(results_dir, exist_ok=True)

    seeds = config['random_seeds']

    ws_per_seed = []
    ksps_per_seed = []
    corrs_standardvae_per_seed = []
    corrs_eavae_per_seed = []

    for seed in seeds:
        ws, ksps, corrs_standardvae, corrs_eavae = fit_schwartz2001(seed)

        ws_per_seed.append(ws)
        ksps_per_seed.append(ksps)
        corrs_standardvae_per_seed.append(corrs_standardvae)
        corrs_eavae_per_seed.append(corrs_eavae)

    plots.plot_w_histograms_cumulative(ws_per_seed, ksps_per_seed, seeds)
