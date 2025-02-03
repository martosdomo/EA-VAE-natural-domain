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
import math
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import torch

from config import config, results_dir


def dprime_e(s1, s2):
    # Calculates the average sd discriminability index between s1 and s2
    return (np.abs(np.mean(s1) - np.mean(s2))
            / (0.5 * (np.std(s1) + np.std(s2))))


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
    save_samples_png(
        test_ims[:n_samps_sqrt * n_samps_sqrt, :],
        config['image_sample_fn'], config['vminmax_imgs'])
    print(f'  saved {n_samps_sqrt}x{n_samps_sqrt} image samples '
          f'to {config["image_sample_fn"]}')

    return test_ims


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
    save_samples_png(
        recfs[:n_samps_sqrt * n_samps_sqrt, :],
        sample_fn, config['vminmax_recfs'])
    print(f'  saved {n_samps_sqrt}x{n_samps_sqrt} image samples '
          f'to {sample_fn}')

    return recfs


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


def active_recfs_postmns(recfs, postmns, threshold=1.0, fn_pre=None,
                         model_name=None):
    active_recfs = recfs[np.std(recfs, axis=1) >= threshold, :]
    active_postmns = postmns[:, np.std(recfs, axis=1) >= threshold]

    inactive_recfs = recfs[np.std(recfs, axis=1) < threshold, :]
    inactive_postmns = postmns[:, np.std(recfs, axis=1) < threshold]

    if fn_pre:
        save_samples_png(
            active_recfs, fn_pre + 'active.png', config['vminmax_recfs'])
        save_samples_png(
            inactive_recfs, fn_pre + 'inactive.png', config['vminmax_recfs'])

    if model_name:
        print(f'found {active_recfs.shape[0]} active and '
              f'{inactive_recfs.shape[0]} inactive z dims '
              f'for {model_name}')

    return active_recfs, active_postmns, inactive_recfs, inactive_postmns


def plot_three_violins_per_filter_pair(
        resps, filter_idcs, resps_type, model_name, format='png'):
    resps = resps[config['img_indices_from']:config['img_indices_to'], :]

    fig, axs = plt.subplots(len(filter_idcs), len(filter_idcs),
                            sharex=True, sharey=True,
                            figsize=[3.2 * len(filter_idcs),
                                     2.4 * len(filter_idcs)],
                            constrained_layout=True)

    for ii, i in enumerate(filter_idcs):
        for jj, j in enumerate(filter_idcs):
            ax = axs[ii, jj]
            if ii == len(filter_idcs) - 1:
                ax.set_xlabel(f'z dimension {j}')
            if jj == 0:
                ax.set_ylabel(f'z dimension {i}')
            if ii != jj:
                resp_i = resps[:, i]
                resp_j = resps[:, j]
                perc_j = np.percentile(resp_j,
                                       config['three_violins_percentiles'])

                ax.violinplot((resp_i[resp_j < perc_j[0]],
                               resp_i[np.logical_and(resp_j >= perc_j[0],
                                                     resp_j < perc_j[1])],
                               resp_i[resp_j >= perc_j[1]]),
                              showmeans=True)
                ax.set_xticks((1, 2, 3))
                ax.set_xticklabels((
                    f'0-{config["three_violins_percentiles"][0]}%',
                    f'{config["three_violins_percentiles"][0]}-'
                    f'{config["three_violins_percentiles"][1]}%',
                    f'{config["three_violins_percentiles"][1]}-100%',
                ))
                ax.axhline(y=0, color='k', lw=0.5)

    fig.suptitle(f'{model_name}, {resps_type} responses, '
                 f'filters {list(filter_idcs)}')

    fig.savefig(config['three_violins_plot_fn_pre']
                + f'{resps_type}_{model_name}.{format}', format=format)

    plt.close()


def plot_three_violin_std_hists(
        resps_standardvae, resps_eavae, filter_idcs,
        resps_type, model_name_standardvae, model_name_eavae,
        format='png', format2='pdf'):
    resps_standardvae = resps_standardvae[config['img_indices_from']:
                                      config['img_indices_to'], :]
    resps_eavae = resps_eavae[config['img_indices_from']:
                                          config['img_indices_to'], :]

    central_stds_standardvae = []
    flanking_stds_standardvae = []
    central_stds_eavae = []
    flanking_stds_eavae = []

    for ii, i in enumerate(filter_idcs):
        for jj, j in enumerate(filter_idcs):
            if ii != jj:
                resp_i_standardvae = resps_standardvae[:, i]
                resp_j_standardvae = resps_standardvae[:, j]
                perc_j_standardvae = np.percentile(
                    resp_j_standardvae, config['three_violins_percentiles'])

                flanking_stds_standardvae.append(np.std(
                    resp_i_standardvae[resp_j_standardvae < perc_j_standardvae[0]]))
                central_stds_standardvae.append(np.std(
                    resp_i_standardvae[np.logical_and(
                        resp_j_standardvae >= perc_j_standardvae[0],
                        resp_j_standardvae < perc_j_standardvae[1])]))
                flanking_stds_standardvae.append(np.std(
                    resp_i_standardvae[resp_j_standardvae >= perc_j_standardvae[1]]))

                resp_i_eavae = resps_eavae[:, i]
                resp_j_eavae = resps_eavae[:, j]
                perc_j_eavae = np.percentile(
                    resp_j_eavae, config['three_violins_percentiles'])

                flanking_stds_eavae.append(np.std(
                    resp_i_eavae[resp_j_eavae
                                       < perc_j_eavae[0]]))
                central_stds_eavae.append(np.std(
                    resp_i_eavae[np.logical_and(
                        resp_j_eavae >= perc_j_eavae[0],
                        resp_j_eavae < perc_j_eavae[1])]))
                flanking_stds_eavae.append(np.std(
                    resp_i_eavae[resp_j_eavae
                                       >= perc_j_eavae[1]]))

    fig, axs = plt.subplots(
        1, 2, sharex=True, sharey=True,
        figsize=[12.8, 4.8], constrained_layout=True)

    ax = axs[0]
    ax.hist(central_stds_standardvae, bins='auto', density=True,
            label=f'{model_name_standardvae}, central')
    ax.hist(flanking_stds_standardvae, bins='auto', density=True,
            label=f'{model_name_standardvae}, flanking')
    ax.set_xlabel('Standard deviation of response')
    ax.set_ylabel('Probability density')
    dprime = dprime_e(np.array(central_stds_standardvae),
                      np.array(flanking_stds_standardvae))
    ax.set_title(f"$d'_e$ = {dprime:.3g}")
    ax.legend()

    ax = axs[1]
    ax.hist(central_stds_eavae, bins='auto', density=True,
            label=f'{model_name_eavae}, central')
    ax.hist(flanking_stds_eavae, bins='auto', density=True,
            label=f'{model_name_eavae}, flanking')
    ax.set_xlabel('Standard deviation of response')
    ax.set_ylabel('Probability density')
    dprime = dprime_e(np.array(central_stds_eavae),
                      np.array(flanking_stds_eavae))
    ax.set_title(f"$d'_e$ = {dprime:.3g}")
    ax.legend()

    fig.suptitle(f'{resps_type.capitalize()} responses, '
                 f'{len(filter_idcs)} filters')

    fig.savefig(config['three_violins_std_hists_plot_fn_pre']
                + f'{resps_type}_{len(filter_idcs)}filters.{format}',
                format=format)
    fig.savefig(config['three_violins_std_hists_plot_fn_pre']
                + f'{resps_type}_{len(filter_idcs)}filters.{format2}',
                format=format2)

    plt.close()

    return (central_stds_standardvae, flanking_stds_standardvae,
            central_stds_eavae, flanking_stds_eavae)


def plot_bowties(
        resps, filter_idcs, resps_type, model_name, n_bins,
        format='png', format2='pdf'):
    resps = resps[config['img_indices_from']:config['img_indices_to'], :]

    bins_sum = np.zeros((n_bins, n_bins))

    fig, axs = plt.subplots(len(filter_idcs), len(filter_idcs),
                            layout='compressed',
                            figsize=[1.5 * len(filter_idcs),
                                     1.2 * len(filter_idcs)])

    for ii, i in enumerate(filter_idcs):
        for jj, j in enumerate(filter_idcs):
            ax = axs[ii, jj]
            if ii == len(filter_idcs) - 1:
                ax.set_xlabel(f'z dimension {j}')
            if jj == 0:
                ax.set_ylabel(f'z dimension {i}')
            if ii == jj:
                max_resp = np.max(np.abs(resps[:, i])) * 1.001
                ax.set_xlim(-max_resp, max_resp)
                ax.set_ylim(-max_resp, max_resp)
                ax.set_xticks((-max_resp, max_resp))
                ax.set_yticks((-max_resp, max_resp))
                ax.set_aspect('equal')
            else:
                max_resp = np.max(np.abs(resps[:, [i, j]])) * 1.001
                idx_i = np.floor((resps[:, i] + max_resp)
                                 / (2 * max_resp) * n_bins).astype(np.int64)
                idx_j = np.floor((resps[:, j] + max_resp)
                                 / (2 * max_resp) * n_bins).astype(np.int64)
                idx_ij = n_bins * idx_i + idx_j

                bins = np.zeros((n_bins, n_bins))
                np.add.at(np.ravel(bins), idx_ij, 1)
                bins_sum += bins

                for j in range(n_bins):
                    if np.max(bins[:, j]) > 0:
                        bins[:, j] /= np.max(bins[:, j])

                ax.imshow(bins, cmap='gray', vmin=0, vmax=1)

                ax.set_xticks((-0.5, n_bins - 0.5))
                ax.set_xticklabels((f'{-max_resp:.3g}', f'{max_resp:.3g}'))
                ax.set_yticks((-0.5, n_bins - 0.5))
                ax.set_yticklabels((f'{-max_resp:.3g}', f'{max_resp:.3g}'))
                ax.set_aspect('equal')

    fig.suptitle(f'{model_name}, {resps_type} responses, '
                 f'filters {list(filter_idcs)}')

    fig.savefig(config['butterflies_plot_fn_pre']
                + f'{resps_type}_{model_name}.{format}', format=format)
    fig.savefig(config['butterflies_plot_fn_pre']
                + f'{resps_type}_{model_name}.{format2}', format=format2)

    plt.close

    for j in range(n_bins):
        if np.max(bins_sum[:, j]) > 0:
            bins_sum[:, j] /= np.max(bins_sum[:, j])

    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(bins_sum, cmap='gray', vmin=0, vmax=1)
    fig.suptitle(f'{model_name}, {resps_type} responses, '
                 f'filters {list(filter_idcs)}')
    fig.savefig(config['butterflies_plot_fn_pre']
                + f'{resps_type}_{model_name}_summed.{format}', format=format)
    fig.savefig(config['butterflies_plot_fn_pre']
                + f'{resps_type}_{model_name}_summed.{format2}',
                format=format2)
    plt.close


if __name__ == '__main__':
    os.makedirs(results_dir, exist_ok=True)

    # standardvae
    model_name_standardvae = config['standardvae_modelname']

    if 'standardvae_ims_fn' in config:
        test_ims_standardvae = load_test_images(
            f'{model_name_standardvae}{config["standardvae_ims_fn"]}')
    else:
        test_ims_standardvae = load_test_images(
            f'{model_name_standardvae}_input_ims.pkl')
    if 'standardvae_recfs_fn' in config:
        recfs_standardvae = load_recfs(
            f'{model_name_standardvae}{config["standardvae_recfs_fn"]}',
            model_name_standardvae,
            config[f'recf_{model_name_standardvae}_sample_fn'])
    else:
        recfs_standardvae = load_recfs(
            f'{model_name_standardvae}_receptive.pkl', model_name_standardvae,
            config[f'recf_{model_name_standardvae}_sample_fn'])
    if 'standardvae_postmns_fn' in config:
        postmns_standardvae = load_postmns(
            f'{model_name_standardvae}{config["standardvae_postmns_fn"]}',
            model_name_standardvae)
    else:
        postmns_standardvae = load_postmns(
            f'{model_name_standardvae}_posteriors_cpu.pkl',
            model_name_standardvae)

    if test_ims_standardvae.shape[0] != postmns_standardvae.shape[0]:
        print('the number of images and posteriors differ; '
              'using the smaller number everywhere')
        n_ims = min(test_ims_standardvae.shape[0], postmns_standardvae.shape[0])
        test_ims_standardvae = test_ims_standardvae[:n_ims, :]
        postmns_standardvae = postmns_standardvae[:n_ims, :]
    if recfs_standardvae.shape[0] != postmns_standardvae.shape[1]:
        print('the number of z dims in receptive fields and posteriors '
              'differ; using the smaller number everywhere')
        n_zdims = min(recfs_standardvae.shape[0], postmns_standardvae.shape[1])
        recfs_standardvae = recfs_standardvae[:n_zdims, :]
        postmns_standardvae = postmns_standardvae[:, :n_zdims]

    active_recfs_standardvae, active_postmns_standardvae, \
        inactive_recfs_standardvae, inactive_postmns_standardvae = \
        active_recfs_postmns(
            recfs_standardvae, postmns_standardvae,
            config['active_recfs_threshold'],
            config[f'recf_{model_name_standardvae}_active_fn_pre'],
            model_name_standardvae)

    linear_responses_standardvae = np.matmul(
        test_ims_standardvae, active_recfs_standardvae.T)
    print(f'calculated linear responses for {model_name_standardvae}')
    print(f'  shape: {linear_responses_standardvae.shape}; '
          f'dtype: {linear_responses_standardvae.dtype}')
    print(f'  mean: {np.mean(linear_responses_standardvae):.5f}; '
          f'std: {np.std(linear_responses_standardvae):.5f}')

    plot_three_violins_per_filter_pair(
        linear_responses_standardvae, config['three_violins_filter_idcs'],
        'linear', config['standardvae_modelname'], format='png')
    plot_three_violins_per_filter_pair(
        linear_responses_standardvae, config['three_violins_filter_idcs'],
        'linear', config['standardvae_modelname'], format='pdf')

    plot_three_violins_per_filter_pair(
        active_postmns_standardvae, config['three_violins_filter_idcs'],
        'model', config['standardvae_modelname'], format='png')
    plot_three_violins_per_filter_pair(
        active_postmns_standardvae, config['three_violins_filter_idcs'],
        'model', config['standardvae_modelname'], format='pdf')

    # eavae
    model_name_eavae = config['eavae_modelname']
    if 'eavae_ims_fn' in config:
        test_ims_eavae = load_test_images(
            f'{model_name_eavae}{config["eavae_ims_fn"]}')
    else:
        test_ims_eavae = load_test_images(
            f'{model_name_eavae}_input_ims.pkl')
    if 'eavae_recfs_fn' in config:
        recfs_eavae = load_recfs(
            f'{model_name_eavae}{config["eavae_recfs_fn"]}',
            model_name_eavae,
            config[f'recf_{model_name_eavae}_sample_fn'])
    else:
        recfs_eavae = load_recfs(
            f'{model_name_eavae}_receptive.pkl', model_name_eavae,
            config[f'recf_{model_name_eavae}_sample_fn'])
    if 'eavae_postmns_fn' in config:
        postmns_eavae = load_postmns(
            f'{model_name_eavae}{config["eavae_postmns_fn"]}',
            model_name_eavae)
    else:
        postmns_eavae = load_postmns(
            f'{model_name_eavae}_posteriors_cpu.pkl',
            model_name_eavae)

    if test_ims_eavae.shape[0] != postmns_eavae.shape[0]:
        print('the number of images and posteriors differ; '
              'using the smaller number everywhere')
        n_ims = min(test_ims_eavae.shape[0],
                    postmns_eavae.shape[0])
        test_ims_eavae = test_ims_eavae[:n_ims, :]
        postmns_eavae = postmns_eavae[:n_ims, :]
    if recfs_eavae.shape[0] != postmns_eavae.shape[1]:
        print('the number of z dims in receptive fields and posteriors '
              'differ; using the smaller number everywhere')
        n_zdims = min(recfs_eavae.shape[0], postmns_eavae.shape[1])
        recfs_eavae = recfs_eavae[:n_zdims, :]
        postmns_eavae = postmns_eavae[:, :n_zdims]

    active_recfs_eavae, active_postmns_eavae, \
        inactive_recfs_eavae, inactive_postmns_eavae = \
        active_recfs_postmns(
            recfs_eavae, postmns_eavae,
            config['active_recfs_threshold'],
            config[f'recf_{model_name_eavae}_active_fn_pre'],
            model_name_eavae)

    linear_responses_eavae = np.matmul(
        test_ims_eavae, active_recfs_eavae.T)
    print(f'calculated linear responses for {model_name_eavae}')
    print(f'  shape: {linear_responses_eavae.shape}; '
          f'dtype: {linear_responses_eavae.dtype}')
    print(f'  mean: {np.mean(linear_responses_eavae):.5f}; '
          f'std: {np.std(linear_responses_eavae):.5f}')

    plot_three_violins_per_filter_pair(
        linear_responses_eavae, config['three_violins_filter_idcs'],
        'linear', config['eavae_modelname'], format='png')
    plot_three_violins_per_filter_pair(
        linear_responses_eavae, config['three_violins_filter_idcs'],
        'linear', config['eavae_modelname'], format='pdf')

    plot_three_violins_per_filter_pair(
        active_postmns_eavae, config['three_violins_filter_idcs'],
        'model', config['eavae_modelname'], format='png')
    plot_three_violins_per_filter_pair(
        active_postmns_eavae, config['three_violins_filter_idcs'],
        'model', config['eavae_modelname'], format='pdf')

    (linresp_central_stds_standardvae, linresp_flanking_stds_standardvae,
     linresp_central_stds_eavae, linresp_flanking_stds_eavae) = (
     plot_three_violin_std_hists(linear_responses_standardvae,
                                 linear_responses_eavae,
                                 config['three_violins_std_hists_filter_idcs'],
                                 'linear',
                                 config['standardvae_modelname'],
                                 config['eavae_modelname'],
                                 format='png', format2='pdf'))

    (modelresp_central_stds_standardvae, modelresp_flanking_stds_standardvae,
     modelresp_central_stds_eavae,
     modelresp_flanking_stds_eavae) = (
     plot_three_violin_std_hists(active_postmns_standardvae,
                                 active_postmns_eavae,
                                 config['three_violins_std_hists_filter_idcs'],
                                 'model',
                                 config['standardvae_modelname'],
                                 config['eavae_modelname'],
                                 format='png', format2='pdf'))

    
    three_violins_stds = {
            'linresp_central_stds_standardvae': linresp_central_stds_standardvae,
            'linresp_flanking_stds_standardvae': linresp_flanking_stds_standardvae,
            'linresp_central_stds_eavae': linresp_central_stds_eavae,
            'linresp_flanking_stds_eavae': linresp_flanking_stds_eavae,
            'modelresp_central_stds_standardvae': modelresp_central_stds_standardvae,
            'modelresp_flanking_stds_standardvae': modelresp_flanking_stds_standardvae,
            'modelresp_central_stds_eavae': (
                modelresp_central_stds_eavae),
            'modelresp_flanking_stds_eavae': (
                modelresp_flanking_stds_eavae),
        }
    # save three_violins_stds with pickle
    with open(config['three_violins_stds_fn'], 'wb') as f:
        pickle.dump(three_violins_stds, f)

    plot_bowties(linear_responses_standardvae, range(10),
                 'linear', config['standardvae_modelname'],
                 config['bowties_n_bins'], 'png', 'pdf')
    plot_bowties(linear_responses_eavae, range(10),
                 'linear', config['eavae_modelname'],
                 config['bowties_n_bins'], 'png', 'pdf')
    plot_bowties(active_postmns_standardvae, range(10),
                 'model', config['standardvae_modelname'],
                 config['bowties_n_bins'], 'png', 'pdf')
    plot_bowties(active_postmns_eavae, range(10),
                 'model', config['eavae_modelname'],
                 config['bowties_n_bins'], 'png', 'pdf')
