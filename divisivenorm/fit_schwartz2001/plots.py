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
import numpy as np
import scipy.spatial.distance as dist
from scipy import stats
import matplotlib.pyplot as plt

from config import config, results_dir
import eval


def plot_w_histograms(w_standardvae_untrained, w_standardvae,
                      w_eavae_untrained, w_eavae, seed,
                      format='png'):
    bins = np.geomspace(config['hist_left_edge'],
                        config['hist_right_edge'],
                        config['hist_num_bins_plus_1'])

    # Plot all matrix elements in $w_{ij}$
    fig, ax = plt.subplots(constrained_layout=True)
    ax.hist(w_standardvae.flatten(),
            bins=bins, density=True, histtype='step',
            label=config['standardvae_modelname'])
    ax.hist(w_standardvae_untrained.flatten(),
            bins=bins, density=True, histtype='step',
            label=config['standardvae_modelname'] + '_untrained')
    ax.hist(w_eavae.flatten(), bins=bins,
            density=True, histtype='step',
            label=config['eavae_modelname'])
    ax.hist(w_eavae_untrained.flatten(), bins=bins,
            density=True, histtype='step',
            label=config['eavae_modelname'] + '_untrained')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel('$w_{ij}$')
    ax.set_ylabel('probability density')
    fig.suptitle(f'{config["standardvae_modelname"]} vs. '
                 f'{config["eavae_modelname"]}, '
                 f'random seed: {seed}')

    ksp_untrained = stats.kstest(w_standardvae_untrained,
                                 w_eavae_untrained,
                                 axis=None).pvalue
    ksp_trained = stats.kstest(w_standardvae, w_eavae, axis=None).pvalue
    ax.set_title(f'Kolmogorov-Smirnov P: '
                 f'untrained: {ksp_untrained:.5g}, '
                 f'trained: {ksp_trained:.5g}')

    fig.savefig(results_dir + f'w_hists_{config["standardvae_modelname"]}_'
                f'{config["eavae_modelname"]}_'
                f'seed{seed}.{format}', format=format)

    plt.close()

    # Plot only off-diagonal matrix elements in $w_{ij}$
    fig, ax = plt.subplots(constrained_layout=True)
    ax.hist(eval.offdiagonal(w_standardvae),
            bins=bins, density=True, histtype='step',
            label=config['standardvae_modelname'])
    ax.hist(eval.offdiagonal(w_standardvae_untrained),
            bins=bins, density=True, histtype='step',
            label=config['standardvae_modelname'] + '_untrained')
    ax.hist(eval.offdiagonal(w_eavae),
            bins=bins, density=True, histtype='step',
            label=config['eavae_modelname'])
    ax.hist(eval.offdiagonal(w_eavae_untrained),
            bins=bins, density=True, histtype='step',
            label=config['eavae_modelname'] + '_untrained')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 1e5)
    ax.legend()
    ax.set_xlabel('$w_{ij}, i \\neq j$')
    ax.set_ylabel('probability density')
    ax.set_title(f'{config["standardvae_modelname"]} vs. '
                 f'{config["eavae_modelname"]}, '
                 f'random seed: {seed}')

    ksp_untrained_offdiagonal = stats.kstest(
        eval.offdiagonal(w_standardvae_untrained),
        eval.offdiagonal(w_eavae_untrained),
        axis=None).pvalue
    ksp_trained_offdiagonal = stats.kstest(
        eval.offdiagonal(w_standardvae),
        eval.offdiagonal(w_eavae),
        axis=None).pvalue
    fig.suptitle(f'Kolmogorov-Smirnov P: '
                 f'untrained: {ksp_untrained_offdiagonal:.10g}, '
                 #f'trained: {ksp_trained_offdiagonal:.10g}')
                 f'trained: {ksp_trained_offdiagonal if ksp_trained_offdiagonal > 0 else "<1e-10"}')

    fig.savefig(results_dir + f'w_offdiagonal_hists_'
                f'{config["standardvae_modelname"]}_'
                f'{config["eavae_modelname"]}_'
                f'seed{seed}.{format}', format=format)

    plt.close()

    return {
        'ksp_untrained': ksp_untrained,
        'ksp_trained': ksp_trained,
        'ksp_untrained_offdiagonal': ksp_untrained_offdiagonal,
        'ksp_trained_offdiagonal': ksp_trained_offdiagonal,
    }


def plot_w_histograms_cumulative(ws_per_seed, ksps_per_seed, seeds,
                                 format='png'):
    alpha = 0.3
    bins = np.geomspace(config['hist_left_edge'],
                        config['hist_right_edge'],
                        config['hist_num_bins_plus_1'])

    # Plot all matrix elements in $w_{ij}$
    fig, ax = plt.subplots(constrained_layout=True)
    for i_seed, seed in enumerate(seeds):
        ax.hist(ws_per_seed[i_seed]['w_standardvae'].flatten(),
                bins=bins, density=True, histtype='step',
                alpha=alpha, color='C0')
        ax.hist(ws_per_seed[i_seed]['w_standardvae_untrained'].flatten(),
                bins=bins, density=True, histtype='step',
                alpha=alpha, color='C1')
        ax.hist(ws_per_seed[i_seed]['w_eavae'].flatten(),
                bins=bins, density=True, histtype='step',
                alpha=alpha, color='C2')
        ax.hist(ws_per_seed[i_seed]['w_eavae_untrained'].flatten(),
                bins=bins, density=True, histtype='step',
                alpha=alpha, color='C3')

    w_standardvae_all = np.zeros_like(
        ws_per_seed[0]['w_standardvae'], shape=(0,))
    w_standardvae_untrained_all = np.zeros_like(
        ws_per_seed[0]['w_standardvae_untrained'], shape=(0,))
    w_eavae_all = np.zeros_like(
        ws_per_seed[0]['w_eavae'], shape=(0,))
    w_eavae_untrained_all = np.zeros_like(
        ws_per_seed[0]['w_eavae_untrained'], shape=(0,))
    ksps_untrained_all = np.zeros_like(
        ksps_per_seed[0]['ksp_untrained'], shape=(len(seeds)),)
    ksps_trained_all = np.zeros_like(
        ksps_per_seed[0]['ksp_trained'], shape=(len(seeds)),)
    for i_seed, seed in enumerate(seeds):
        w_standardvae_all = np.concatenate(
            (w_standardvae_all,
             ws_per_seed[i_seed]['w_standardvae'].flatten()))
        w_standardvae_untrained_all = np.concatenate(
            (w_standardvae_untrained_all,
             ws_per_seed[i_seed]['w_standardvae_untrained'].flatten()))
        w_eavae_all = np.concatenate(
            (w_eavae_all,
             ws_per_seed[i_seed]['w_eavae'].flatten()))
        w_eavae_untrained_all = np.concatenate(
            (w_eavae_untrained_all,
             ws_per_seed[i_seed]['w_eavae_untrained'].flatten()))
        ksps_untrained_all[i_seed] = ksps_per_seed[i_seed]['ksp_untrained']
        ksps_trained_all[i_seed] = ksps_per_seed[i_seed]['ksp_trained']

    ax.hist(w_standardvae_all,
            bins=bins, density=True, histtype='step', color='C0',
            label=config['standardvae_modelname'])
    ax.hist(w_standardvae_untrained_all,
            bins=bins, density=True, histtype='step', color='C1',
            label=config['standardvae_modelname'] + '_untrained')
    ax.hist(w_eavae_all, bins=bins,
            density=True, histtype='step', color='C2',
            label=config['eavae_modelname'])
    ax.hist(w_eavae_untrained_all, bins=bins,
            density=True, histtype='step', color='C3',
            label=config['eavae_modelname'] + '_untrained')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel('$w_{ij}$')
    ax.set_ylabel('probability density')
    fig.suptitle(f'{config["standardvae_modelname"]} vs. '
                 f'{config["eavae_modelname"]}, '
                 f'random seeds: {seeds}')

    ksp_untrained_mean = np.mean(ksps_untrained_all)
    ksp_untrained_sem = np.std(ksps_untrained_all) / np.sqrt(len(seeds))
    ksp_trained_mean = np.mean(ksps_trained_all)
    ksp_trained_sem = np.std(ksps_trained_all) / np.sqrt(len(seeds))
    ax.set_title(f'Kolmogorov-Smirnov P: '
                 f'untrained: {ksp_untrained_mean:.3g}±'
                 f'{ksp_untrained_sem:.3g}, '
                 f'trained: {ksp_trained_mean:.3g}±'
                 f'{ksp_trained_sem:.3g}')

    fig.savefig(results_dir + f'w_hists_{config["standardvae_modelname"]}_'
                f'{config["eavae_modelname"]}_'
                f'allseeds.{format}', format=format)

    plt.close()

    # Plot only off-diagonal matrix elements in $w_{ij}$
    fig, ax = plt.subplots(constrained_layout=True)
    for i_seed, seed in enumerate(seeds):
        ax.hist(eval.offdiagonal(ws_per_seed[i_seed]['w_standardvae']),
                bins=bins, density=True, histtype='step',
                alpha=alpha, color='C0')
        ax.hist(eval.offdiagonal(ws_per_seed[i_seed]['w_standardvae_untrained']),
                bins=bins, density=True, histtype='step',
                alpha=alpha, color='C1')
        ax.hist(eval.offdiagonal(ws_per_seed[i_seed]['w_eavae']),
                bins=bins, density=True, histtype='step',
                alpha=alpha, color='C2')
        ax.hist(eval.offdiagonal(
            ws_per_seed[i_seed]['w_eavae_untrained']),
                bins=bins, density=True, histtype='step',
                alpha=alpha, color='C3')

    w_standardvae_offdiagonal_all = np.zeros_like(
        ws_per_seed[0]['w_standardvae'], shape=(0,))
    w_standardvae_untrained_offdiagonal_all = np.zeros_like(
        ws_per_seed[0]['w_standardvae_untrained'], shape=(0,))
    w_eavae_offdiagonal_all = np.zeros_like(
        ws_per_seed[0]['w_eavae'], shape=(0,))
    w_eavae_untrained_offdiagonal_all = np.zeros_like(
        ws_per_seed[0]['w_eavae_untrained'], shape=(0,))
    ksps_untrained_offdiagonal_all = np.zeros_like(
        ksps_per_seed[0]['ksp_untrained_offdiagonal'], shape=(len(seeds)),)
    ksps_trained_offdiagonal_all = np.zeros_like(
        ksps_per_seed[0]['ksp_trained_offdiagonal'], shape=(len(seeds)),)
    for i_seed, seed in enumerate(seeds):
        w_standardvae_offdiagonal_all = np.concatenate(
            (w_standardvae_offdiagonal_all,
             eval.offdiagonal(ws_per_seed[i_seed]['w_standardvae'])))
        w_standardvae_untrained_offdiagonal_all = np.concatenate(
            (w_standardvae_untrained_offdiagonal_all,
             eval.offdiagonal(ws_per_seed[i_seed]['w_standardvae_untrained'])))
        w_eavae_offdiagonal_all = np.concatenate(
            (w_eavae_offdiagonal_all,
             eval.offdiagonal(ws_per_seed[i_seed]['w_eavae'])))
        w_eavae_untrained_offdiagonal_all = np.concatenate(
            (w_eavae_untrained_offdiagonal_all,
             eval.offdiagonal(ws_per_seed[i_seed]['w_eavae_untrained'])))
        ksps_untrained_offdiagonal_all[i_seed] = \
            ksps_per_seed[i_seed]['ksp_untrained_offdiagonal']
        ksps_trained_offdiagonal_all[i_seed] = \
            ksps_per_seed[i_seed]['ksp_trained_offdiagonal']

    ax.hist(w_standardvae_offdiagonal_all,
            bins=bins, density=True, histtype='step', color='C0',
            label=config['standardvae_modelname'])
    ax.hist(w_standardvae_untrained_offdiagonal_all,
            bins=bins, density=True, histtype='step', color='C1',
            label=config['standardvae_modelname'] + '_untrained')
    ax.hist(w_eavae_offdiagonal_all, bins=bins,
            density=True, histtype='step', color='C2',
            label=config['eavae_modelname'])
    ax.hist(w_eavae_untrained_offdiagonal_all, bins=bins,
            density=True, histtype='step', color='C3',
            label=config['eavae_modelname'] + '_untrained')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel('$w_{ij}, i \\neq j$')
    ax.set_ylabel('probability density')
    fig.suptitle(f'{config["standardvae_modelname"]} vs. '
                 f'{config["eavae_modelname"]}, '
                 f'random seeds: {seeds}')

    ksp_untrained_offdiagonal_mean = np.mean(ksps_untrained_offdiagonal_all)
    ksp_untrained_offdiagonal_sem = np.std(ksps_untrained_offdiagonal_all) \
        / np.sqrt(len(seeds))
    ksp_trained_offdiagonal_mean = np.mean(ksps_trained_offdiagonal_all)
    ksp_trained_offdiagonal_sem = np.std(ksps_trained_offdiagonal_all) \
        / np.sqrt(len(seeds))
    ax.set_title(f'Kolmogorov-Smirnov P: '
                 f'untrained: {ksp_untrained_offdiagonal_mean:.3g}±'
                 f'{ksp_untrained_offdiagonal_sem:.3g}, '
                 f'trained: {ksp_trained_offdiagonal_mean:.3g}±'
                 f'{ksp_trained_offdiagonal_sem:.3g}')

    fig.savefig(results_dir + f'w_offdiagonal_hists_'
                f'{config["standardvae_modelname"]}_'
                f'{config["eavae_modelname"]}_'
                f'allseeds.{format}', format=format)

    plt.close()


def plot_recf_props_vs_w_corrs(recfs, w, model_name, seed, format='png'):
    coms = np.apply_along_axis(eval.calc_center_of_mass, 1, recfs)
    com_distances = eval.offdiagonal(
        dist.squareform(dist.pdist(coms, metric='euclidean')))

    wvs = np.apply_along_axis(eval.calc_wave_vector, 1, recfs)
    # unify wv angles to interval [-pi/2, pi/2]
    wvs = np.piecewise(wvs,
                       [wvs[:, 0] < 0, wvs[:, 0] >= 0],
                       [lambda x: -x, lambda x: x])
    wv_norm_distances = eval.offdiagonal(
        dist.squareform(dist.pdist(
            np.linalg.norm(wvs, axis=1, keepdims=True), metric='euclidean')))
    wv_cos_distances = eval.offdiagonal(
        dist.squareform(dist.pdist(wvs, metric='cosine')))
    wv_cos_distances[np.isnan(wv_cos_distances)] = 2.0

    w_offdiagonal = eval.offdiagonal(w)

    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    markersize = 1
    alpha = 0.1

    # distances between receptive field centers vs. w
    ax = axs[0, 0]
    ax.scatter(com_distances, w_offdiagonal, s=markersize, alpha=alpha)
    ax.set_xlabel('distance between receptive\nfield centers [pixels]')
    ax.set_ylabel('$w_{ij}$')
    ax.set_xlim(-1, 21)
    ax.set_ylim(5e-7, 50)
    ax.set_yscale('log')
    corr_com_distances = np.corrcoef(com_distances, w_offdiagonal)[0, 1]
    ax.set_title(f'corr. coeff.: {corr_com_distances:.5g}')

    # cosine distances between dominant wave vectors vs. w
    ax = axs[0, 1]
    ax.scatter(wv_cos_distances, w_offdiagonal, s=markersize, alpha=alpha)
    ax.set_xlabel('cosine distance between\ndominant wave vectors')
    ax.set_ylabel('$w_{ij}$')
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(5e-7, 50)
    ax.set_yscale('log')
    corr_wv_cos_distances = np.corrcoef(wv_cos_distances, w_offdiagonal)[0, 1]
    ax.set_title(f'corr. coeff.: {corr_wv_cos_distances:.5g}')

    # distances between wave vector amplitudes vs. w
    ax = axs[1, 0]
    ax.scatter(wv_norm_distances, w_offdiagonal, s=markersize, alpha=alpha)
    ax.set_xlabel('distance between wave\nvector amplitudes [1/pixels]')
    ax.set_ylabel('$w_{ij}$')
    ax.set_xlim(-0.2, 3.7)
    ax.set_ylim(5e-7, 50)
    ax.set_yscale('log')
    corr_wv_norm_distances = np.corrcoef(wv_norm_distances,
                                         w_offdiagonal)[0, 1]
    ax.set_title(f'corr. coeff.: {corr_wv_norm_distances:.5g}')

    # distances between receptive field centers vs. w
    sum_scaled_distances = (com_distances / np.max(com_distances)
                            + wv_cos_distances / np.max(wv_cos_distances)
                            + wv_norm_distances / np.max(wv_norm_distances))
    ax = axs[1, 1]
    ax.scatter(sum_scaled_distances, w_offdiagonal, s=markersize, alpha=alpha)
    ax.set_xlabel('summed scaled distances')
    ax.set_ylabel('$w_{ij}$')
    ax.set_xlim(-0.1, 3.1)
    ax.set_ylim(5e-7, 50)
    ax.set_yscale('log')
    corr_sum_scaled_distances = np.corrcoef(sum_scaled_distances,
                                            w_offdiagonal)[0, 1]
    ax.set_title(f'corr. coeff.: {corr_sum_scaled_distances:.5g}')

    fig.suptitle(f'{model_name}, random seed: {seed}')
    fig.savefig(results_dir + f'corr_recfs_w_{model_name}_seed{seed}.{format}',
                format=format)

    plt.close()

    return {
        'corr_com_distances': corr_com_distances,
        'corr_wv_cos_distances': corr_wv_cos_distances,
        'corr_wv_norm_distances': corr_wv_norm_distances,
        'corr_sum_scaled_distances': corr_sum_scaled_distances,
    }
