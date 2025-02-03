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
import pickle

from config import config, results_dir
import plots


if __name__ == '__main__':
    seeds = config['random_seeds']

    ws_per_seed = []
    ksps_per_seed = []

    for seed in seeds:
        w_standardvae_untrained = pickle.load(
            open(results_dir +
                 f'w_standardvae_untrained_{config["standardvae_modelname"]}_'
                 f'seed{seed}.pkl', 'rb'))
        w_standardvae = pickle.load(
            open(results_dir +
                 f'w_standardvae_{config["standardvae_modelname"]}_'
                 f'seed{seed}.pkl', 'rb'))
        w_eavae_untrained = pickle.load(
            open(results_dir +
                 f'w_eavae_untrained_'
                 f'{config["eavae_modelname"]}_'
                 f'seed{seed}.pkl', 'rb'))
        w_eavae = pickle.load(
            open(results_dir +
                 f'w_eavae_{config["eavae_modelname"]}_'
                 f'seed{seed}.pkl', 'rb'))
        ws = {
            'w_standardvae_untrained': w_standardvae_untrained,
            'w_standardvae': w_standardvae,
            'w_eavae_untrained': w_eavae_untrained,
            'w_eavae': w_eavae,
        }
        ws_per_seed.append(ws)

        ksps = plots.plot_w_histograms(w_standardvae_untrained, w_standardvae,
                                       w_eavae_untrained, w_eavae,
                                       seed, format='pdf')
        ksps_per_seed.append(ksps)

    plots.plot_w_histograms_cumulative(ws_per_seed, ksps_per_seed, seeds,
                                       format='pdf')
