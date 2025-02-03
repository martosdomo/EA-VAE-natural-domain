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
from env import ROOT_DIR as root

results_dir = os.path.join(root, 'divisivenorm/results/')

# vanilla and eavae model names
standardvae_modelname = 'StandardVAE'
eavae_modelname = 'EAVAE_lognormal'

config = {
    'image_sample_fn': results_dir + 'image_samples40.png',
    f'recf_{standardvae_modelname}_sample_fn': results_dir + f'recf_{standardvae_modelname}_samples.png',
    f'recf_{eavae_modelname}_sample_fn': (results_dir +
                                     f'recf_{eavae_modelname}_samples.png'),
    f'recf_{standardvae_modelname}_active_fn_pre': results_dir + f'recf_{standardvae_modelname}_',
    f'recf_{eavae_modelname}_active_fn_pre': results_dir + f'recf_{eavae_modelname}_',

    'three_violins_plot_fn_pre': results_dir + 'three_violins_',
    'three_violins_std_hists_plot_fn_pre': (results_dir
                                            + 'three_violins_std_hists_'),
    'butterflies_plot_fn_pre': results_dir + 'butterflies_',

    'three_violins_stds_fn': (results_dir + f'three_violins_stds_{standardvae_modelname}_{eavae_modelname}.pkl'),

    'active_recfs_threshold': 0.3,

    'three_violins_percentiles': (25, 75),
    'three_violins_filter_idcs': range(10),
    'three_violins_std_hists_filter_idcs': range(100),

    'bowties_n_bins': 20,

    'img_indices_from': 0,
    'img_indices_to': 64000,

    'standardvae_modelname': standardvae_modelname,
    'eavae_modelname': eavae_modelname,

    'standardvae_ims_fn': f'/{standardvae_modelname}_natural_ims.pt',
    'standardvae_recfs_fn': f'/{standardvae_modelname}_receptive_fields.pt',
    'standardvae_postmns_fn': f'/{standardvae_modelname}_posterior_natural.pt',

    'eavae_ims_fn': f'/{eavae_modelname}_natural_ims.pt',
    'eavae_recfs_fn': f'/{eavae_modelname}_receptive_fields.pt',
    'eavae_postmns_fn': f'/{eavae_modelname}_z_posterior_natural.pt',

    'smvae_data_dir': os.path.join(root, 'eval_data/DN/'),

    'vminmax_imgs': 3.0,
    'vminmax_recfs': 6.0,
}