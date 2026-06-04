def _model():
    from hvae_backbone.block import InputBlock, ContrastiveOutputBlock, ContrastiveGenBlock
    from hvae_backbone.hvae import hVAE as hvae
    from hvae_backbone.elements.layers import Flatten, Unflatten
    from hvae_backbone.utils import OrderedModuleDict

    _blocks = OrderedModuleDict(
        x=InputBlock(
            net=Flatten(start_dim=1) #0: batch-flatten, 1: sample-flatten
        ),
        z=ContrastiveGenBlock(
            prior_net=None,
            posterior_net=x_to_z_net,
            input_id="z_prior",
            condition="x",
            output_distribution="laplace",
            contrast_distribution='softlaplace',
        ),
        x_hat=ContrastiveOutputBlock(
            input_id="z",
            contrast_dims=1,
            net=[z_to_x_net, Unflatten(1, data_params.shape)],
            output_distribution="normal",
            stddev=0.4
        ),
    )

    prior_shape = (1800, )

    _prior = dict(
        z_prior=torch.nn.Parameter(torch.zeros(2*prior_shape[0]),
                                   requires_grad=False)
    )

    __model = hvae(
        blocks=_blocks,
        init=_prior
    )

    return __model



# --------------------------------------------------
# HYPERPARAMETERS
# --------------------------------------------------
from hvae_backbone import Hyperparams


"""
--------------------
LOGGING HYPERPARAMETERS
--------------------
"""
# Trained EAVAE model from the manuscript
# with softplus(laplace) prior for the scaling variable
import os
from env import ROOT_DIR as root
softlaplace_manuscript = os.path.join(root, 'experiments/EAVAE_softlaplace/manuscript/checkpoints/checkpoint_5000.pth')

# Your trained models
# ...

log_params = Hyperparams(
    name='EAVAE_softlaplace',

    # TRAIN LOG
    # --------------------
    # Defines how often to save a model checkpoint and logs to disk.

    save_checkpoints_locally=True,
    checkpoint_interval_in_epochs=500,
    eval_interval_in_epochs=1,

    load_from_train=None,  # resume checkpoint from local path
    load_from_eval=softlaplace_manuscript,
)

"""
--------------------
MODEL HYPERPARAMETERS
--------------------
"""

model_params = Hyperparams(
    model=_model,
    device='cuda',
    seed=0,

    # Latent layer distribution base can be in ('std', 'logstd').
    # Determines if the model should predict
    # std (with softplus) or logstd (std is computed with exp(logstd)).
    distribution_base='std',
    distribution_sigma_param="std",

    # Latent layer Gradient smoothing beta. ln(2) ~= 0.6931472.
    # Setting this parameter to 1. disables gradient smoothing (not recommended)
    gradient_smoothing_beta=0.6931472,
    model_name='EAVAE_softlaplace',
    model_type='eavae',
)
    
"""
--------------------
DATA HYPERPARAMETERS
--------------------
"""
from data.textures.textures import TexturesDataset

TRAINING_SIZE = 512000
data_params = Hyperparams(
    dataset=TexturesDataset,
    params=dict(type='natural', image_size=40, whitening='old'),

    # Image metadata
    shape=(1, 40, 40),
    type='natural',
)
"""
--------------------
TRAINING HYPERPARAMETERS
--------------------
"""
import math
batch_size = 512
NUM_EPOCHS = 5000
train_params = Hyperparams(
    # The total number of training updates
    total_train_epochs=NUM_EPOCHS+1,
    # training batch size
    batch_size=batch_size,
    steps_per_epoch=math.ceil(TRAINING_SIZE / batch_size),
    # Freeze specific layers
    unfreeze_first=False,
    freeze_nets=[],
)

"""
--------------------
OPTIMIZER HYPERPARAMETERS
--------------------
"""
optimizer_params = Hyperparams(
    # Optimizer can be one of ('Radam', 'Adam', 'Adamax').
    # Adam and Radam should be avoided on datasets when the global norm is large!!
    type='Adam',
    # Learning rate decay scheme
    # can be one of ('cosine', 'constant', 'exponential', 'noam')
    learning_rate_scheme='constant',

    # Defines the initial learning rate value
    learning_rate=3e-5,

    # Adam/Radam/Adamax parameters
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    # L2 weight decay
    l2_weight=1e-6,

    # noam/constant/cosine warmup
    warmup_steps=100.,
    # exponential or cosine
    #   Defines the number of steps over which the learning rate
    #   decays to the minimum value (after decay_start)
    decay_steps=35 * train_params.steps_per_epoch,
    #   Defines the update at which the learning rate starts to decay
    decay_start=40 * train_params.steps_per_epoch,
    #   Defines the minimum learning rate value
    min_learning_rate=3e-5,#1e-4,
    # exponential only
    #   Defines the decay rate of the exponential learning rate decay
    decay_rate=0.5,


    # Gradient
    #  clip_norm value should be defined for nats/dim loss.
    clip_gradient_norm=False,
    gradient_clip_norm_value=300.,

    # Whether to use gradient skipping.
    gradient_skip=True,
    # Defines the threshold at which to skip the update.
    # Also defined for nats/dim loss.
    gradient_skip_threshold=1e10
)

"""
--------------------
LOSS HYPERPARAMETERS
--------------------
"""
loss_params = Hyperparams(
    reconstruction_loss="default",
    kldiv_loss="default",
    custom_loss=None,

    # ELBO beta warmup (from NVAE).
    # schedule can be in ('None', 'Logistic', 'Linear')
    variation_schedule='Linear',

    # linear beta schedule
    vae_beta_anneal_start=100 * train_params.steps_per_epoch,
    vae_beta_anneal_steps=100 * train_params.steps_per_epoch,
    vae_beta_min=1,             # latent z starting beta
    contrast_beta_start=10.0,   # latent s starting beta

    # logistic beta schedule
    vae_beta_activation_steps=10000,
    vae_beta_growth_rate=1e-5,
)

"""
--------------------
EVALUATION HYPERPARAMETERS
--------------------
"""
eval_params = Hyperparams(
    # Defines how many validation samples to validate on every time we're going to write to tensorboard
    # Reduce this number of faster validation. Very small subsets can be non descriptive of the overall distribution
    n_samples_for_validation=12800,
    n_samples_for_reconstruction=4,

    # validation batch size
    batch_size=512,

    use_mean=False,
)

"""
--------------------
SYNTHESIS HYPERPARAMETERS
--------------------
"""
analysis_params = Hyperparams(
    batch_size=128,

    white_noise_analysis=dict(
        z=dict(
            n_samples=1000,
            sigma=0.1,
        )
    ),

    latent_step_analysis=dict(
        z=dict(
            diff=1,
            value=1,
        )
    ),
)

"""
--------------------
CUSTOM BLOCK HYPERPARAMETERS
--------------------
"""
import torch
# add your custom block hyperparameters here
x_size = 40*40
z_size = 1800

x_to_z_net = Hyperparams(
    type='mlp',
    input_size=x_size,
    hidden_sizes=[2000, 2000],
    output_size=2*z_size,
    activation=torch.nn.Softplus(),
    residual=False,
    activate_output=False,
    output_activation=None,
)

z_to_x_net = Hyperparams(
    type='mlp',
    input_size=z_size-1,
    hidden_sizes=[],
    output_size=x_size,
    activation=None,
    residual=False,
)


