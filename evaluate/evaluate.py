import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from hvae_backbone.utils import save_to_file
from hvae_backbone.analysis import generate_white_noise_analysis, generate_latent_step_analysis
from hvae_backbone import utils, init_globals
from config import get_hparams
from data.mnist import MNIST, FashionMNIST, ChestMNIST
from scripts.test import main as test_main


def contrast_correlation(model, dataloader, logger: logging.Logger, use_mean=True):
    '''
        Computes and saves the image contrasts and posterior over the data.

        Args:
            model: trained model
            dataloader: dataloader of the data
            logger: logger object
            use_mean: if True, uses the mean of the posterior, else uses the sample
    '''
    logger.info('Computing contrast correlation')
    model_name = params.model_params.model_name
    model_type = params.model_params.model_type
    dataset_size = dataloader.dataset.__len__()
        
    contrast = torch.zeros(dataset_size)
    # replace 1800 with the number of latent dimensions
    posterior_mean = torch.zeros(dataset_size, 1800)
    posterior_std = torch.zeros(dataset_size, 1800)
    posterior_sample = torch.zeros(dataset_size, 1800)

    for i, batch in enumerate(dataloader):
        start_idx = i*dataloader.batch_size
        end_idx = min((i+1)*dataloader.batch_size, dataset_size)

        contrast[start_idx:end_idx] = torch.std(batch, dim=(1,2,3), keepdim=False)
        computed, _ = model(batch, use_mean=use_mean)
        posterior_mean[start_idx:end_idx] = computed['z_posterior_mean']
        posterior_std[start_idx:end_idx] = computed['z_posterior_std']
        if model_type == 'eavae':
            posterior_sample[start_idx:end_idx] = computed['z_posterior_sample']

        if i % 10 == 0:
            torch.cuda.empty_cache()

    latent_size = posterior_mean.shape[1]
    correlations = [0 for i in range(latent_size)]

    for i in range(latent_size):
        current_data = torch.stack((contrast, posterior_mean[:, i]))
        corr = torch.corrcoef(current_data)
        correlations[i] = corr[0, 1]

    save_to_file(contrast, model_name, 'contrast')
    save_to_file(posterior_mean, model_name, 'posterior_mean')
    save_to_file(posterior_std, model_name, 'posterior_std')

    if model_type == 'eavae':
        save_to_file(posterior_sample, model_name, 'posterior_sample')

    logger.info(f'Contrast correlation successful')

    return contrast, posterior_mean, posterior_std


def get_dn_data(model, dataloader, logger: logging.Logger, use_mean=True):
    '''
        Computes and saves data for divisive normalization: 
        - receptive fields
        - projective fields
        - posterior moments
        - natural images

        Args:
            model: trained model
            dataloader: dataloader of the data
            logger: logger object
            use_mean: if True, uses the mean of the posterior, else uses the sample
    '''
    logger.info('Computing data for divisive normalization')
    model_name = params.model_params.model_name
    model_type = params.model_params.model_type
    imsize = params.data_params.shape[1]
    sample = next(iter(dataloader))
    shape = params.data_params.shape
    dataset_size = dataloader.dataset.__len__()

    for target_block, config in params.analysis_params.white_noise_analysis.items():
        n_samples = config['n_samples']
        sigma = config['sigma']
        receptive_fields = \
            generate_white_noise_analysis(model, target_block, shape, params, n_samples, sigma)
        save_to_file(receptive_fields, model_name, 'receptive_fields', subdir='DN/')

    for target_block, config in params.analysis_params.latent_step_analysis.items():
        projective_fields = generate_latent_step_analysis(model, sample, target_block, params, **config)
        save_to_file(projective_fields, model_name, 'projective_fields', subdir='DN/')
        save_to_file(projective_fields, model_name, 'projective_fields')

    contrast = torch.zeros(dataset_size)
    # replace 1800 with the number of latent dimensions
    posterior_mean = torch.zeros(dataset_size, 1800)
    natural_ims = torch.zeros(dataset_size, 1, imsize, imsize)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            start_idx = i*dataloader.batch_size
            end_idx = min((i+1)*dataloader.batch_size, dataset_size)

            contrast[start_idx:end_idx] = torch.std(batch, dim=(1,2,3), keepdim=False)
            natural_ims[start_idx:end_idx] = batch
            computed, _ = model(batch, use_mean=use_mean)
            posterior_mean[start_idx:end_idx] = computed['z_posterior_mean']

            if i % 10 == 0:
                torch.cuda.empty_cache()

    if model_type == 'eavae':
        z_posterior_mean = posterior_mean[:, :-1]
        c_posterior_mean = posterior_mean[:, -1]
        save_to_file(z_posterior_mean, model_name, 'z_posterior_natural', subdir='DN/')
        save_to_file(c_posterior_mean, model_name, 'c_posterior_natural', subdir='DN/')
    else:
        save_to_file(posterior_mean, model_name, 'posterior_natural', subdir='DN/')
    save_to_file(natural_ims, model_name, 'natural_ims', subdir='DN/')
    save_to_file(contrast, model_name, 'contrasts', subdir='DN/')
    
    logger.info(f'DN data saving successful')


def get_posterior_moments(latent_zmu, latent_zvar, contrast_values, logger: logging.Logger):
    '''
        Computes and saves binned posterior moments for visualization.

        Args:
            latent_zmu: posterior means
            latent_zvar: posterior stds
            contrast_values: contrast values of the images
            logger: logger object
    '''
    logger.info('Computing posterior moments')
    model_name = params.model_params.model_name

    latent_zmu = latent_zmu.detach().cpu().numpy()
    latent_zvar = latent_zvar.pow(2).detach().cpu().numpy()
    contrast_values = contrast_values.detach().cpu().numpy()
    
    signal_mean = np.linalg.norm(latent_zmu, axis=1)
    noise_var = latent_zvar.mean(axis=1)
    
    contrast_rounds = np.linspace(np.min(contrast_values), np.max(contrast_values), 101)
    
    rounded_contrast = contrast_rounds[np.argmin(abs(np.subtract.outer(contrast_values, contrast_rounds)), axis=1)]
    
    binned_signal_mean = np.zeros_like(contrast_rounds)
    binned_signal_var = np.zeros_like(contrast_rounds)
    binned_noise_var = np.zeros_like(contrast_rounds)
    for k in range(len(contrast_rounds)):
        binned_signal_mean[k] = signal_mean[rounded_contrast == contrast_rounds[k]].mean()
        binned_signal_var[k] = latent_zmu[rounded_contrast == contrast_rounds[k]].var(axis=0).mean()
        binned_noise_var[k] = noise_var[rounded_contrast == contrast_rounds[k]].mean()
    
    binned_signal_mean_error = np.zeros_like(contrast_rounds)
    binned_signal_var_error = np.zeros_like(contrast_rounds)
    binned_noise_var_error = np.zeros_like(contrast_rounds)
    M = np.shape(latent_zmu)[1]
    N = np.array([sum(rounded_contrast == contrast_rounds[k]) for k in range(len(contrast_rounds))])
    for k in range(len(contrast_rounds)):
        binned_signal_mean_error[k] = signal_mean[rounded_contrast == contrast_rounds[k]].std() / np.sqrt(N[k])
        binned_signal_var_error[k] = binned_signal_var[k] * np.sqrt(2 / (N[k] - 1)) / np.sqrt(M)
        binned_noise_var_error[k] = latent_zvar[rounded_contrast == contrast_rounds[k]].std() / np.sqrt(M * N[k])

    save_to_file(binned_signal_mean, model_name, 'binned_signal_mean')
    save_to_file(binned_signal_var, model_name, 'binned_signal_var')
    save_to_file(binned_noise_var, model_name, 'binned_noise_var')
    save_to_file(binned_signal_mean_error, model_name, 'binned_signal_mean_error')
    save_to_file(binned_signal_var_error, model_name, 'binned_signal_var_error')
    save_to_file(binned_noise_var_error, model_name, 'binned_noise_var_error')
    save_to_file(contrast_rounds, model_name, 'contrast_rounds')
    
    logger.info(f'Posterior moments saving successful')


def out_of_distribution(model, ood_data_dict, logger: logging.Logger, use_mean=True):
    '''
        Computes and saves the posterior moments for out of distribution data.

        Args:
            model: trained model
            ood_data_dict: dictionary containing out of distribution data
            logger: logger object
            use_mean: if True, uses the mean of the posterior, else uses the sample
    '''
    logger.info('Computing out of distribution')
    model_name = params.model_params.model_name
    model_type = params.model_params.model_type
    
    for ood_name, ood_dataloader in ood_data_dict.items():
        with memory_manager():
            # replace 1800 with the number of latent dimensions
            posterior_mean_ood = torch.zeros(ood_dataloader.dataset.__len__(), 1800)
            posterior_std_ood = torch.zeros(ood_dataloader.dataset.__len__(), 1800)
            if model_type == 'eavae':
                posterior_sample_ood = torch.zeros(ood_dataloader.dataset.__len__(), 1800)

            for i, ood_data in enumerate(ood_dataloader):
                start_idx = i*ood_dataloader.batch_size
                end_idx = min((i+1)*ood_dataloader.batch_size, ood_dataloader.dataset.__len__())

                computed_ood, _ = model(ood_data, use_mean=use_mean)
                posterior_mean_ood[start_idx:end_idx] = computed_ood['z_posterior_mean']
                posterior_std_ood[start_idx:end_idx] = computed_ood['z_posterior_std']
                if model_type == 'eavae':
                    posterior_sample_ood[start_idx:end_idx] = computed_ood['z_posterior_sample']

            save_to_file(posterior_mean_ood, model_name, f'posterior_mean_{ood_name}')
            save_to_file(posterior_std_ood, model_name, f'posterior_std_{ood_name}')
            if model_type == 'eavae':
                save_to_file(posterior_sample_ood, model_name, f'posterior_sample_{ood_name}')

    logger.info(f'Out of distribution data saving successful')


from contextlib import contextmanager
import gc

@contextmanager
def memory_manager():
    try:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        yield
    finally:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == '__main__':

    hparams = get_hparams()
    params = init_globals(hparams)

    checkpoint, save_path = utils.load_experiment_for('test', log_params=params.log_params)
    logger = utils.setup_logger(save_path)
    assert checkpoint is not None

    device = params.model_params.device
    
    with torch.no_grad():

        model = checkpoint.get_model().to(device)

        model.eval()
        logger.info(f'Model Checkpoint is loaded from {params.log_params.load_from_eval}')

        model.summary()
        dataset_object = params.data_params.dataset(**params.data_params.params)
        dataloader = dataset_object.get_test_loader(params.analysis_params.batch_size)
        imsize = params.data_params.shape[1]

        test_data = dataset_object.test_set.data
        test_data = torch.tensor(test_data, dtype=torch.float32, device=device)

        # Get model output (image reconstructions) for testset

        test_main()

        # Contrasts and posterior moments

        with memory_manager():
            contrast, post_mean, post_std = contrast_correlation(model, dataloader, logger)

        # Divisive normalization data

        with memory_manager():
            get_dn_data(model, dataloader, logger)

        # Binned posterior moments

        std_of_means = torch.std(post_mean, dim=0)
        active_dims = np.where(np.array(std_of_means) > 0.5, True, False)
        if params.model_params.model_type == 'eavae':
            active_dims[-1] = False

        post_mean_active = post_mean[:, active_dims==True]
        post_std_active = post_std[:, active_dims==True]

        with memory_manager():
            get_posterior_moments(post_mean_active, post_std_active, contrast, logger)

        # Out of distribution data

        natural_test_shuffled = test_data.clone().reshape(-1, imsize, imsize)
        for i in range(imsize):
            for j in range(imsize):
                torch.manual_seed(imsize*i+j)
                natural_test_shuffled[:, i, j] = natural_test_shuffled[:, i, j][torch.randperm(natural_test_shuffled.shape[0])]
        natural_test_shuffled = natural_test_shuffled.reshape(-1, imsize*imsize)
        natural_shuffled_loader = DataLoader(natural_test_shuffled, 
                                             batch_size=params.analysis_params.batch_size, 
                                             shuffle=False)

        mnist = MNIST(z_score=True, resize=(imsize,imsize))
        fashion = FashionMNIST(z_score=True, resize=(imsize,imsize))
        chest = ChestMNIST(z_score=True, resize=(imsize,imsize))

        mnist_loader = mnist.get_train_loader(params.analysis_params.batch_size)
        fashion_loader = fashion.get_train_loader(params.analysis_params.batch_size)
        chest_loader = chest.get_train_loader(params.analysis_params.batch_size)

        ood_data_dict = {'shuffled_natural': natural_shuffled_loader,
                        'mnist': mnist_loader,
                        'fashion': fashion_loader,
                        'chest': chest_loader}

        with memory_manager():
            out_of_distribution(model, ood_data_dict, logger)

    logger.info('Evaluation finished')