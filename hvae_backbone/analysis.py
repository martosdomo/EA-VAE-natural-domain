import numpy as np
import torch

#from . import params

def generate_white_noise_analysis(model, target_block, shape, params, n_samples=100, sigma=0.6):
    import scipy

    white_noise = np.random.normal(size=(n_samples, np.prod(shape)),
                                   loc=0.0, scale=1.).astype(np.float32)

    # apply ndimage.gaussian_filter with sigma=0.6
    for i in range(n_samples):
        white_noise[i, :] = scipy.ndimage.gaussian_filter(
            white_noise[i, :].reshape(shape), sigma=sigma).reshape(np.prod(shape))

    with torch.no_grad():
        model.eval()
        computed, _ = model(torch.ones(1, *shape, device=params.device), stop_at=target_block)
        target_block_dim = computed[target_block].shape[1:]
        target_block_values = torch.zeros((n_samples, *target_block_dim), device=params.device)

        # loop over a batch of 128 white_noise images
        batch_size = params.analysis_params.batch_size
        for i in range(0, n_samples, batch_size):
            batch = white_noise[i:i+batch_size, :].reshape(-1, *shape)
            computed_target, _ = model(torch.tensor(batch, device=params.device),
                                       use_mean=True, stop_at=target_block)
            target_block_values[i:i+batch_size] = computed_target[target_block]

        target_block_values = torch.flatten(target_block_values, start_dim=1)
        # multiply transpose of target block_values with white noise tensorially
        receptive_fields = np.matmul(
            target_block_values.T.cpu().numpy(), white_noise
        ) / np.sqrt(n_samples)
        return receptive_fields

def generate_latent_step_analysis(model, sample, target_block, params, diff=1, value=1):
    def copy_computed(computed):
        return {k: v.clone() for k, v in computed.items()}

    with torch.no_grad():
        model.eval()
        target_computed, _ = model(sample.to(params.device), use_mean=True, stop_at=target_block)
        input_0 = target_computed[target_block]
        shape = input_0.shape
        n_dims = np.prod(shape[1:])

        computed_checkpoint = copy_computed(target_computed)
        output_computed, _ = model(computed_checkpoint, use_mean=True)
        output_0 = torch.mean(output_computed['output'], dim=0)

        visualizations = []
        for i in range(n_dims):
            input_i = torch.zeros([1, n_dims], device=params.device)
            input_i[0, i] = value
            input_i = input_i.reshape(shape[1:]).to(params.device)
            input_i = input_0 + input_i
            target_computed[target_block] = input_i

            computed_checkpoint = copy_computed(target_computed)
            trav_output_computed, _ = model(computed_checkpoint, use_mean=True)
            output_i = torch.mean(trav_output_computed['output'], dim=0)

            latent_step_vis = output_i - diff * output_0
            visualizations.append(latent_step_vis.detach().cpu().numpy())

        return visualizations