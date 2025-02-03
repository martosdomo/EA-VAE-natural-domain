import logging
import time

import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader

from .checkpoint import Checkpoint
from .utils import prepare_for_log, print_line, image_log_rule, save_to_file

from . import params


def compute_loss(targets: tensor, distributions: dict, computed: dict = None, step_n: int = 0, with_stats=False) -> dict:
    """
    Compute loss for VAE (custom or default)
    based on Efficient-VDVAE paper

    :param targets: tensor, the target images
    :param distributions: list, the distributions for each generator block
    :param computed: tensor, the logits for the output block
    :param step_n: int, the current step number
    :return: dict, containing the loss values

    """
    # Use custom loss function if provided
    if params.loss_params.custom_loss is not None:
        return params.free_loss(targets=targets, distributions=distributions, computed=computed, step_n=step_n) 

    output_distribution = distributions['output']
    if with_stats:
        feature_matching_loss, feature_matching_stats = params.reconstruction_loss(targets, output_distribution, with_stats=True)
    else:
        feature_matching_loss = params.reconstruction_loss(targets, output_distribution)

    global_variational_prior_losses = []
    kl_divs, kl_stats = dict(), dict()
    for block_name, dists in distributions.items():
        if block_name == 'output' or dists is None or dists[1] is None: #or len(dists) != 3
            continue
        if len(dists) == 2:
            dists = (dists[0], dists[1], "default")
        prior, posterior, kl_type = dists
        
        if kl_type == "default":
            kl_fun = params.kl_divergence
        else:
            from .elements import losses
            kl_fun = losses.get_kl_loss(kl_type)
        if with_stats:
            block_kl, stats = kl_fun(prior, posterior, with_stats=True)
            kl_stats.update({f"{block_name}_{k}": v for k, v in stats.items()})
        else:
            block_kl = kl_fun(prior, posterior)
        global_variational_prior_losses.append(block_kl)
        kl_divs.update({f"{block_name}_kl": block_kl})
       

    global_variational_prior_losses = torch.stack(global_variational_prior_losses)
    kl_div = torch.sum(global_variational_prior_losses)  # / np.log(2.)
    #global_variational_prior_loss = kl_div \
    global_variational_prior_loss = global_variational_prior_losses
    global_var_loss = params.kldiv_schedule(step_n) * global_variational_prior_loss  # beta
    global_var_loss = torch.sum(global_var_loss).to(targets.device)
    loss = -feature_matching_loss + global_var_loss
    elbo = feature_matching_loss - kl_div
    
    kl_divs_processed = dict()
    kl_stats_processed = dict()
    for kl_name, kl_value in kl_divs.items():
        block_name = kl_name.strip('_')
        if block_name not in kl_divs_processed:
            kl_divs_processed[block_name] = kl_value
        else:
            kl_divs_processed[block_name] += kl_value
            
    for kl_stat_name, kl_stat_value in kl_stats.items():
        stat_name_with_block = kl_stat_name.strip('_')
        if stat_name_with_block not in kl_stats_processed:
            kl_stats_processed[stat_name_with_block] = kl_stat_value
        else:
            kl_stats_processed[stat_name_with_block] += kl_stat_value
            
    loss_dict = dict(
        loss=loss,
        elbo=elbo,
        reconstruction_loss=-feature_matching_loss,
        kl_div=kl_div)#,
        #**kl_divs_processed,
    #)
    if with_stats:
        loss_dict.update(kl_stats_processed)
        loss_dict.update(feature_matching_stats)
    
    return loss_dict


def gradient_norm(net):
    """
    Compute the global norm of the gradients of the network parameters
    based on Efficient-VDVAE paper
    :param net: hVAE, the network
    """
    parameters = [p for p in net.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = torch.tensor(0.0)
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
    return total_norm


def gradient_clip(net):
    """
    Clip the gradients of the network parameters
    based on Efficient-VDVAE paper
    """
    if params.optimizer_params.clip_gradient_norm:
        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                    max_norm=params.optimizer_params.gradient_clip_norm_value)
    else:
        total_norm = gradient_norm(net)
    return total_norm


def gradient_skip(global_norm):
    """
    Skip the gradient update if the global norm is too high
    based on Efficient-VDVAE paper
    :param global_norm: tensor, the global norm of the gradients
    """
    if params.optimizer_params.gradient_skip:
        if torch.any(torch.isnan(global_norm)) or global_norm >= params.optimizer_params.gradient_skip_threshold:
            skip = True
            gradient_skip_counter_delta = 1.
        else:
            skip = False
            gradient_skip_counter_delta = 0.
    else:
        skip = False
        gradient_skip_counter_delta = 0.
    return skip, gradient_skip_counter_delta


def reconstruction_step(net, inputs: tensor, step_n=None, use_mean=False, with_stats=False):
    """
    Perform a reconstruction with the given network and inputs
    based on Efficient-VDVAE paper

    :param net: hVAE, the network
    :param inputs: tensor, the input images
    :param step_n: int, the current step number
    :param use_mean: use the mean of the distributions instead of sampling
    :return: tensor, tensor, dict, the output images, the computed features, the loss values
    """
    net.eval()
    with torch.no_grad():
        computed, distributions = net(inputs, use_mean=use_mean)
        if step_n is None:
            step_n = params.loss_params.vae_beta_anneal_steps * 10.
        results = compute_loss(inputs, distributions, computed, step_n=step_n, with_stats=with_stats)
        return computed, distributions, results


def reconstruct(net, dataset: DataLoader,
                use_mean=False, global_step=None, logger: logging.Logger = None, log_images=False,
                offline_test=False):
    """
    Reconstruct the images from the given dataset
    based on Efficient-VDVAE paper

    :param net: hVAE, the network
    :param dataset: DataLoader, the dataset
    :param use_mean: use the mean of the distributions instead of sampling
    :param global_step: int, the current step number
    :param logger: logging.Logger, the logger
    :return: list, the input/output pairs
    """

    # Evaluation
    n_samples_for_eval = params.eval_params.n_samples_for_validation
    results, (original, output_samples, output_means) = \
        evaluate(net, dataset, n_samples=n_samples_for_eval, use_mean=use_mean, global_step=global_step, logger=logger)
    
    if offline_test:
        model_name = params.model_params.model_name
        save_to_file((original, output_samples, output_means), model_name, 'reconstruction')
        
    return results


def train_step(net, optimizer, schedule, inputs, step_n, with_stats=False):
    """
    Perform a training step with the given network and inputs
    based on Efficient-VDVAE paper

    :param net: hVAE, the network
    :param optimizer: torch.optim.Optimizer, the optimizer
    :param schedule: torch.optim.lr_scheduler.LRScheduler, the scheduler
    :param inputs: tensor, the input images
    :param step_n: int, the current step number
    :return: tensor, dict, tensor, the output images, the loss values, the global norm of the gradients
    """
    computed, distributions = net(inputs)
    output_sample = computed['output']
    results = compute_loss(inputs, distributions, computed, step_n=step_n, with_stats=with_stats)

    loss = results["loss"]
    loss.backward()

    global_norm = gradient_clip(net)
    skip, gradient_skip_counter_delta = gradient_skip(global_norm)

    if not skip:
        optimizer.step()
        schedule.step()
        
    # calculate momentum from momentum scheduler
    # update ema if available
    if params.loss_params.custom_loss is not None and isinstance(params.loss_params.custom_loss, list):
        momentum = params.ema_momentum_schedule(step_n)
        net.update_ema(momentum)

    optimizer.zero_grad()
    return output_sample, results, global_norm, gradient_skip_counter_delta


def train(net,
          optimizer, schedule,
          train_loader: DataLoader, val_loader: DataLoader,
          start_epoch: int,
          checkpoint_path: str, logger: logging.Logger) -> None:
    """
    Train the network
    based on Efficient-VDVAE paper

    :param net: hVAE, the network
    :param optimizer: torch.optim.Optimizer, the optimizer
    :param schedule: torch.optim.lr_scheduler.LRScheduler, the scheduler
    :param train_loader: DataLoader, the training dataset
    :param val_loader: DataLoader, the validation dataset
    :param start_step: int, the step number to start from
    :param checkpoint_path: str, the path to save the checkpoints to
    :param logger: logging.Logger, the logger
    :return: None
    """
    global_step = start_epoch * len(train_loader)
    gradient_skip_counter = 0.
    total_train_epochs = params.train_params.total_train_epochs

    for epoch in range(start_epoch, total_train_epochs):
        net.train()
        train_stats = None
        epoch_time = 0
        for batch_n, train_inputs in enumerate(train_loader):
            global_step += 1
            train_inputs = train_inputs.to(params.device, non_blocking=True)
            with_stats = batch_n == len(train_loader) - 1
            start_time = time.time()
            train_outputs, train_results, global_norm, gradient_skip_counter_delta = \
                train_step(net, optimizer, schedule, train_inputs, global_step, with_stats=with_stats)
            end_time = round((time.time() - start_time), 2)
            gradient_skip_counter += gradient_skip_counter_delta

            epoch_time += end_time
            #print(train_results["loss"].item())
            train_stats = train_results if train_stats is None else \
                {k: v + train_results[k] for k, v in train_stats.items()}
                
        train_results = {k: v / (batch_n + 1) for k, v in train_stats.items()}
        train_results["epoch"] = epoch
        train_results = prepare_for_log(train_results)
        logger.info((epoch,
                    ('Time/Epoch (sec)', round(epoch_time, 2)),
                    ('LOSS', round(train_results["loss"], 4)),
                    ('Reconstruction Loss', round(train_results["reconstruction_loss"], 3)),
                    ('KL loss', round(train_results["kl_div"], 3))))

        """
        EVALUATION AND CHECKPOINTING
        """
        net.eval()
        first_epoch = epoch == start_epoch
        eval_time = epoch % params.log_params.eval_interval_in_epochs == 0
        checkpoint_time = epoch % params.log_params.checkpoint_interval_in_epochs == 0
        if eval_time or checkpoint_time:
            print_line(logger, newline_after=False)

        if eval_time or first_epoch:
            train_ssim = params.ssim_metric(train_inputs, train_outputs,
                                            global_batch_size=params.train_params.batch_size)
            logger.info(
                f'Train Stats | '
                f'LOSS {train_results["loss"]:.4f} | '
                f'ELBO {train_results["elbo"]:.4f} | '
                f'Reconstruction Loss {train_results["reconstruction_loss"]:.4f} |'
                f'KL Div {train_results["kl_div"]:.4f} |'
                f'SSIM: {train_ssim:.4f}')
            val_results = reconstruct(net, val_loader,
                                        use_mean=params.eval_params.use_mean, 
                                        global_step=global_step, logger=logger, log_images=image_log_rule(epoch))
            val_results = prepare_for_log(val_results)

        experiment = Checkpoint(epoch, net, optimizer, schedule, params)
        if checkpoint_time:
            # Save new  checkpoint
            path = experiment.save(checkpoint_path,
                                   save_locally=params.log_params.save_checkpoints_locally)
            logger.info(f'Saved checkpoint for epoch {epoch} to {path}')
        else:
            # Update latest checkpoint
            path = experiment.save(checkpoint_path)

        if eval_time or checkpoint_time:
            print_line(logger, newline_after=True)

    logger.info(f'Finished training after {epoch} epochs!')
    return


def evaluate(net, val_loader: DataLoader, n_samples: int, global_step: int = None,
             use_mean=False, logger: logging.Logger = None) -> tuple:
    """
    Evaluate the network on the given dataset

    :param net: hVAE, the network
    :param val_loader: DataLoader, the dataset
    :param n_samples: number of samples to evaluate
    :param global_step: int, the current step number
    :param use_mean: use the mean of the distributions instead of sampling
    :param logger: logging.Logger, the logger
    :return: dict, tensor, tensor, the loss values, the output images, the input images
    """
    net.eval()

    val_step = 0
    global_results, original, output_samples, output_means = None, None, None, None
    #c_posterior_means, c_posterior_std = None, None
    for val_step, val_inputs in enumerate(val_loader):
        n_samples -= params.eval_params.batch_size
        val_inputs = val_inputs.to(params.device, non_blocking=True)
        val_computed, val_distributions, val_results = \
            reconstruction_step(net, inputs=val_inputs, step_n=global_step, use_mean=use_mean, with_stats=True)
        val_outputs = val_computed["output"]
        val_output_means = val_distributions['output'].mean
        val_results["ssim"] = params.ssim_metric(val_inputs, val_outputs,
                                                 params.eval_params.batch_size).detach().cpu()
        val_results["mean_ssim"] = params.ssim_metric(val_inputs, val_output_means,
                                                      params.eval_params.batch_size).detach().cpu()

        posterior_means = val_computed['z_posterior_mean']
        #posterior_std = computed['z_posterior_std']
        c_posterior_means = posterior_means[:, -1]
        #c_posterior_std = torch.cat((c_posterior_std, posterior_std[:, -1]))
        val_results["c_mean"] = torch.mean(c_posterior_means).detach().cpu()
        val_results["c_means_std"] = torch.std(c_posterior_means).detach().cpu()
        contrast = torch.std(val_inputs, dim=(1,2,3), keepdim=False)

        corr_stack = torch.stack((contrast, c_posterior_means))
        val_results["contrast_correlation"] = torch.corrcoef(corr_stack)[0, 1]

        val_inputs = val_inputs.detach().cpu()
        val_outputs = val_outputs.detach().cpu()
        val_output_means = val_output_means.detach().cpu()
        if global_results is None:
            global_results = val_results
            original = val_inputs
            output_samples = val_outputs
            output_means = val_output_means
        else:
            global_results = {k: v + val_results[k] for k, v in global_results.items()}
            original = torch.cat((original, val_inputs), dim=0)
            output_samples = torch.cat((output_samples, val_outputs), dim=0)
            output_means = torch.cat((output_means, val_output_means), dim=0)

        if n_samples <= 0:
            break

    global_results = {k: v / (val_step + 1) for k, v in global_results.items()}

    log = logger.info if logger is not None else print
    log(
        f'Validation Stats |'
        f' LOSS {global_results["loss"]:.4f} |'
        f' ELBO {global_results["elbo"]:.4f} |'
        f' Reconstruction Loss {global_results["reconstruction_loss"]:.4f} |'
        f' KL Div {global_results["kl_div"]:.4f} |'
        f' SSIM: {global_results["ssim"]:.4f} |'
        f' c_mean: {global_results["c_mean"]:.4f} |'
        f' c_means_std: {global_results["c_means_std"]:.4f} |'
        f' contrast_correlation: {global_results["contrast_correlation"]:.4f}')

    return global_results, (original, output_samples, output_means)


def model_summary(net):
    """
    Print the model summary
    :param net: nn.Module, the network
    :return: None
    """
    from torchinfo import summary
    shape = (1,) + params.data_params.shape
    return summary(net, input_size=shape, depth=8)