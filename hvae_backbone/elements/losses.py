import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.distribution import Distribution
from torch.distributions.kl import kl_divergence

from ..utils import softclip
from .. import params


def get_reconstruction_loss():
    """
    Get reconstruction loss based on hparams
    :return: nn.Module object for calculating reconstruction loss
    """

    if params.loss_params.reconstruction_loss == 'default':
        return LogProb(data_shape=params.data_params.shape)
    elif params.loss_params.reconstruction_loss == 'mse':
        return MSELoss(data_shape=params.data_params.shape)
    elif params.loss_params.reconstruction_loss == None:
        return None
    else:
        raise ValueError(f'Unknown reconstruction loss: {params.loss_params.reconstruction_loss}')


def get_kl_loss(kl_type=params.loss_params.kldiv_loss):
    """
    Get kl loss based on hparams
    :return: nn.Module object for calculating kl loss
    """
    kl_class = get_kl_class_by_name(kl_type)
    kl_loss = kl_class(data_shape=params.data_params.shape)
    return kl_loss


def get_kl_class_by_name(name):
    if name == 'default':
        return KLDivergence
    elif name == None:
        return None
    else:
        raise ValueError(f'Unknown kl loss: {params.loss_params.kldiv_loss}')


class LogProb(nn.Module):

    """
    Log probability loss
    based on original implementation of TDVAE
    with extra functionality from Efficient-VDVAE paper

    :param data_shape: shape of the data
    """
    def __init__(self, data_shape):
        super(LogProb, self).__init__()
        self.data_shape = data_shape

    def forward(self, targets, distribution: Distribution, *, with_stats=False):
        targets = targets.reshape(distribution.batch_shape)
        log_probs = distribution.log_prob(targets + 1e-8)
        log_p_x = torch.flatten(log_probs, start_dim=1)
        mean_axis = list(range(1, len(log_p_x.size())))

        # loss for batch
        per_example_loss = torch.sum(log_p_x, dim=mean_axis)  # B
        batch_size = per_example_loss.size()[0]
        scalar = batch_size * np.prod(self.data_shape)
        loss = torch.sum(per_example_loss) / (scalar * np.log(2)) # divide by ln(2) to convert to bit range
        
        if with_stats:
            mse = torch.nn.functional.mse_loss(targets, distribution.mean)
            return loss, dict(mse=mse)
        return loss


class MSELoss(nn.Module):
    """
    Log probability loss
    based on original implementation of TDVAE
    with extra functionality from Efficient-VDVAE paper

    :param data_shape: shape of the data
    """
    def __init__(self, data_shape):
        super(MSELoss, self).__init__()
        self.data_shape = data_shape

    def forward(self, targets, distribution: Distribution, *, with_stats=False):
        targets = targets.reshape(distribution.batch_shape)
        sample = distribution.rsample()
        mse = nn.functional.mse_loss(targets, sample, reduction='none')
        log_p_x = torch.flatten(mse, start_dim=1)
        mean_axis = list(range(1, len(log_p_x.size())))

        # loss for batch
        per_example_loss = torch.sum(log_p_x, dim=mean_axis)  # B
        batch_size = per_example_loss.size()[0]
        scalar = batch_size * np.prod(self.data_shape)
        loss = torch.sum(per_example_loss) / (scalar * np.log(2)) # divide by ln(2) to convert to bit range
        loss = -1*loss
        sigma = distribution.stddev[0,0,0,0].item()
        var = sigma**2
        loss = loss / (2*var)
        
        if with_stats:
            return loss, dict()
        return loss


class BernoulliLoss(nn.Module):

    """
    Bernoulli loss
    from Efficient-VDVAE paper
    """
    def __init__(self, data_shape):
        super(BernoulliLoss, self).__init__()
        self.data_shape = data_shape

    def forward(self, targets, logits):
        targets = targets[:, :, 2:30, 2:30]
        logits = logits[:, :, 2:30, 2:30]

        loss_value = Bernoulli(logits=logits)
        recon = loss_value.log_prob(targets)
        mean_axis = list(range(1, len(recon.size())))
        per_example_loss = - torch.sum(recon, dim=mean_axis)
        batch_size = per_example_loss.size()[0]
        scalar = batch_size * np.prod(self.data_shape)
        loss = torch.sum(per_example_loss) / scalar
        return loss
    
    
            
#def custom_kl_divergence(posterior, prior):
#    """
#    Custom kl divergence loss
#    """
#    #check if prior and posterior are Laplace distributions
#    if isinstance(prior, torch.distributions.laplace.Laplace) and isinstance(posterior, torch.distributions.laplace.Laplace):
#        scale_ratio = posterior.scale / prior.scale
#        loc_abs_diff = (posterior.loc - prior.loc).abs()
#        t1 = -scale_ratio.log()
#        t2 = loc_abs_diff / prior.scale
#        t3 = scale_ratio * torch.exp(-loc_abs_diff / posterior.scale)
#        return t1 + t2 + t3 -1
#    else:
#        #raise not implemeted error, print information about the distributions
#        raise NotImplementedError(f"custom KL divergence between {type(prior)} and {type(posterior)} not implemented")
    


class KLDivergence(nn.Module):
    """
    KL divergence loss
    from Efficient-VDVAE paper
    """

    def __init__(self, data_shape):
        super(KLDivergence, self).__init__()
        self.data_shape = data_shape

    def forward(self, prior: Distribution, posterior: Distribution, *, with_stats=False):
        full_loss = kl_divergence(posterior, prior)
        if type(full_loss) != list:
            full_loss = [full_loss]
        reduced_loss = torch.empty(len(full_loss))
        for i, loss in enumerate(full_loss):
            mean_axis = list(range(1, len(loss.size())))
            # normalizing occasional inf values
            #loss = torch.where(loss == float('inf'), torch.tensor(10), loss)
            #loss = torch.clamp(loss, max=1000.0)
            per_example_loss = torch.sum(loss, dim=mean_axis)

            assert len(per_example_loss.shape) == 1
            batch_size = per_example_loss.size()[0]

            scalar = batch_size * np.prod(self.data_shape)
            loss = torch.sum(per_example_loss) / (scalar * np.log(2)) # divide by ln(2) to convert to bit range

            reduced_loss[i] = loss
            #loss = torch.sum(loss)

        if with_stats:
            stats = self.stats(prior, posterior)
            return reduced_loss, stats
        return reduced_loss
    
    def stats(self, prior, posterior):
        with torch.no_grad():
            scale_ratio = posterior.scale / prior.scale
            loc_abs_diff = (posterior.loc - prior.loc).abs()
            t1 = -scale_ratio.log()
            t2 = loc_abs_diff / prior.scale
            t3 = scale_ratio * torch.exp(-loc_abs_diff / posterior.scale)
            batch_size = loc_abs_diff.size()[0]
            mse_mean = nn.functional.mse_loss(posterior.mean, prior.mean)
        return dict(
            t1=t1.sum() / batch_size,
            t2=t2.sum() / batch_size,
            t3=t3.sum() / batch_size,
            scale_ratio=scale_ratio.sum() / batch_size,
            loc_abs_diff=loc_abs_diff.sum() / batch_size,
            mean_mse=mse_mean,
        )

class LimesKLDivergence(KLDivergence):
    
    def __init__(self, data_shape):
        super(LimesKLDivergence, self).__init__(data_shape)
    
    
    def forward(self, prior: Distribution, posterior: Distribution, * , with_stats=False):
        """
        Custom kl divergence loss
        """
        #check if prior and posterior are Laplace distributions
        assert isinstance(prior, torch.distributions.laplace.Laplace) and isinstance(posterior, torch.distributions.laplace.Laplace), \
            "prior and posterior must be Laplace distributions"
        
        loc_abs_diff = (posterior.loc - prior.loc).abs()
        scale = softclip(prior.scale, 1e-3)
        t1 = scale.log()
        t2 = loc_abs_diff / scale
        loss = t1 + t2
        
        mean_axis = list(range(1, len(loss.size())))
        per_example_loss = torch.sum(loss, dim=mean_axis)

        assert len(per_example_loss.shape) == 1
        batch_size = per_example_loss.size()[0]

        scalar = batch_size * np.prod(self.data_shape)
        loss = torch.sum(per_example_loss) / (scalar * np.log(2)) # divide by ln(2) to convert to bit range
        mse_mean = nn.functional.mse_loss(posterior.mean, prior.mean)
        
        if with_stats:
            return t1 + t2, dict(
                t1=t1,
                t2=t2,
                loc_abs_diff=loc_abs_diff,
                scale=scale,
                mean_mse=mse_mean,
            )
        
        return loss
    
    
class LogProbKLDivergence(LogProb):
    # calculate reconstructions other than what the model is trained for using GenBlock
    def __init__(self, data_shape):
        super(LogProbKLDivergence, self).__init__(data_shape)
        
    def forward(self, prior: Distribution, posterior: Distribution, *, with_stats=False):
        # posterior.mean = target
        # prior = output distribution
        return super().forward(posterior.mean, prior, with_stats=with_stats)


def balanced_kl_divergence(cls, alpha=0.8):
    class BalancedKLDivergence(cls):
        """
        Balanced KL divergence loss
        """
        def __init__(self, data_shape, alpha=alpha):
            super(BalancedKLDivergence, self).__init__(data_shape)
            self.alpha = alpha
            
        def forward(self, prior: Distribution, posterior: Distribution, *, with_stats=False):
            detached_prior = prior.__class__(prior.loc.detach(), prior.scale.detach())
            detached_posterior = posterior.__class__(posterior.loc.detach(), posterior.scale.detach())
            
            kl_prior, stats = super().forward(prior, detached_posterior, with_stats=True)
            kl_posterior = super().forward(detached_prior, posterior)
            
            loss = self.alpha * kl_prior + (1 - self.alpha) * kl_posterior
            if with_stats:
                return loss, stats
            return loss
    return BalancedKLDivergence


def only_stats_kl_divergence(cls):
    class OnlyStatsKLDivergence(cls):
        def __init__(self, **kwargs):
            super(OnlyStatsKLDivergence, self).__init__(**kwargs)
            
        def forward(self, prior: Distribution, posterior: Distribution, *, with_stats=False):
            device = prior.loc.device
            if not with_stats:
                return torch.tensor(0., device=device)
            with torch.no_grad():
                loss, stats = super().forward(prior, posterior, with_stats=True)
                stats["loss"] = loss
            return torch.zeros_like(loss, device=device), stats
    return OnlyStatsKLDivergence
    
    

class SSIM(nn.Module):
    """
    Structural similarity index measure
    from Efficient-VDVAE paper
    """
    def __init__(self, image_channels, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
        super(SSIM, self).__init__()
        self.max_val = max_val

        self.k1 = k1
        self.k2 = k2
        self.filter_size = filter_size

        self.compensation = 1.

        self.kernel = SSIM._fspecial_gauss(filter_size, filter_sigma, image_channels)

    @staticmethod
    def _fspecial_gauss(filter_size, filter_sigma, image_channels):
        """Function to mimic the 'fspecial' gaussian MATLAB function."""
        coords = torch.arange(0, filter_size, dtype=torch.float32)
        coords -= (filter_size - 1.0) / 2.0

        g = torch.square(coords)
        g *= -0.5 / np.square(filter_sigma)

        g = torch.reshape(g, shape=(1, -1)) + torch.reshape(g, shape=(-1, 1))
        g = torch.reshape(g, shape=(1, -1))  # For tf.nn.softmax().
        g = F.softmax(g, dim=-1)
        g = torch.reshape(g, shape=(1, 1, filter_size, filter_size))
        return torch.tile(g, (image_channels, 1, 1, 1))  # .cuda()  # out_c, in_c // groups, h, w

    def _apply_filter(self, x):
        shape = list(x.size())
        if len(shape) == 3:
            x = torch.unsqueeze(x, dim=1)
        elif len(shape) == 4:
            x = torch.reshape(x, shape=[-1] + shape[-3:])  # b , c , h , w
        y = F.conv2d(x, weight=self.kernel.to(x.device), stride=1, padding=(self.filter_size - 1) // 2,
                     groups=x.shape[1])  # b, c, h, w
        return y

    def _compute_luminance_contrast_structure(self, x, y):
        c1 = (self.k1 * self.max_val) ** 2
        c2 = (self.k2 * self.max_val) ** 2

        # SSIM luminance measure is
        # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
        mean0 = self._apply_filter(x)
        mean1 = self._apply_filter(y)
        num0 = mean0 * mean1 * 2.0
        den0 = torch.square(mean0) + torch.square(mean1)
        luminance = (num0 + c1) / (den0 + c1)

        # SSIM contrast-structure measure is
        #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
        # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
        #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
        #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
        num1 = self._apply_filter(x * y) * 2.0
        den1 = self._apply_filter(torch.square(x) + torch.square(y))
        c2 *= self.compensation
        cs = (num1 - num0 + c2) / (den1 - den0 + c2)

        # SSIM score is the product of the luminance and contrast-structure measures.
        return luminance, cs

    def _compute_one_channel_ssim(self, x, y):
        luminance, contrast_structure = self._compute_luminance_contrast_structure(x, y)
        return (luminance * contrast_structure).mean(dim=(-2, -1))

    def forward(self, targets, outputs):
        ssim_per_channel = self._compute_one_channel_ssim(targets, outputs)
        return ssim_per_channel.mean(dim=-1)


class StructureSimilarityIndexMap(nn.Module):

    """
    Structural similarity index map loss
    from Efficient-VDVAE paper
    """

    def __init__(self, image_channels, unnormalized_max=255., filter_size=11):
        super(StructureSimilarityIndexMap, self).__init__()
        self.ssim = SSIM(image_channels=image_channels, max_val=unnormalized_max, filter_size=filter_size)

    def forward(self, targets, outputs, global_batch_size):
        if targets.size() != outputs.size():
            targets = targets.reshape(outputs.size())
        targets = targets * 127.5 + 127.5
        outputs = outputs * 127.5 + 127.5
        assert targets.size() == outputs.size()

        if len(targets.size()) == 5:
            batch_size, timeframe, channels, height, width = outputs.size()
            targets = targets.view(batch_size, timeframe * channels, height, width)
            outputs = outputs.view(batch_size, timeframe * channels, height, width)
        return self.calculate(targets, outputs, global_batch_size)

    def calculate(self, targets, outputs, batch_size):
        per_example_ssim = self.ssim(targets, outputs)
        mean_axis = list(range(1, len(per_example_ssim.size())))
        per_example_ssim = torch.sum(per_example_ssim, dim=mean_axis)

        loss = torch.sum(per_example_ssim) / batch_size
        return loss
    
    
class JEPALoss(nn.Module):
    """
    loss_definition: [
        {
            "name": "jepa_latent",
            "in_keys": ["manifold_belief", "manifold_state_target"] 
            "type": "hinge",
            "weight": 1.0,
        }
    ]
    
    """
    
    def __init__(self, loss_definition: list, data_shape):
        super(JEPALoss, self).__init__()
        for loss in loss_definition:
            assert isinstance(loss, dict)
            assert "name" in loss.keys(), "name is required in loss definition"
            assert "in_keys" in loss.keys(), "in_keys are required in loss definition"
            assert "type" in loss.keys(), "type is required in loss definition"
            assert "weight" in loss.keys(), "weight is required in loss definition"
        self.loss_definition = loss_definition
        self.data_shape = data_shape
        
    
    def forward(self, targets, distributions, computed, step_n):
        
        losses = dict(
            loss = 0,
        )
        
        # reconstruction loss
        if params.reconstruction_loss is not None:
            # input: x, output: x_hat
            loss_reconstruct = params.reconstruction_loss(targets, distributions['output'])
            
            #loss_reconstruct = 0
            #for name, value in computed.items():
            #    if name.endswith('x'):
            #        target = value
            #        output_distribution = distributions[name + '_hat'][0] # prior
            #        loss_reconstruct += params.reconstruction_loss(target, output_distribution)
            losses['reconstruction_loss'] = -loss_reconstruct
            losses['loss'] -= loss_reconstruct
                
                    
        # kl divergence loss
        if params.kl_divergence is not None:
            global_variational_prior_losses = []
            kl_divs = dict()
            for block_name, dists in distributions.items():
                if block_name == 'output' or dists is None or len(dists) != 2 or dists[1] is None:
                    continue
                prior, posterior = dists
                block_kl = params.kl_divergence(prior, posterior)
                global_variational_prior_losses.append(block_kl)
                
                if f"kl_{block_name.strip('_')}" in kl_divs:
                    kl_divs[f"kl_{block_name.strip('_')}"] += block_kl
                else:
                    kl_divs[f"kl_{block_name.strip('_')}"] = block_kl

            global_variational_prior_losses = torch.stack(global_variational_prior_losses)
            kl_div = torch.sum(global_variational_prior_losses)  # / np.log(2.)
            global_variational_prior_loss = kl_div \
                if not params.loss_params.use_gamma_schedule \
                else params.gamma_schedule(global_variational_prior_losses, step_n=step_n)  # does this function still works with one less parameter?
            global_var_loss = params.kldiv_schedule(step_n) * global_variational_prior_loss  # beta
            
            losses['kl_div'] = kl_div
            losses['loss'] += global_var_loss
            losses.update(kl_divs)
            
        if "reconstruction_loss" in losses.keys() and "kl_div" in losses.keys():
            losses['elbo'] = -losses['reconstruction_loss'] + losses['kl_div']
        
        
        # custom losses
        for loss_def in self.loss_definition:
            name = loss_def['name']
            in_keys = loss_def['in_keys']
            
            prediction_key = in_keys[0]
            target_key = in_keys[1]
            
            loss_fn = self.get_loss_fn(loss_def['type'])
            weight = loss_def['weight']
            
            this_custom_loss = 0
            for key, value in computed.items():
                if key.endswith(target_key):
                    target = value
                    prediction = computed[key[:-len(target_key)] + prediction_key]
                    this_custom_loss += loss_fn(target, prediction)
                    
            mean_axis = list(range(1, len(this_custom_loss.size())))
            per_example_loss = torch.sum(this_custom_loss, dim=mean_axis)

            assert len(per_example_loss.shape) == 1
            batch_size = per_example_loss.size()[0]

            scalar = batch_size * np.prod(self.data_shape)
            this_custom_loss = torch.sum(per_example_loss) / scalar # divide by ln(2) to convert to bit range
                    
            losses[name] = this_custom_loss
            losses['loss'] += weight * this_custom_loss
            
        return losses
            

    def get_loss_fn(self, loss_fn):
        if loss_fn == 'hinge':
            return nn.HingeEmbeddingLoss(reduction="none")
        elif loss_fn == 'l1':
            return nn.L1Loss(reduction="none")
        elif loss_fn == 'mse':
            return nn.MSELoss(reduction="none")
        elif loss_fn == 'cross_entropy':
            return nn.CrossEntropyLoss(reduction="none")
        elif loss_fn == 'bce':
            return nn.BCELoss(reduction="none")
        elif loss_fn == 'bce_with_logits':
            return nn.BCEWithLogitsLoss(reduction="none")
        else:
            raise ValueError(f'Unknown loss function: {loss_fn}')
            
        
        
               
        
        
            
            
        
    
