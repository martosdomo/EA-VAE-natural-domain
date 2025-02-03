import torch
from torch import tensor, distributions as dist
from torch.distributions.kl import register_kl, kl_divergence
import numpy as np
from torch.nn.functional import relu, softplus
from ..utils import split_mu_sigma

from .. import params


dist.distribution.Distribution.set_default_validate_args(False)


def generate_distribution(logits, distribution='normal', temperature=None) -> dist.Distribution:
    """
    Generate parameterized distribution

    :param mu: the mean of the distribution
    :param sigma: the standard deviation of the distribution, not needed for mixture of logistics
    :param distribution: 'mixtures_of_logistics', 'normal', 'laplace'
    :param sigma_nonlin: 'logstd', 'std'
    :param sigma_param: 'var', 'std'

    NORMAL: ("normal", [sigma_nonlin], [sigma_param])
    LAPLACE: ("laplace", [sigma_nonlin], [sigma_param])
    LOGNORMAL: ("lognormal", [sigma_nonlin], [sigma_param])
    LOGLAPLACE: ("loglaplace", [sigma_nonlin], [sigma_param])
    SOFTLAPLACE: ("softlaplace", [sigma_nonlin], [sigma_param])
        
    :return: torch.distributions.Distribution object
    """
    
    distribution_name = distribution if isinstance(distribution, str) else distribution[0]
    
    assert distribution_name in ['normal', 'laplace', 'lognormal', 'loglaplace', 'softlaplace'], \
        f'Unknown distribution {distribution_name}'
    
    from .. import params
    model_params = params.model_params
    beta = model_params.gradient_smoothing_beta
    
    if isinstance(distribution, str):
        sigma_nonlin = model_params.distribution_base
        sigma_param = model_params.distribution_sigma_param
    elif isinstance(distribution, tuple):
        assert len(distribution) == 3, f"Expected 3 parameters for {distribution_name}, got {len(distribution)}"
        sigma_nonlin = distribution[1]
        sigma_param = distribution[2]
        
    mu, sigma = split_mu_sigma(logits)
    return generate_loc_scale(mu, sigma, distribution_name, sigma_nonlin, sigma_param, beta, temperature)


def generate_loc_scale(mu, sigma, distribution_name, sigma_nonlin, sigma_param, beta, temperature=None):
        if temperature is not None:
            sigma = sigma + torch.ones_like(sigma) * np.log(temperature)
    
        if sigma_nonlin == 'logstd':
            if distribution_name not in ['laplace', 'loglaplace', 'softlaplace']:
                sigma = torch.exp(0.5*sigma)
            else:
                sigma = torch.exp(sigma)
        elif sigma_nonlin == 'std':
            sigma = torch.nn.Softplus(beta=beta)(sigma)
        elif sigma_nonlin != 'none':
            raise ValueError(f'Unknown sigma_nonlin {sigma_nonlin}')

        if sigma_param == 'var':
            sigma = torch.sqrt(sigma)
        elif sigma_param != 'std':
            raise ValueError(f'Unknown sigma_param {sigma_param}')

        if distribution_name == 'normal':
            return dist.Normal(loc=mu, scale=sigma)
        elif distribution_name == 'laplace':
            return dist.Laplace(loc=mu, scale=sigma)
        elif distribution_name == 'lognormal':
            mu = torch.clamp(mu, min=-5.2983)#, max=2.3025) # exp(-5.2983) = 0.005, exp(2.3025) = 10
            return dist.LogNormal(loc=mu, scale=sigma)
        elif distribution_name == 'loglaplace':
            mu = torch.clamp(mu, min=-5.2983)#, max=2.3025) # exp(-5.2983) = 0.005, exp(2.3025) = 10
            return LogLaplace(loc=mu, scale=sigma)
        elif distribution_name == 'softlaplace':
            return SoftLaplace(loc=mu, scale=sigma)
        else:
            raise ValueError(f'Unknown distribution {distribution_name}')


class SoftLaplace(dist.Laplace):
    def __init__(self, loc, scale, validate_args=None):
        super().__init__(loc=loc, scale=scale, validate_args=validate_args)
        
    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply softplus transformation"""
        return softplus(x)
    
    '''@property
    def mean(self, sample_shape) -> torch.Tensor:
        """
        Mean of SoftLaplace is intractable, returns transformed sample instead.
        """
        with torch.no_grad():
            return self.rsample(sample_shape)'''
    
    def sample(self, sample_shape = torch.Size()) -> torch.Tensor:
        """Generate non-differentiable samples"""
        return self.rsample(sample_shape).detach()
    
    def rsample(self, sample_shape = torch.Size()) -> torch.Tensor:
        """Generate differentiable samples"""
        return self._transform(super().rsample(sample_shape))
    

class LogLaplace(dist.Laplace):
    def __init__(self, loc, scale, validate_args=None):
        super(LogLaplace, self).__init__(loc=loc, scale=scale, validate_args=validate_args)
        self.loc = loc
        self.scale = scale

    @property
    def mean(self):
        # Scale parameter of loglaplace should be less than 1 to have a finite mean
        # where condition is not met, return nan
        mask = self.scale < 1
        masked_mean = torch.where(mask, torch.exp(self.loc) / (1 - self.scale**2), torch.nan)
        return masked_mean

    @property
    def stddev(self):
        # Scale parameter of loglaplace should be smaller than 0.5 to have a finite stddev
        mask = self.scale < 0.5
        masked_var = torch.where(mask, 
                                 torch.exp(2*self.loc) / (1 - 4*self.scale**2) - \
                                    torch.exp(2*self.loc) / (1 - self.scale**2)**2, 
                                 torch.nan)
        return torch.sqrt(masked_var)
    
    def sample(self, sample_shape=torch.Size()):
        #laplace_sample = super(LogLaplace, self).sample(sample_shape)
        #return torch.exp(laplace_sample)
        return self.rsample(sample_shape).detach()
    
    def rsample(self, sample_shape=torch.Size()):
        laplace_sample = super(LogLaplace, self).rsample(sample_shape)
        return torch.exp(laplace_sample)

class ConcatenatedDistribution(dist.distribution.Distribution):
    """
    Concatenated distribution
    
    :param distributions: list of distributions
    :param fuse: 'sum' or 'mean'
    """
    def __init__(self, distributions: list, fuse: str = 'sum'):
        self.distributions = distributions
        self.fuse = fuse
        dbs = distributions[0].batch_shape
        batch_shape = torch.Size([dbs[0], len(distributions), *dbs[1:]])
        super().__init__(batch_shape=batch_shape)

    def extend(self, distributions: list):
        self.distributions.extend(distributions)
        return ConcatenatedDistribution(self.distributions, self.fuse)

    @property
    def mean(self) -> torch.Tensor:
        means = [d.mean for d in self.distributions]
        means = torch.cat(means, dim=1)
        return means

    @property
    def variance(self) -> torch.Tensor:
        variances = [d.variance for d in self.distributions]
        variances = torch.cat(variances, dim=1)
        return variances
    
    @property
    def loc(self) -> torch.Tensor:
        locs = [d.loc for d in self.distributions]
        locs = torch.cat(locs, dim=1)
        return locs
    
    @property
    def scale(self) -> torch.Tensor:
        scales = [d.scale for d in self.distributions]
        scales = torch.cat(scales, dim=1)
        return scales

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        samples = [d.rsample(sample_shape) for d in self.distributions]
        samples = torch.cat(samples, dim=1)
        return samples

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        log_probs = [d.log_prob(value[:, i]) for i, d in enumerate(self.distributions)]
        log_probs = torch.cat(log_probs, dim=1)
        if self.fuse == 'sum':
            log_probs = torch.sum(log_probs, dim=1)
        elif self.fuse == 'mean':
            log_probs = torch.mean(log_probs, dim=1)
        else:
            raise ValueError(f'Unknown fuse {self.fuse}')
        return log_probs

    def entropy(self) -> torch.Tensor:
        entropies = [d.entropy() for d in self.distributions]
        entropies = torch.cat(entropies, dim=1)
        if self.fuse == 'sum':
            entropies = torch.sum(entropies, dim=0)
        elif self.fuse == 'mean':
            entropies = torch.mean(entropies, dim=0)
        else:
            raise ValueError(f'Unknown fuse {self.fuse}')
        return entropies

@register_kl(ConcatenatedDistribution, ConcatenatedDistribution)
def _kl_concat_concat(p, q):
    kl_divs = []
    for i in range(len(p.distributions)):
        kl_divs.append(kl_divergence(p.distributions[i], q.distributions[i]))

    return kl_divs