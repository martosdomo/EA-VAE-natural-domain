import torch
from torch import tensor
from .elements.distributions import generate_distribution
from .elements.nets import get_net
from .utils import SerializableModule
from hvae_backbone.elements.distributions import generate_distribution, ConcatenatedDistribution


"""
Sampling and forward methods
"""

class _Block(SerializableModule):
    """
    Base class for all blocks
    """

    def __init__(self, input_id = None):
        super(_Block, self).__init__()
        self.input = InputPipeline(input_id)
        self.output = None

    def forward(self, computed: dict, **kwargs) -> (dict, None):
        return dict(), None

    def sample_from_prior(self, computed: dict, t = None, **kwargs) -> (dict, None):
        return self.forward(computed)

    def freeze(self, net_name: str):
        for name, param in self.named_parameters():
            if net_name in name:
                param.requires_grad = False

    def set_output(self, output: str) -> None:
        self.output = output

    def serialize(self) -> dict:
        return dict(
            input=self.input.serialize(),
            output=self.output,
            type=self.__class__
        )


class InputPipeline(SerializableModule):
    """
    Helper class for preprocessing pipeline
    """

    def __init__(self, input_pipeline: str or tuple or list):
        super(InputPipeline, self).__init__()
        self.inputs = self.parse(input_pipeline)

    def forward(self, computed):
        return self._load(computed, self.inputs)

    def parse(self, input_pipeline):
        if isinstance(input_pipeline, str):
            return input_pipeline
        elif isinstance(input_pipeline, tuple):
            return tuple([self.parse(i) for i in input_pipeline])
        elif isinstance(input_pipeline, list):
            return [self.parse(i) for i in input_pipeline]
        elif isinstance(input_pipeline, (SerializableModule)):
            self.register_module(str(len(self._modules)), input_pipeline)
            return input_pipeline
        elif hasattr(input_pipeline, "config"):
            net = get_net(input_pipeline)
            self.register_module(str(len(self._modules)), net)
            return net
        else:
            raise ValueError(f"Unknown input pipeline element {input_pipeline}")

    def serialize(self):
        return self._serialize(self.inputs)

    def _serialize(self, item):
        if isinstance(item, str):
            return item
        elif isinstance(item, list):
            return [i.serialize() if isinstance(i, (SerializableModule))
                    else self._serialize(i) for i in item]
        elif isinstance(item, tuple):
            return tuple([self._serialize(i) for i in item])

    @staticmethod
    def deserialize(serialized):
        if isinstance(serialized, str):
            return serialized
        elif isinstance(serialized, list):
            return [i["type"].deserialize(i) if isinstance(i, dict) and "type" in i.keys()
                    else InputPipeline.deserialize(i) for i in serialized]
        elif isinstance(serialized, tuple):
            return tuple([InputPipeline.deserialize(i) for i in serialized])

    @staticmethod
    def _load(computed: dict, inputs):
        def _validate_get(_inputs):
            if not isinstance(_inputs, str):
                raise ValueError(f"Input {_inputs} must be a string")
            if _inputs not in computed.keys():
                raise ValueError(f"Input {_inputs} not found in computed")
            return computed[_inputs]

        # single input
        if isinstance(inputs, str):
            return _validate_get(inputs)

        # multiple inputs
        elif isinstance(inputs, tuple):
            return tuple([InputPipeline._load(computed, i) for i in inputs])

        # list of operations
        elif isinstance(inputs, list):
            if len(inputs) < 2:
                raise ValueError(f"Preprocessing pipeline must have at least 2 elements, got {len(inputs)}"
                                 f"Provide the inputs in [inputs, operation1, operation2, ...] format")
            if not isinstance(inputs[0], (str, tuple)):
                raise ValueError(f"First element of the preprocessing pipeline "
                                 f"must be the input id or tuple of input ids, got {inputs[0]}")
            input_tensors = InputPipeline._load(computed, inputs[0])
            for op in inputs[1:]:
                if callable(op):
                    input_tensors = op(*input_tensors) if isinstance(input_tensors, tuple) \
                                                        else op(input_tensors)
                elif isinstance(op, str):
                    if op == "concat":
                        input_tensors = torch.cat(input_tensors, dim=1)
                    elif op == "sub":
                        input_tensors = input_tensors[0] - input_tensors[1]
                    elif op == "add":
                        input_tensors = input_tensors[0] + input_tensors[1]
            return input_tensors


class SimpleBlock(_Block):
    """
    Simple block that takes an input and returns an output
    No sampling is performed
    """

    def __init__(self, net, input_id):
        super(SimpleBlock, self).__init__(input_id)
        self.net = get_net(net)

    def forward(self, computed: dict, **kwargs) -> (dict, None):
        inputs = self.input(computed)
        output = self.net(inputs)
        computed[self.output] = output
        return computed, None

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["net"] = self.net.serialize()
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        net = serialized["net"]["type"].deserialize(serialized["net"])
        return SimpleBlock(net=net, input_id=InputPipeline.deserialize(serialized["input"]))


class InputBlock(SimpleBlock):
    """
    Block that takes an input
    and runs it through a preprocessing net if one is given
    """

    def __init__(self, net=None):
        super(InputBlock, self).__init__(net, "input")

    def forward(self, inputs: dict, **kwargs) -> tuple:
        if isinstance(inputs, dict):
            computed = inputs
        elif isinstance(inputs, torch.Tensor):
            computed = {"inputs": inputs,
                        self.output: self.net(inputs)}
        else:
            raise ValueError(f"Input must be a tensor or a dict got {type(inputs)}")
        distributions = dict()
        return computed, distributions

    @staticmethod
    def deserialize(serialized: dict):
        net = serialized["net"]["type"].deserialize(serialized["net"])
        return InputBlock(net=net)


class SimpleGenBlock(_Block):
    """
    Takes an input and samples from a prior distribution
    """

    def __init__(self, net, input_id, output_distribution: str = 'normal'):
        super(SimpleGenBlock, self).__init__(input_id)
        self.prior_net = get_net(net)
        self.output_distribution: str = output_distribution

    def _sample_uncond(self, x: tensor, t: float or int = None, use_mean=False) -> tensor:
        x_prior = self.prior_net(x)
        prior = generate_distribution(x_prior, self.output_distribution, t)
        z = prior.rsample() if not use_mean else prior.mean
        return z, (prior, None)

    def forward(self, computed: dict, use_mean=False, **kwargs) -> (dict, tuple):
        x = self.input(computed)
        z, distribution = self._sample_uncond(x, use_mean=use_mean)
        computed[self.output] = z
        return computed, distribution

    def sample_from_prior(self, computed: dict, t: float or int = None, use_mean=False, **kwargs) -> (dict, tuple):
        x = self.input(computed)
        z, dist = self._sample_uncond(x, t, use_mean=use_mean)
        computed[self.output] = z
        return computed, dist

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["prior_net"] = self.prior_net.serialize()
        serialized["output_distribution"] = self.output_distribution
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        prior_net = serialized["prior_net"]["type"].deserialize(serialized["prior_net"])
        return SimpleGenBlock(
            net=prior_net,
            input_id=InputPipeline.deserialize(serialized["input"]),
            output_distribution=serialized["output_distribution"]
        )
        
    def extra_repr(self) -> str:
        return super().extra_repr() + f"\noutput_distribution={self.output_distribution}\n"


class OutputBlock(SimpleGenBlock):
    def __init__(self, net, input_id, output_distribution: str = 'normal', stddev = None):
        assert stddev is not None, "stddev must be provided"
        super(OutputBlock, self).__init__(net, input_id, output_distribution)

        self.stddev = stddev if isinstance(stddev, torch.Tensor) else torch.tensor(stddev)
        assert isinstance(self.output_distribution, str), \
                "Output distribution must be a string for OutputBlock. " \
                "Standard deviation is used with no nonlinearity."

    def forward(self, computed: dict, use_mean=False, **kwargs) -> (dict, tuple):
        x = self.input(computed)
        pm = self.prior_net(x)
        pv = self.stddev * torch.ones_like(pm, device=pm.device)

        x_prior = torch.cat([pm, pv], dim=1)
        prior = generate_distribution(x_prior, (self.output_distribution, 'none', 'std'))
        z = prior.sample() if not use_mean else prior.mean
        distribution = (prior, None)
        computed[self.output] = z
        return computed, distribution 
        
    def sample_from_prior(self, computed: dict, t: float or int = None, use_mean=False, **kwargs) -> (dict, tuple):
        x = self.input(computed)
        pm = self.prior_net(x)
        pv = self.stddev * torch.ones_like(pm, device=pm.device)
        x_prior = torch.cat([pm, pv], dim=1)
        prior = generate_distribution(x_prior, (self.output_distribution, 'none', 'std'), t)
        z = prior.sample() if not use_mean else prior.mean
        distribution =  (prior, None)
        computed[self.output] = z
        return computed, distribution
    

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["stddev"] = self.stddev
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        prior_net = serialized.pop("prior_net")
        prior_net = prior_net["type"].deserialize(prior_net)
        return OutputBlock(
            net=prior_net,
            input_id=InputPipeline.deserialize(serialized.pop("input")),
            output_distribution=serialized["output_distribution"],
            stddev=serialized.pop("stddev"),
        )
        
    def extra_repr(self) -> str:
        return super().extra_repr() + f"\nstddev={self.stddev}\n"


class GenBlock(SimpleGenBlock):
    """
    Takes an input,
    samples from a prior distribution,
    (takes a condition,
    samples from a posterior distribution),
    and returns the sample
    """

    def __init__(self,
                 prior_net,
                 posterior_net,
                 input_id, condition,
                 output_distribution: str = 'normal',
                 kl_loss = 'default'):
        super(GenBlock, self).__init__(prior_net, input_id, output_distribution)
        self.prior_net = get_net(prior_net)
        self.posterior_net = get_net(posterior_net)
        self.condition = InputPipeline(condition)
        self.kl_loss = kl_loss

    def _sample(self, x: tensor, cond: tensor, variate_mask=None, use_mean=False) -> (tensor, tuple):
        x_prior = self.prior_net(x)
        prior = generate_distribution(x_prior, self.output_distribution)

        x_posterior = self.posterior_net(cond)
        posterior = generate_distribution(x_posterior, self.output_distribution)
        z = posterior.rsample() if not use_mean else posterior.mean

        if variate_mask is not None:
            z_prior = prior.rsample() if not use_mean else prior.mean
            z = self.prune(z, z_prior, variate_mask)

        return z, (prior, posterior, self.kl_loss)

    def _sample_uncond(self, x: tensor, t: float or int = None, use_mean=False) -> tensor:
        x_prior = self.prior_net(x)
        prior = generate_distribution(x_prior, self.output_distribution, t)
        z = prior.sample() if not use_mean else prior.mean
        return z, (prior, None)

    def forward(self, computed: dict, variate_mask=None, use_mean=False, **kwargs) -> (dict, tuple):
        x = self.input(computed)
        cond = self.condition(computed)
        z, distributions = self._sample(x, cond, variate_mask, use_mean=use_mean)
        computed['z_posterior_mean'] = distributions[1].mean
        computed['z_posterior_std'] = distributions[1].stddev
        computed[self.output] = z
        return computed, distributions

    def sample_from_prior(self, computed: dict, t: float or int = None, use_mean=False, **kwargs) -> (dict, tuple):
        x = self.input(computed)
        z, dist = self._sample_uncond(x, t, use_mean=use_mean)
        computed[self.output] = z
        return computed, dist

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["prior_net"] = self.prior_net.serialize()
        serialized["posterior_net"] = self.posterior_net.serialize()
        serialized["condition"] = self.condition.serialize()
        serialized["output_distribution"] = self.output_distribution
        serialized["kl_loss"] = self.kl_loss
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        prior_net = serialized["prior_net"]["type"].deserialize(serialized["prior_net"])
        posterior_net = serialized["posterior_net"]["type"].deserialize(serialized["posterior_net"])
        return GenBlock(
            prior_net=prior_net,
            posterior_net=posterior_net,
            input_id=InputPipeline.deserialize(serialized["input"]),
            condition=InputPipeline.deserialize(serialized["condition"]),
            output_distribution=serialized["output_distribution"],
            kl_loss=serialized.pop("kl_loss", 'default')
        )
    
    def extra_repr(self) -> str:
        return super().extra_repr()


class ContrastiveOutputBlock(OutputBlock):
    def __init__(self, net, input_id, contrast_dims: int = 1, output_distribution: str = 'normal', stddev=None):
        super().__init__(net, input_id, output_distribution, stddev)
        self.contrast_dims = contrast_dims

    def forward(self, computed: dict, use_mean=False, **kwargs) -> (dict, tuple):
        x = self.input(computed)
        x_input = x[:, :-self.contrast_dims]
        contrast = x[:, -self.contrast_dims:]
        
        pm = x_input
        pm_shape = pm.shape
        pm_flattened = torch.flatten(pm, start_dim=1)
        pm = pm_flattened * contrast
        pm = pm.reshape(pm_shape)       

        pm = self.prior_net(pm) # decoder
        pv = self.stddev * torch.ones_like(pm, device=pm.device)

        x_prior = torch.cat([pm, pv], dim=1)
        prior = generate_distribution(x_prior, (self.output_distribution, 'none', 'std'))
        z = prior.sample() if not use_mean else prior.mean
        distribution =  (prior, None)
        computed[self.output] = z
        return computed, distribution

    def sample_from_prior(self, computed: dict, t: float or int = None, use_mean=False, **kwargs) -> (dict, tuple):
        x = self.input(computed)
        x_input = x[:, :-self.contrast_dims]
        contrast = x[:, -self.contrast_dims:]
        pm = x_input

        pm_shape = pm.shape
        pm_flattened = torch.flatten(pm, start_dim=1)
        pm = pm_flattened * contrast
        pm = pm.reshape(pm_shape)

        pm = self.prior_net(x_input)
        pv = self.stddev * torch.ones_like(pm, device=pm.device)
        x_prior = torch.cat([pm, pv], dim=1)        
        prior = generate_distribution(x_prior, (self.output_distribution, 'none', 'std'), t)
        z = prior.sample() if not use_mean else prior.mean
        distribution = (prior, None)
        computed[self.output] = z
        return z, distribution
    
    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["contrast_dims"] = self.contrast_dims
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        prior_net = serialized.pop("prior_net")
        prior_net = prior_net["type"].deserialize(prior_net)
        return ContrastiveOutputBlock(
            net=prior_net,
            input_id=InputPipeline.deserialize(serialized["input"]),
            contrast_dims=serialized["contrast_dims"],
            output_distribution=serialized["output_distribution"],
            stddev=serialized.pop("stddev"),
        )

class ContrastiveGenBlock(SimpleGenBlock):
    '''
        Enables having multiple distributions on different latens dimensions.

        The regular output distribution can be: 'normal', 'laplace'
        The contrast distribution can be: 'lognormal', 'softlaplace', 'loglaplace'
    '''

    def __init__(self,
                 prior_net,
                 posterior_net,
                 input_id, condition,
                 output_distribution: str = 'normal',
                 contrast_distribution: str = 'lognormal',
                 contrast_dims: int = 1):
        super(ContrastiveGenBlock, self).__init__(prior_net, input_id, output_distribution)
        self.prior_net = get_net(prior_net)
        self.posterior_net = get_net(posterior_net)
        self.condition = InputPipeline(condition)
        self.contrast_distribution = contrast_distribution
        self.contrast_dims = contrast_dims

    def generate_concatenated(self, z, z_distribution, contrast_distribution, temperature=None):
        length = z.shape[1]
        mean_values = z[:, :length//2]
        sigma_values = z[:, length//2:]
        z_dims = torch.cat((mean_values[:, :-self.contrast_dims], 
                            sigma_values[:, :-self.contrast_dims]), 
                            dim=1)
        contrast_dims = torch.cat((mean_values[:, -self.contrast_dims:],
                                   sigma_values[:, -self.contrast_dims:]), 
                                   dim=1)
        p = generate_distribution(z_dims, z_distribution, temperature) # z dims 
        q = generate_distribution(contrast_dims, contrast_distribution, temperature) # s (contrast) dims
        return ConcatenatedDistribution([p, q])

    def _sample(self, x: tensor, cond: tensor, variate_mask=None, use_mean=False) -> (tensor, tuple):
        x_prior = self.prior_net(x)
        prior = self.generate_concatenated(x_prior, self.output_distribution, self.contrast_distribution)
        x_posterior = self.posterior_net(cond)
        posterior = self.generate_concatenated(x_posterior, self.output_distribution, self.contrast_distribution)
        z = posterior.rsample() if not use_mean else posterior.mean

        if variate_mask is not None:
            z_prior = prior.rsample() if not use_mean else prior.mean
            z = self.prune(z, z_prior, variate_mask)

        return z, (prior, posterior)

    def _sample_uncond(self, x: tensor, t: float or int = None, use_mean=False) -> tensor:
        x_prior = self.prior_net(x)
        prior = self.generate_concatenated(x_prior, self.output_distribution, self.contrast_distribution, t)
        z = prior.sample() if not use_mean else prior.mean
        return z, (prior, None)

    def forward(self, computed: dict, variate_mask=None, use_mean=False, **kwargs) -> (dict, tuple):
        x = self.input(computed)
        cond = self.condition(computed)
        z, distributions = self._sample(x, cond, variate_mask, use_mean=use_mean)
        computed['z_posterior_mean'] = distributions[1].mean
        computed['z_posterior_std'] = distributions[1].stddev
        computed['z_posterior_sample'] = distributions[1].sample()
        computed['z_posterior_loc'] = distributions[1].loc
        computed[self.output] = z
        return computed, distributions

    def sample_from_prior(self, computed: dict, t: float or int = None, use_mean=False, **kwargs) -> (dict, tuple):
        x = self.input(computed)
        z, dist = self._sample_uncond(x, t, use_mean=use_mean)
        computed[self.output] = z
        return computed, dist

    def serialize(self) -> dict:
        serialized = super().serialize()
        serialized["contrast_distribution"] = self.contrast_distribution
        return serialized

    @staticmethod
    def deserialize(serialized: dict):
        prior_net = serialized["prior_net"]["type"].deserialize(serialized["prior_net"])
        posterior_net = serialized["posterior_net"]["type"].deserialize(serialized["posterior_net"])
        return ContrastiveGenBlock(
            prior_net=prior_net,
            posterior_net=posterior_net,
            input_id=InputPipeline.deserialize(serialized["input"]),
            condition=InputPipeline.deserialize(serialized["condition"]),
            output_distribution=serialized["output_distribution"],
            contrast_distribution=serialized["contrast_distribution"],
        )
    
    def extra_repr(self) -> str:
        return super().extra_repr()
