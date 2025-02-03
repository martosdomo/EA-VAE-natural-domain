from collections import OrderedDict
from torch import tensor
from torch.nn import init
#from torch.nn.utils.parametrizations import weight_norm as wn

from ..elements.layers import *
from ..utils import SerializableModule, SerializableSequential as Sequential, unit_weight_norm
# from ..utils import SerializableModule, unit_weight_norm
from .. import Hyperparams


def get_net(model):
    """
    Get net from
    -string model type,
    -hyperparameter config
    -or list of the above

    :param model: str, Hyperparams, SerializableModule, list
    """

    if model is None:
        return Sequential()

    # Load model from hyperparameter config
    elif isinstance(model, Hyperparams):

        if "type" not in model.keys():
            raise ValueError("Model type not specified.")
        if model.type == 'mlp':
            return MLPNet.from_hparams(model)
        else:
            raise NotImplementedError("Model type not supported.")


    # Load model from SerializableModule
    elif isinstance(model, SerializableModule):
        return model

    # Load model from list of any of the above
    elif isinstance(model, list):
        # Load model from list
        return Sequential(*list(map(get_net, model)))

    else:
        raise NotImplementedError("Model type not supported.")


class MLPNet(SerializableModule):

    """
    Parametric multilayer perceptron network

    :param input_size: int, the size of the input
    :param hidden_sizes: list of int, the sizes of the hidden layers
    :param output_size: int, the size of the output
    :param residual: bool, whether to use residual connections
    :param activation: torch.nn.Module, the activation function to use
    """
    def __init__(self, input_size, hidden_sizes, output_size, residual=False, activation=nn.ReLU(), weight_norm=False, init=None, **kwargs):
        super(MLPNet, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.residual = residual
        self.weight_norm = weight_norm

        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            l = nn.Linear(sizes[i], sizes[i + 1])
            if self.weight_norm:
                l = unit_weight_norm(l, dim=0)
            layers.append(l)
            if i < len(sizes) - 2:
                layers.append(self.activation)

        self.mlp_layers = nn.Sequential(*layers)
        self.init = init
        if init is not None:
            self.initialize_parameters(method=init)

    def forward(self, inputs):
        x = inputs.to(self.mlp_layers[0].weight.device)
        #import pdb; pdb.set_trace()
        x = self.mlp_layers(x)
        outputs = x if not self.residual else inputs + x
        return outputs

    @staticmethod
    def from_hparams(hparams: Hyperparams):
        return MLPNet(
            input_size=hparams.input_size,
            hidden_sizes=hparams.hidden_sizes,
            output_size=hparams.output_size,
            activation=hparams.activation,
            residual=hparams.residual,
        )

    def serialize(self):
        return dict(
            type=self.__class__,
            state_dict=self.state_dict(),
            params=dict(
                input_size=self.input_size,
                hidden_sizes=self.hidden_sizes,
                output_size=self.output_size,
                activation=self.activation,
                residual=self.residual,
                weight_norm=self.weight_norm
            )
        )

    @staticmethod
    def deserialize(serialized):
        net = MLPNet(**serialized["params"])
        net.load_state_dict(serialized["state_dict"])
        return net
    
    def initialize_parameters(self, method='xavier_uniform'):
        """
        Initialize the parameters of a PyTorch module using the specified method.

        Parameters:
        - module (torch.nn.Module): The PyTorch module whose parameters need to be initialized.
        - method (str): The initialization method. Default is 'xavier_uniform'.
                       Other options include 'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
                       'orthogonal', 'uniform', 'normal', etc.

        Returns:
        None
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if method == 'xavier_uniform':
                    init.xavier_uniform_(param)
                elif method == 'xavier_normal':
                    init.xavier_normal_(param)
                elif method == 'kaiming_uniform':
                    init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
                elif method == 'kaiming_normal':
                    init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
                elif method == 'orthogonal':
                    init.orthogonal_(param)
                elif method == 'uniform':
                    init.uniform_(param, a=0.0, b=1.0)
                elif method == 'normal':
                    init.normal_(param, mean=0.0, std=1.0)
                else:
                    raise ValueError(f"Unsupported initialization method: {method}")
                
    def extra_repr(self) -> str:
        return super().extra_repr() + f"residual={self.residual}, weight_norm={self.weight_norm}"
    