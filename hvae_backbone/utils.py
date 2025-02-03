import os
import logging
import pickle
import numpy as np
import torch
from torch.nn import Sequential, Module, ModuleList
import torch.nn.functional as F

"""
-------------------
MODEL UTILS
-------------------
"""


class OrderedModuleDict(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._keys = list()
        self._values = ModuleList([])
        for key, module in kwargs.items():
            self[key] = module

    def update(self, modules):
        for key, module in modules.items():
            self[key] = module

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        elif isinstance(key, str):
            index = self._keys.index(key)
            return self._values[index]
        else:
            raise KeyError(f"Key {key} is not a string or an integer")

    def __setitem__(self, key, module):
        if key in self._keys:
            raise KeyError(f"Key {key} already exists")
        self._keys.append(key)
        self._values.append(module)

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        return iter(self._values)

    def keys(self):
        return self._keys

    def values(self):
        return self._values


def split_mu_sigma(x, chunks=2, dim=1):
    if x.shape[dim] % chunks != 0:
        if x.shape[dim] == 1:
            return x, None
        """
        raise ValueError(f"Can't split tensor of shape "
                         f"{x.shape} into {chunks} chunks "
                         f"along dim {dim}")"""
    chunks = torch.chunk(x, chunks, dim=dim)
    if chunks[0].shape[dim] == 1:
        for chunk in chunks:
            chunk.squeeze(dim=dim)
    return chunks


"""
-------------------
TRAIN/LOG UTILS
-------------------
"""


def load_model(load_from):
    from .checkpoint import Checkpoint

    assert load_from is not None
    experiment = None
    print(f"Loading experiment from {load_from}")
    if os.path.exists(load_from):
        load_from_file = load_from

    # load experiment from checkpoint
    if load_from_file is not None and os.path.isfile(load_from_file):
        print(f"Loading experiment from {load_from_file}")
        experiment = Checkpoint.load(load_from_file)
    return experiment

import os
from env import ROOT_DIR as root

def load_experiment_for(mode='train', log_params=None):
    if mode == 'train':
        if log_params.load_from_train is None:
            checkpoint = None
        else:
            checkpoint = load_model(log_params.load_from_train)
        import datetime
        save_dir = os.path.join(root, 
                                f"experiments/{log_params.name}/{datetime.datetime.now().strftime('%Y-%m-%d__%H-%M')}")
        os.makedirs(save_dir, exist_ok=True)
        return checkpoint, save_dir

    elif mode == 'test':
        checkpoint = load_model(log_params.load_from_eval)
        return checkpoint, None
    else:
        raise ValueError(f"Unknown mode {mode}")


def setup_logger(checkpoint_path: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('logger')
    if checkpoint_path is not None:
        file_handler = logging.FileHandler(os.path.join(checkpoint_path, 'log.txt'))
        logger.addHandler(file_handler)
    return logger


def prepare_for_log(results: dict):
    results = {k: (v.detach().cpu().item()
                   if isinstance(v, torch.Tensor)
                   else prepare_for_log(v) if isinstance(v, dict)
    else v)
               for k, v in results.items() if v is not None}
    return results


def print_line(logger: logging.Logger, newline_after: False):
    logger.info('\n' + '-' * 89 + ('\n' if newline_after else ''))



"""
-------------------
SERIALIZATION UTILS
-------------------
"""


class SerializableModule(Module):

    def __init__(self, *args):
        super().__init__()

    def serialize(self):
        return dict(type=self.__class__)

    @staticmethod
    def deserialize(serialized):
        return serialized["type"]
    

class SerializableSequential(Sequential, SerializableModule):

    def __init__(self, *args):
        super().__init__(*args)

    def serialize(self):
        serialized = dict(
            type=self.__class__,
            params=[layer.serialize() for layer in self._modules.values()]
        )
        return serialized

    @staticmethod
    def deserialize(serialized):
        for layer in serialized["params"]:
            if not isinstance(layer, dict):
                print(layer)
        sequential = SerializableSequential(*[
            layer["type"].deserialize(layer)
            for layer in serialized["params"]
        ])
        return sequential


def unpickle(file):
    with open(file, 'rb') as fo:
        _dict = pickle.load(fo)
    return _dict

def softclip(tensor, min):
    
    """
    Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials 
    from https://arxiv.org/pdf/2006.13202.pdf
    
    """
    result_tensor = min + F.softplus(tensor - min)
    return result_tensor


def unit_weight_norm(module: Module, name: str = 'weight', dim: int = 0):
    import torch.nn.utils as utils
    
    class _UnitWeightNorm(torch.nn.Module):
        def __init__(self, dim = 0) -> None:
            super().__init__()
            self.dim = dim

        def forward(self, weight):
            return weight / weight.norm(2, self.dim)
          
    
    _weight_norm = _UnitWeightNorm(dim)
    utils.parametrize.register_parametrization(module, name, _weight_norm, unsafe=True)
    return module


def image_log_rule(epoch):
    if epoch > 1000:
        return (epoch % 1000 == 0)
    elif epoch > 100:
        return (epoch % 100 == 0)
    else:
        return (epoch % 10 == 0)


'''
-------------------
ANALYSIS UTILS
-------------------
'''

from env import ROOT_DIR as root

def save_to_file(data, model_name, file_name, savedir=os.path.join(root,'eval_data/'), subdir='', torch_save=True, device='cpu'):
    '''
        Saves data to a pickle file under the dir: {savedir}/{model_name}/{model_name}_{file_name}.pkl. 

        Args:
            data: data to be saved
            model_name: name of the model
            file_name: name of the file
            savedir: directory to save the file
            subdir: subdirectory to save the file (optional, 'name_of_subdir/')
            torch_save: if True, saves the data as a torch file, else saves as a pickle file
    '''
    if file_name == None:
        file_name = input("Please enter the file name (with desired extension): ")

    file_path = f'{savedir}{subdir}{model_name}/{model_name}_{file_name}'
    os.makedirs(f'{savedir}{subdir}{model_name}', exist_ok=True)

    if isinstance(data, torch.Tensor):
        save_data = data.to(device)
    else:
        save_data = data

    if torch_save:
        file_path += '.pt'
        torch.save(save_data, file_path)
    else:
        file_path += '.pkl'
        with open(file_path, 'wb') as file:
            pickle.dump(save_data, file)
    
    
    print(f"Data saved to {file_path}")

def get_filter_pairs(filters, n_filter_pairs, imsize, reduction):
    paired_filters_list = [torch.zeros(n_filter_pairs, n_filter_pairs, 1, imsize, imsize) for i in range(reduction)]
    for k in range(reduction):
        for i in range(n_filter_pairs):
            for j in range(n_filter_pairs):
                paired_filters_list[k][i, j] = filters[n_filter_pairs*k+i] + filters[n_filter_pairs*k+j]

    return paired_filters_list