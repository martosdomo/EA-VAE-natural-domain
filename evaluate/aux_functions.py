import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

def load(name, analysis, path='../eval_data/'):
    file_name = f'{path}{name}/{name}_{analysis}'
    if os.path.exists(file_name + '.pkl'):
        file_name += '.pkl'
        with open(file_name, 'rb') as f:
            file = pickle.load(f)
    elif os.path.exists(file_name + '.pt'):
        file_name += '.pt'
        file = torch.load(file_name)
    else:
        raise FileNotFoundError(f'File {file_name} not found.')
    print('File loaded from', file_name)
    return file

def save_fig(path, format='png', dpi=300):
    '''
        Save the current figure to the path.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path+f'.{format}', format=format, bbox_inches='tight')


def get_active_dims(params, threshold=0.5, passive=False, only_mask=False):
    '''
        Get active dims from the latents based on threshold. 
        Params should be a list of [model name, model type, mean or std].

        Args:
            params (list): model name, model type, mean or std
            threshold (float): threshold of standard deviation of mean
            passive (bool): if true, return both passive and active latent dimensions as (active, passive)
            only_mask (bool): if true, return only the mask of active latent dimensions
    '''
    mean_data = load(params[0], 'posterior_mean')
    if params[1] != 'posterior_mean':
        return_data = load(params[0], params[1])
    else: 
        return_data = mean_data
    std_of_means = torch.std(mean_data, dim=0)
    active_dims = np.where(np.array(std_of_means) > threshold, True, False)
    if 'EAVAE' in params[0]:
        active_dims[-1] = False
    print(params, '# of active dims: ', active_dims.sum())
    if only_mask:
        return active_dims
    elif passive:
        return return_data[:, active_dims==True], return_data[:, active_dims==0]
    else:
        return return_data[:, active_dims==True]