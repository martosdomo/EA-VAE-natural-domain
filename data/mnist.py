import torch
from torchvision import datasets, transforms
from hvae_backbone.elements.dataset import DataSet
import numpy as np
import medmnist
from torch.nn import functional as F

def z_score_normalize(data, eps=1e-8):
    mean = data.mean()
    std = data.std()
    data = (data - mean) / (std + eps)
    return data.squeeze()

class BaseMNIST(DataSet):
    def __init__(self, dataset_class, download_path, z_score=False, resize=None):
        self.dataset_class = dataset_class
        self.download_path = download_path
        self.z_score = z_score
        self.resize = resize
        super(BaseMNIST, self).__init__()

    def load(self):
        transform = transforms.Compose([transforms.ToTensor()])
        
        try: # for torchvision.datasets based
            trainset = self.dataset_class(self.download_path, download=True, train=True, transform=transform)
            testset = self.dataset_class(self.download_path, download=True, train=False, transform=transform)
        except: # for medmnist based
            trainset = self.dataset_class(download=True, split='train', transform=transform)
            testset = self.dataset_class(download=True, split='test', transform=transform)

        trainset = np.stack(list(map(lambda x: x[0], trainset)))  # Discard labels
        testset = np.stack(list(map(lambda x: x[0], testset))) # Discard labels

        if self.resize is not None:
            trainset = F.interpolate(torch.tensor(trainset), size=self.resize, mode='bilinear').reshape(-1, self.resize[0]*self.resize[1])
            testset = F.interpolate(torch.tensor(testset), size=self.resize, mode='bilinear').reshape(-1, self.resize[0]*self.resize[1])

        if self.z_score:
            trainset = z_score_normalize(trainset)
            testset = z_score_normalize(testset)

        val_size = len(testset) // 2
        valset = testset[:val_size]
        testset = testset[val_size:]

        return trainset, valset, testset
    

class MNIST(BaseMNIST):
    def __init__(self, z_score=False, resize=None):
        super(MNIST, self).__init__(datasets.MNIST, '~/.pytorch/MNIST_data/', z_score, resize)


class FashionMNIST(BaseMNIST):
    def __init__(self, z_score=False, resize=None):
        super(FashionMNIST, self).__init__(datasets.FashionMNIST, '~/.pytorch/FashionMNIST_data/', z_score, resize)


class ChestMNIST(BaseMNIST):
    def __init__(self, z_score=False, resize=None):
        super(ChestMNIST, self).__init__(medmnist.ChestMNIST, '~/.pytorch/ChestMNIST_data/', z_score, resize)
