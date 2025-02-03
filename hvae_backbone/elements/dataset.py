from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset as TorchDataset

from ..elements.data_preproc import default_transform


class DataSet(ABC):

    """
    Abstract class for datasets
    Inherit from this class to create a new dataset in /data
    """

    def __init__(self,
                 train_transform=default_transform,
                 val_transform=default_transform,
                 test_transform=default_transform,
                 with_labels=False):
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.val_transform = val_transform
        self.with_labels = with_labels
        
        data = self.load()
        
        if not with_labels:
            train_data, val_data, test_data = data

            if isinstance(train_data, TorchDataset):
                self.train_set = train_data
                self.val_set = val_data
                self.test_set = test_data
            else:
                self.train_set = FunctionalDataset(train_data, transform=self.train_transform)
                self.val_set = FunctionalDataset(val_data, transform=self.val_transform)
                self.test_set = FunctionalDataset(test_data, transform=self.test_transform)
        else:
            if isinstance(data[0], TorchDataset):
                self.train_set = data[0]
                self.val_set = data[1]
                self.test_set = data[2]
            else:
                train_data, val_data, test_data, train_labels, val_labels, test_labels = data
                self.train_set = FunctionalDataset(train_data, train_labels, self.train_transform)
                self.val_set = FunctionalDataset(val_data, val_labels, self.val_transform)
                self.test_set = FunctionalDataset(test_data, test_labels, self.test_transform)

    def get_train_loader(self, batch_size=128):
        return torch.utils.data.DataLoader(dataset=self.train_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           drop_last=True)

    def get_val_loader(self, batch_size=128):
        return torch.utils.data.DataLoader(dataset=self.val_set,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           pin_memory=True,
                                           drop_last=True)

    def get_test_loader(self, batch_size=128):
        if not self.with_labels:
            return torch.utils.data.DataLoader(dataset=self.test_set,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               drop_last=True)

    @abstractmethod
    def load(self):
        """
        Load data from disk
        :return: train, val, test
        """
        pass


class FunctionalDataset(TorchDataset):

    """
    Dataset class for functional datasets (train, validation, test)
    """
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        from .. import params
        item = self.data[idx]
        #if self.transform:
        #    item = self.transform(item)
        item = torch.tensor(item).view(params.data_params.shape).to(torch.float32)
        if self.labels is None:
            return item

        label = self.labels[idx]
        label = torch.tensor(label)
        return item, label

    def __len__(self):
        return self.data.shape[0]
