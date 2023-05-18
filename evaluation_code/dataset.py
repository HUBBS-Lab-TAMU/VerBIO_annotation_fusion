import numpy as np
import pandas as pd
import os
from torch.utils.data.dataset import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn as nn
from train import train_model
from model import Model
from loss import CCCLoss


class VerBioDataset(Dataset):
    def __init__(self, data_dict, partition):
        
        feature_arr , label_arr = data_dict[partition]['feature'], data_dict[partition]['label']
        self.n_samples = len(feature_arr)
        
        feature_lens = []
        for feature in feature_arr:
            feature_lens.append(len(feature))
        self.feature_lens = torch.tensor(feature_lens)
        
        if partition == 'train':
            self.features = pad_sequence([torch.tensor(feature, dtype=torch.float) for feature in feature_arr],
                                         batch_first=True)
            self.labels = pad_sequence([torch.tensor(label, dtype=torch.float) for label in label_arr], batch_first=True)
            
        else:
            self.features = [torch.tensor(feature, dtype=torch.float) for feature in feature_arr]
            self.labels = [torch.tensor(label, dtype=torch.float) for label in label_arr]

        

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        feature = self.features[idx]
        feature_len = self.feature_lens[idx]
        label = self.labels[idx]


        sample = feature, feature_len, label
        return sample

