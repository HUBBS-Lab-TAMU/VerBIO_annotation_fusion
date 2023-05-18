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


# def load_data(label_type):

#     pid_list = [1,3,4,5,6,8,9,11,13,14]
    
#     data_dict = {'train':{'feature': [], 'label': []},
#                  'val': {'feature': [], 'label': []}}
#     for i in range(1,74):
#         pid =f'P{str(i).zfill(3)}'
        
#         try:
#             feature = pd.read_excel(f'./feature/PRE_{pid}_eGemaps.xlsx', engine = 'openpyxl')
#             label = pd.read_excel(f'./label_sk_random/{pid}_EM.xlsx', engine = 'openpyxl')
#             # label = pd.read_excel(f'../SkNorm/EM_sknorm/{pid}_EM.xlsx', engine = 'openpyxl')
#         except:
#             continue
        
#         print(f'{pid} -- Feature: {len(feature)}, Label: {len(label)}')
        
#         t = min(len(feature), len(label))
    
#         if i<57:
#             data_dict['train']['feature'].append(feature.iloc[:t].to_numpy())
#             data_dict['train']['label'].append(label.iloc[:t][label_type].to_numpy().reshape(-1,1))
#         else:
#             data_dict['val']['feature'].append(feature.iloc[:t].to_numpy())
#             data_dict['val']['label'].append(label.iloc[:t][label_type].to_numpy().reshape(-1,1))
        
#         print(f'new {pid} -- Feature: {len(feature)}, Label: {len(label)}')
#         # break
#     return data_dict


# # verbio_data = VerBioDataset(data_dict, partition='train')
# # data_loader = {}

# # dt = VerBioDataset(data_dict, partition='train')
# # data_loader['train'] = torch.utils.data.DataLoader(dt, batch_size=3, shuffle=False)
# # dt = VerBioDataset(data_dict, partition='val')
# # data_loader['val'] = torch.utils.data.DataLoader(dt, batch_size=1, shuffle=False)



def load_data(label_type, label_folder):
    
    data_dict = {'train':{'feature': [], 'label': []},
                 'val': {'feature': [], 'label': []}}

    if label_folder == 'Selective':
        label_path = './label_Normal_selective'
    elif label_folder == 'Mean':
        label_path = './label_Normal'
    else:
        label_path = './label_baseline'

    feature_dir = './feature_split/train'
    label_dir = f'{label_path}/train'


    print("######Training Data########")
    for fid in os.listdir(feature_dir):
        [subject_id, session_id, tmp] = fid.split('_')

        feature_fid = f'{feature_dir}/{subject_id}_{session_id}_eGemaps.xlsx'
        label_fid = f'{label_dir}/{subject_id}_{session_id}_annotation.xlsx'


        feature = pd.read_excel(feature_fid, engine = 'openpyxl')
        label = pd.read_excel(label_fid, engine = 'openpyxl')

        
        
        
        t = min(len(feature), len(label))
    
        
        data_dict['train']['feature'].append(feature.iloc[:t].to_numpy())
        data_dict['train']['label'].append(label.iloc[:t][label_type].to_numpy().reshape(-1,1))

        # print(f'{subject_id}_{session_id} -- Feature: {len(feature)}, Label: {len(label)}')


    
    feature_dir = './feature_split/test'
    label_dir = f'{label_path}/test'


    print("######Test Data########")
    for fid in os.listdir(feature_dir):
        [subject_id, session_id, tmp] = fid.split('_')

        feature_fid = f'{feature_dir}/{subject_id}_{session_id}_eGemaps.xlsx'
        label_fid = f'{label_dir}/{subject_id}_{session_id}_annotation.xlsx'


        feature = pd.read_excel(feature_fid, engine = 'openpyxl')
        label = pd.read_excel(label_fid, engine = 'openpyxl')

        
        
        
        t = min(len(feature), len(label))
    
        
        data_dict['val']['feature'].append(feature.iloc[:t].to_numpy())
        data_dict['val']['label'].append(label.iloc[:t][label_type].to_numpy().reshape(-1,1))
        
        # print(f'{subject_id}_{session_id} -- Feature: {len(feature)}, Label: {len(label)}')

        # break
    return data_dict