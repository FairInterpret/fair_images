import os
import numpy as np
import pandas 
import pickle

import torch
from torch.utils.data import Dataset
from PIL import Image

def split_data(data,
               random_state,
               sample_frac_test = 0.2, 
               sample_frac_calib = 0.2, 
               sample_frac_valid = 0.2,
               save_space='data/results/split_idx/'):
    # Train
    train_tmp = data.sample(frac=(1-(sample_frac_calib+sample_frac_test)), 
                            random_state=random_state)
    
    train = train_tmp.sample(frac=(1-sample_frac_valid),
                             random_state=random_state)
    valid = train_tmp.loc[~train_tmp.img_id.isin(train.img_id), :]

    # Calib/Test
    test_tmp = data.loc[~data.img_id.isin(train_tmp.img_id), :]
    
    test = test_tmp.sample(frac=(1-((sample_frac_calib+sample_frac_calib)/2)), 
                           random_state=random_state)
    calib = test_tmp.loc[~test_tmp.img_id.isin(test.img_id), :]

    save_dict = {
        'train': train.img_id.to_list(), 
        'valid': valid.img_id.to_list(), 
        'calib': calib.img_id.to_list(), 
        'test': test.img_id.to_list()
    }
    save_string = save_space + f'splits_seed_{random_state}.pkl'
    with open(save_string, 'wb') as con_:
        pickle.dump(save_dict, con_)

    return train, valid, calib, test

class FaceDataSet(Dataset):
    def __init__(self,
                 labels,
                 label_idx,
                 sens_idx=None,
                 root_dir='./data/training/preprocessed_faces',
                 ):
        self.labels = labels
        self.root_dir = root_dir
        self.label_idx = label_idx
        self.sens_idx = sens_idx
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        
        fullname = os.path.join(self.root_dir,
                                f'{img_name}')
        
        image = torch.load(fullname)

        labels = np.float32(np.array(self.labels.iloc[idx, self.label_idx]))
        labels_sens = np.float32(np.array(self.labels.iloc[idx, self.sens_idx]))

        return [image, labels, labels_sens]

        
class FaceDataSetAdjustment(Dataset):
    def __init__(self,
                 labels,
                 label_idx,
                 sens_idx=None,
                 root_dir='./data/training/adjustment_faces/',
                 ):
        self.labels = labels
        self.root_dir = root_dir
        self.label_idx = label_idx
        self.sens_idx = sens_idx
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        
        fullname = os.path.join(self.root_dir,
                                f'{img_name}')
        
        image = torch.load(fullname)

        labels = np.float32(np.array(self.labels.iloc[idx, self.label_idx]))
        labels_sens = np.float32(np.array(self.labels.iloc[idx, self.sens_idx]))

        return [image, labels, labels_sens]


