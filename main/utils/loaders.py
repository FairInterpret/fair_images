import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class FaceDataSetAlign(Dataset):
    def __init__(self,
                 labels,
                 preprocessor_=None,
                 label_idx=25,
                 root_dir='./data/raw/imgs/img_align_celeba/',
                 sens_idx=None,
                 subset=False,
                 transform=None):
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
        self.label_idx = label_idx
        self.preprocessor = preprocessor_
        self.sens_idx = sens_idx
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        fullname = os.path.join(self.root_dir, img_name)
        image = Image.open(fullname)
        # Preprocess images with resnet preprocessor
        if self.preprocessor:
            image = self.preprocessor(image)
            
        labels = np.float32(np.array(self.labels.iloc[idx, self.label_idx]))

        if self.transform:
            image = self.transform(image)
        if self.sens_idx:
            labels_sens = np.float32(np.array(self.labels.iloc[idx, self.sens_idx]))
            return [image, labels, labels_sens]
        else:
            return [image, labels]


class FaceDataSet(Dataset):
    def __init__(self,
                 labels,
                 preprocessor_=None,
                 label_idx=25,
                 root_dir='./data/raw/imgs/img_align_celeba/',
                 sens_idx=None,
                 subset=False,
                 transform=None):
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
        self.label_idx = label_idx
        self.preprocessor = preprocessor_
        self.sens_idx = sens_idx
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        fullname = os.path.join(self.root_dir, img_name)
        image = Image.open(fullname)
        # Preprocess images with resnet preprocessor
        if self.preprocessor:
            image = self.preprocessor(image)
            
        labels = np.float32(np.array(self.labels.iloc[idx, self.label_idx]))

        if self.transform:
            image = self.transform(image)
        if self.sens_idx:
            labels_sens = np.float32(np.array(self.labels.iloc[idx, self.sens_idx]))
            return [image, labels, labels_sens]
        else:
            return [image, labels]


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


class FaceDataSetPrep(Dataset):
    def __init__(self,
                 labels,
                 label_idx,
                 sens_idx=None,
                 root_dir='./data/prepared/preprocessed_faces',
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
    
