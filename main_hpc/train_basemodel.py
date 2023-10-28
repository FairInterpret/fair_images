import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import pandas as pd
import csv

from arch import final_net
from utils import FaceDataSet

if torch.cuda.is_available():
    device_ = torch.device('cuda')
else:
    device_ = 'cpu'

# Set up model
weights_resnet = ResNet50_Weights.IMAGENET1K_V2
model_res = resnet50(weights=weights_resnet)

# Set up class layer
num_in = model_res.fc.out_features
last_layers = final_net(num_in=num_in)

for param in model_res.parameters():
    param.requires_grad = False
model_res = model_res.to(device_)

last_layers = last_layers.to(device_)
model = nn.Sequential(model_res, last_layers)
model = model.to(device_)

with open('data/raw/annotations/list_attr_celeba.txt') as csv_file:
            data = list(csv.reader(csv_file, delimiter=' ', skipinitialspace=True))

data_labs = pd.DataFrame(data[2:])
data_labs.columns = ['img_id'] + data[1][:-1]
data_labs.iloc[:,25] = np.where(data_labs.iloc[:,25] == '1', 1, 0)
data_labs.iloc[:,21] = np.where(data_labs.iloc[:,21] == '1', 1, 0)

# Run
preprocessor_ = weights_resnet.transforms()
train_ds = FaceDataSet(data_labs, sens_idx=21,transform=preprocessor_)
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)

# Setup
loss_crit = torch.nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(params=model.parameters(), lr=0.001)

for epoch in range(20):
    for inputs, labels, _ in train_dl:       
        inputs, labels = Variable(inputs.to(device_)), Variable(labels.to(device_)) 
        optimizer.zero_grad()

        outputs = model(inputs)
        outputs = outputs.squeeze()

        loss = loss_crit(outputs, labels)

        loss.backward()
        optimizer.step()

torch.save(model.state_dict(),
           'data/test_model.torch')
