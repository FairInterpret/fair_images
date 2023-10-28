import pandas as pd
import numpy as np
import time

# DL Stack
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# metric 
from sklearn import metrics

# Custom
from utils import FaceDataSet, split_data
from arch import ClassifierLayer

if __name__ == "__main__":
    start_time = time.time()

    device_ = 'cpu'

    # if torch.cuda.is_available():
    #     device_ = torch.device('cuda')
    # else:
    #     device_ = 'cpu'

    print(device_)

    # Load up dataset
    data_labs = pd.read_csv('./data/training/label_loader.csv')

    idx_young = int(np.where(data_labs.columns == 'Young')[0])
    idx_male = int(np.where(data_labs.columns == 'Male')[0])

    # Fix seed
    seed_ = 42
    
    # Split
    train, valid, _, _ = split_data(data_labs, random_state=seed_)

    # Initialise Classes and parallelise
    class_net = ClassifierLayer([512,256,64])

    model = nn.DataParallel(class_net,
                            device_ids=[0,1,2,3,4,5,6,7])

    model = model.to(device_)

    train_dataset = FaceDataSet(train,
                                label_idx=idx_young,
                                sens_idx=idx_male)
    
    valid_dataset = FaceDataSet(valid,
                                label_idx=idx_young,
                                sens_idx=idx_male)

    train_loader = DataLoader(train_dataset, 
                              batch_size=256,
                              num_workers=8, 
                              shuffle=True)
    
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=256, 
                              shuffle=False)

    valid_set = []
    epoch_set = []

    loss_crit = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(params=model.parameters(),
                            lr=0.001)
    for epoch in range(10):
        print(epoch)
        running_loss = 0.0
        running_corrects = 0.0
        model = model.train()
        for i, (inputs, labels, input_fair) in enumerate(train_loader):
            inputs, labels, inputs_fair = inputs.to(device_), labels.to(device_), input_fair.to(device_)

            optimizer.zero_grad()

            outputs = model(inputs.squeeze(1), inputs_fair)
            loss = loss_crit(outputs.squeeze().squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()


        valid_preds = []
        valid_labs = []
        model = model.eval()
        for i, (inputs, labels, input_fair) in enumerate(valid_loader):
            inputs, labels, inputs_fair = inputs.to(device_), labels.to(device_), input_fair.to(device_)
    
            with torch.no_grad():
                outputs = model(inputs.squeeze(1), inputs_fair)

            if device_ != 'cpu':
                out_ = torch.sigmoid(outputs.squeeze()).cpu().detach().numpy()
                labs_ = labels.cpu().detach().numpy()
            else:
                out_ = torch.sigmoid(outputs.squeeze()).detach().numpy()
                labs_ = labels.detach().numpy()

            valid_preds.append(out_)
            valid_labs.append(labs_)

        preds_valid = np.concatenate(valid_preds)
        labs_valid = np.concatenate(valid_labs)

        print(labs_valid)

        fpr, tpr, thresholds = metrics.roc_curve(labs_valid,
                                                preds_valid,
                                                pos_label=1)

        auc_valid = metrics.auc(fpr, tpr)
        valid_set.append(auc_valid)
        epoch_set.append(epoch)

        pd.DataFrame({'epoch': epoch_set, 
                      'metric': valid_set}).to_csv(f'data/results/metrics/metric_{seed_}.csv', 
                                                   index=False)


        print(auc_valid)

        # Save metrics and model
        torch.save(model.module.state_dict(),
                   f'data/results/models/model_simple_{seed_}.pt')
        
    