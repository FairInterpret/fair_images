import torch
import torch.nn as nn
import torch.nn.functional as F

class final_net(nn.Module):
    def __init__(self, num_in=2048):
        super(final_net, self).__init__()
        self.fc = nn.Linear(num_in, 128)
        self.cut_1 = nn.Linear(128, 32)
        self.out_1 = nn.Linear(32, 1)
            
    def forward(self, x):
        x = F.tanh(self.fc(x))
        x = F.tanh(self.cut_1(x))
        out = self.out_1(x)

        return out
    

class ClassifierLayer(nn.Module):
    def __init__(self, 
                 hidden_layer_architecture,
                 num_in=2048,
                 activation='relu',
                 dropout=0.1):
        super().__init__()
        self.architecture = hidden_layer_architecture
        self.input_dim = num_in + 1
        self.activation = activation
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self._set_architecture()

    def _set_architecture(self):
        current_dim = self.input_dim

        for layer_ in self.architecture:
            self.layers.append(nn.Linear(current_dim, layer_))
            self.layers.append(nn.Dropout(self.dropout))
            current_dim = layer_
        
        self.output = nn.Linear(current_dim, 1)

    def forward(self, input_img, input_fair):
        x = torch.cat((input_img, input_fair.view(-1,1)),
                      dim=1)
        if self.activation == 'relu':
            for lay_ in self.layers:
                if isinstance(lay_, torch.nn.modules.linear.Linear):
                    x = F.relu(lay_(x))
                else:
                    x = lay_(x)
        else:
            x = lay_(x)

        x = self.output(x)

        return x
    
class ClassifierLayerNoSens(nn.Module):
    def __init__(self, 
                 hidden_layer_architecture,
                 num_in=2048,
                 activation='relu',
                 dropout=0.1):
        super().__init__()
        self.architecture = hidden_layer_architecture
        self.input_dim = num_in
        self.activation = activation
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self._set_architecture()

    def _set_architecture(self):
        current_dim = self.input_dim

        for layer_ in self.architecture:
            self.layers.append(nn.Linear(current_dim, layer_))
            self.layers.append(nn.Dropout(self.dropout))
            current_dim = layer_
        
        self.output = nn.Linear(current_dim, 1)

    def forward(self, input_img):
        x = input_img
        if self.activation == 'relu':
            for lay_ in self.layers:
                if isinstance(lay_, torch.nn.modules.linear.Linear):
                    x = F.relu(lay_(x))
                else:
                    x = lay_(x)
        else:
            x = lay_(x)

        x = self.output(x)

        return x