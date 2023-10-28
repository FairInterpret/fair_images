import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from .classification_layers import final_net

def set_basemodel(device='cpu'):
    weights_resnet = ResNet50_Weights.IMAGENET1K_V2
    model_res = resnet50(weights=weights_resnet)

    for param_ in model_res.parameters():
        param_.requires_grad = False

    num_in = model_res.fc.in_features
    model_res.fc = nn.Identity()
    last_layers = final_net(num_in)

    model = nn.Sequential(model_res, last_layers)
    model = model.to(device)

    return model