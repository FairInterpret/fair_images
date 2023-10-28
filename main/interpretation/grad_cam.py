"""
Adaptedd from 
https://github.com/Caoliangjie/pytorch-gradcam-resnet50/blob/master/grad-cam.py
to work with the models used in the paper
"""
from typing import Any
import cv2
import torch
import numpy as np

class FeatureExtractor:
    def __init__(self,
                 model,
                 target_layers):
        self.model_ = model[0]
        self.classifier_mod = model[1]
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)
    
    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model_._modules.items():
            if name == 'fc':
                x = module(x.reshape(1,-1))
            else:
                x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x
    
    
class ModelOutputs():
	def __init__(self,
	      model, target_layers):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model, target_layers)
	
	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)

		output = self.model[1](output)

		return target_activations, output

class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()

		self.extractor = ModelOutputs(self.model,
									  target_layer_names)

	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, index = 0):
		features, output = self.extractor(input)

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = torch.Tensor(torch.from_numpy(one_hot))
		one_hot.requires_grad = True
	
		one_hot = torch.sum(one_hot * output)

		self.model.zero_grad()##
		#self.model.zero_grad()
		one_hot.backward(retain_graph=True)##

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
		#print('grads_val',grads_val.shape)
		target = features[-1]
		target = target.data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		#print('weights',weights.shape)
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)
		#print('cam',cam.shape)
		#print('features',features[-1].shape)
		#print('target',target.shape)
		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (224, 224))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam