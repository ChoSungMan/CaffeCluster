import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.autograd import Variable

import os

from PIL import Image
from mobilenetv2 import MobileNetV2
from sklearn.decomposition import PCA

import numpy as np

def get_vector(image_name):
    weight_path = 'weight/mobilenet_v2.pth.tar'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = Image.open(image_name)

    model = MobileNetV2()
    model.load_state_dict(torch.load(weight_path, map_location=lambda storage, loc: storage))
    
    print(model)

    feature_net = model.features.to(device)
    
    feature_net_5 = feature_net[:-14]
    feature_net_10 = feature_net[:-9]
    feature_net_15 = feature_net[:-4]
    
    model.eval()

    transforms = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.229, 0.224, 0.225])
        ])
    
    image_tensor = transforms(image).unsqueeze(0)
    
    image_tensor = image_tensor.to(device)
    
    output_5 = feature_net_5(image_tensor)
    output_5 = output_5.view(output_5.size(0), 32*28*28)
    output_5 = nn.Linear(32*28*28, 1024)(output_5)

    output_10 = feature_net_10(image_tensor)
    output_10 = output_10.view(output_10.size(0), 64*14*14)
    output_10 = nn.Linear(64*14*14, 1024)(output_10)

    output_15 = feature_net_15(image_tensor)
    output_15 = output_15.view(output_15.size(0), 160*7*7)
    output_15 = nn.Linear(160*7*7, 1024)(output_15)

    output = (output_5 + output_10 + output_15) / 3 
    print(output.shape)

get_vector('test.jpg')
