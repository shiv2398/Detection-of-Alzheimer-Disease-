import torch.nn as nn
import torchvision.models as models

import os 
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchvision
from files.utils import freeze_layer

class vgg_model:
    def __init__(self, num_class):
        self.wt = torchvision.models.VGG16_Weights.DEFAULT
        self.vgg16 = models.vgg16(pretrained=True)
        self.num_class = num_class
    
    def model(self): 
        self.vgg16 = freeze_layer(self.vgg16)
        model = nn.Sequential(
            self.vgg16.features,
            nn.Flatten(),
            nn.Linear(32768, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, self.num_class))
        return model
