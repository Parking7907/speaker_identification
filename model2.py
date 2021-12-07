import torch
import torch.nn as nn
import torch.nn.functional as F
#from model_util import embedding_network
import pdb
from torchvision import models
import numpy as np


class MyNet(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.backbone = models.resnet34(pretrained=True)
        #pdb.set_trace()
        self.backbone.fc = torch.nn.Identity()
        self.backbone.conv1 = nn.Conv2d(1,64,7,2,3)
        #freeze_parameters(self.backbone, train_fc=True)
        #self.FC1 = nn.Linear(512,out_features)
    def forward(self, rgb):
        B = rgb.shape[0]
        rgb = torch.unsqueeze(rgb, 1)
        #pdb.set_trace()
        out = self.backbone(rgb)

        
        #out = self.FC1(out)
        return out