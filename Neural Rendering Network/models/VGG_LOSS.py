import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import functools
from torchvision import models
from collections import namedtuple

class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
        #X=normalize(X)
        X = 0.5 * (X + 1.0) # map to [0,1]

        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram



class VGGLOSS(torch.nn.Module):
    def __init__(self):
        super(VGGLOSS, self).__init__()
        self.model = VGG16()
        self.criterionL2 = torch.nn.MSELoss(reduction='mean')

    def forward(self, fake, target, content_weight = 1.0, style_weight = 1.0):
        vgg_fake = self.model(fake)
        vgg_target = self.model(target)

        content_loss = self.criterionL2(vgg_target.relu2_2, vgg_fake.relu2_2)

        # gram_matrix
        gram_style = [gram_matrix(y) for y in vgg_target]
        style_loss = 0.0
        for ft_y, gm_s in zip(vgg_fake, gram_style):
            gm_y = gram_matrix(ft_y)
            style_loss += self.criterionL2(gm_y, gm_s)

        total_loss = content_weight * content_loss + style_weight * style_loss
        return total_loss