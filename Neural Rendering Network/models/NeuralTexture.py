import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import functools
from PIL import Image
from util import util
from torchvision import models
from collections import namedtuple

################
###  HELPER  ###
################
from BaselModel.basel_model import *
INVALID_UV = -1.0



#####################################
########   static texture   #########
#####################################
class StaticNeuralTexture(nn.Module):
    def __init__(self, texture_dimensions, texture_features):
        super(StaticNeuralTexture, self).__init__()
        self.texture_dimensions = texture_dimensions #256 #texture dimensions
        self.out_ch = texture_features # output feature, after evaluating the texture


        self.register_parameter('data', torch.nn.Parameter(torch.randn(1, self.out_ch, self.texture_dimensions, self.texture_dimensions, requires_grad=True)))
        ####

    def forward(self, expressions, audio_features, uv_inputs):     
        b = audio_features.shape[0] # batchsize
        if b != 1:
            print('ERROR: NeuralTexture forward only implemented for batchsize==1')
            exit(-1)
        uvs = torch.stack([uv_inputs[:,0,:,:], uv_inputs[:,1,:,:]], 3)
        return torch.nn.functional.grid_sample(self.data, uvs, mode='bilinear', padding_mode='border')


#####################################
########   audio texture   ##########
#####################################
class DynamicNeuralTextureAudio(nn.Module):
    def __init__(self, texture_dimensions, texture_features_intermediate, texture_features):
        super(DynamicNeuralTextureAudio, self).__init__()
        self.texture_features_intermediate = texture_features_intermediate #16 #features stored in texture
        self.texture_dimensions = texture_dimensions #256 #texture dimensions
        self.out_ch = texture_features # output feature, after evaluating the texture

        # input 16 x 29
        self.convNet = nn.Sequential(
            nn.Conv2d(29, 32, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), #  29 x 16 x 1 => 32 x 8 x 1
            nn.LeakyReLU(0.02, True),
            nn.Conv2d(32, 32, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 32 x 8 x 1 => 32 x 4 x 1
            nn.LeakyReLU(0.02, True),
            nn.Conv2d(32, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 32 x 4 x 1 => 64 x 2 x 1
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 64 x 2 x 1 => 64 x 1 x 1
            nn.LeakyReLU(0.2, True),
        )
        conv_output_size = 64
        self.fullNet = nn.Sequential(
            nn.Linear(in_features = conv_output_size, out_features=128, bias = True),
            nn.LeakyReLU(0.02),
            nn.Linear(in_features = 128, out_features=64, bias = True),
            nn.LeakyReLU(0.02),
            nn.Linear(in_features = 64, out_features=self.out_ch*4*4*self.texture_features_intermediate, bias = True),          
            nn.Tanh()
            )
 
        self.register_parameter('data', torch.nn.Parameter(torch.randn(1, self.texture_features_intermediate, self.texture_dimensions, self.texture_dimensions, requires_grad=True)))
        ####

    def forward(self, expressions, audio_features, uv_inputs):     
        b = audio_features.shape[0] # batchsize
        if b != 1:
            print('ERROR: NeuralTexture forward only implemented for batchsize==1')
            exit(-1)
        # b x 1 x 16 x 29 --> transpose
        audio_features = torch.transpose(audio_features, 1, 3)
        audio_conv_res = self.convNet( audio_features )
        conv_filter = torch.reshape(  self.fullNet( torch.reshape( audio_conv_res, (b,1,-1))), (self.out_ch,self.texture_features_intermediate,4,4))
        self.tex_eval = nn.functional.conv2d(self.data, conv_filter, stride=1, padding=2)
        uvs = torch.stack([uv_inputs[:,0,:,:], uv_inputs[:,1,:,:]], 3)

        return torch.nn.functional.grid_sample(self.tex_eval, uvs, mode='bilinear', padding_mode='border')


#####################################
######   expression texture   #######
#####################################

class DynamicNeuralTextureExpression(nn.Module):
    def __init__(self, texture_dimensions, texture_features_intermediate, texture_features):
        super(DynamicNeuralTextureExpression, self).__init__()
        self.texture_features_intermediate = texture_features_intermediate #16 #features stored in texture
        self.texture_dimensions = texture_dimensions #256 #texture dimensions
        self.out_ch = texture_features # output feature, after evaluating the texture

        # input: 76
        input_size = 76
        self.fullNet = nn.Sequential(
            nn.Linear(in_features = input_size, out_features=128, bias = True),
            nn.LeakyReLU(0.02),
            nn.Linear(in_features = 128, out_features=64, bias = True),
            nn.LeakyReLU(0.02),
            nn.Linear(in_features = 64, out_features=self.out_ch*4*4*self.texture_features_intermediate, bias = True),          
            nn.Tanh()
            )
   
        self.register_parameter('data', torch.nn.Parameter(torch.randn(1, self.texture_features_intermediate, self.texture_dimensions, self.texture_dimensions, requires_grad=True)))
        ####

    def forward(self, expressions, audio_features, uv_inputs):
        b = expressions.shape[0] # batchsize
        if b != 1:
            print('ERROR: NeuralTexture forward only implemented for batchsize==1')
            exit(-1)
        conv_filter = torch.reshape(  self.fullNet( torch.reshape( expressions, (1,1,-1))), (self.out_ch,self.texture_features_intermediate,4,4))
        self.tex_eval = nn.functional.conv2d(self.data, conv_filter, stride=1, padding=2)
        uvs = torch.stack([uv_inputs[:,0,:,:], uv_inputs[:,1,:,:]], 3)

        return torch.nn.functional.grid_sample(self.tex_eval, uvs, mode='bilinear', padding_mode='border')

