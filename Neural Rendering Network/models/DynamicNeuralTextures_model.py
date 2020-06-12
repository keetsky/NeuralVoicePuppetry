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


from . import VGG_LOSS
from . import UNET
from . import NeuralTexture

################
###  HELPER  ###
################
from BaselModel.basel_model import *
INVALID_UV = -1.0


def define_Texture(opt, gpu_ids=[]):
    net = None

    if opt.textureModel == 'DynamicNeuralTextureAudio':
        net = NeuralTexture.DynamicNeuralTextureAudio(texture_dimensions=opt.tex_dim, texture_features_intermediate=opt.tex_features_intermediate, texture_features=opt.tex_features)
    elif opt.textureModel == 'DynamicNeuralTextureExpression':
        net = NeuralTexture.DynamicNeuralTextureExpression(texture_dimensions=opt.tex_dim, texture_features_intermediate=opt.tex_features_intermediate, texture_features=opt.tex_features)
    elif opt.textureModel == 'StaticNeuralTexture':
        net = NeuralTexture.StaticNeuralTexture(texture_dimensions=opt.tex_dim, texture_features=opt.tex_features)
    
    return networks.init_net(net, opt.init_type, opt.init_gain, gpu_ids)

def define_TextureDecoder(renderer, n_feature, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = networks.get_norm_layer(norm_type=norm)
    N_OUT = 3
    #renderer=='UNET_5_level'
    net = UNET.UnetRenderer(renderer, n_feature, N_OUT, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    return networks.init_net(net, init_type, init_gain, gpu_ids)

def define_Inpainter(renderer, n_feature, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = networks.get_norm_layer(norm_type=norm)
    N_OUT = 3
    #renderer=='UNET_5_level'
    net = UNET.UnetRenderer(renderer, n_feature, N_OUT, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    return networks.init_net(net, init_type, init_gain, gpu_ids)



class DynamicNeuralTexturesModel(BaseModel):
    def name(self):
        return 'DynamicNeuralTexturesModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        #parser.set_defaults(norm='batch', netG='unet_256')
        parser.set_defaults(norm='instance', netG='unet_256')
        parser.set_defaults(dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, no_lsgan=True)
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        self.trainRenderer = not opt.fix_renderer

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_total', 'G_L1_Rendering', 'G_VGG_Rendering', 'G_GAN']

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.isTrain:
            self.visual_names = ['input_uv', 'fake', 'target']
        else:
            self.visual_names = ['input_uv', 'fake', 'target']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['texture', 'texture_decoder', 'inpainter' ,'netD']
        else:  # during test time, only load Gs
            self.model_names = ['texture', 'texture_decoder', 'inpainter']

        # load/define networks
        self.texture = define_Texture(opt,  self.gpu_ids)
        self.texture_decoder = define_TextureDecoder(opt.rendererType, opt.tex_features+3, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.inpainter = define_Inpainter(opt.rendererType, 6, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        # optimizer
        self.loss_G_GAN = 0.0

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss(reduction='mean')
            self.criterionL1Smooth = torch.nn.SmoothL1Loss(reduction='mean')
            self.criterionL2 = torch.nn.MSELoss(reduction='mean')

            if self.opt.lossType == 'VGG':
                self.vggloss = VGG_LOSS.VGGLOSS().to(self.device)

            # initialize optimizers
            self.optimizers = []

            self.optimizer_texture_decoder = torch.optim.Adam(self.texture_decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_texture_decoder)

            self.optimizer_inpainter = torch.optim.Adam(self.inpainter.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_inpainter)

            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)

            self.optimizer_T = torch.optim.Adam(self.texture.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_T)


    def maskErosion(self, mask, erosionFactor):
        offsetY = int(erosionFactor * 40)
        # throat
        mask2 = mask[:,:,0:-offsetY,:]
        mask2 = torch.cat([torch.ones_like(mask[:,:,0:offsetY,:]), mask2], 2)
        # forehead
        offsetY = int(erosionFactor * 8) #<<<<
        mask3 = mask[:,:,offsetY:,:]
        mask3 = torch.cat([mask3, torch.ones_like(mask[:,:,0:offsetY,:])], 2)
        mask = mask * mask2 * mask3

        offsetX = int(erosionFactor * 15)
        # left
        mask4 = mask[:,:,:,0:-offsetX]
        mask4 = torch.cat([torch.ones_like(mask[:,:,:,0:offsetX]), mask4], 3)
        # right
        mask5 = mask[:,:,:,offsetX:]
        mask5 = torch.cat([mask5,torch.ones_like(mask[:,:,:,0:offsetX])], 3)
        return mask * mask4 * mask5

    def set_input(self, input):
        self.target = input['TARGET'].to(self.device)
        self.input_uv = input['UV'].to(self.device)             
        self.intrinsics = input['intrinsics']
        self.extrinsics = input['extrinsics']
        self.expressions = input['expressions'].cuda()

        self.image_paths = input['paths']
        self.audio_features = input['audio_deepspeech'].cuda()

        ## in training phase introduce some noise
        #if self.isTrain:
        #    if self.opt.input_noise_augmentation:
        #        audio_noise = torch.randn_like(self.audio_features)*0.05 # check magnitude of noise
        #        self.audio_features = self.audio_features + audio_noise


    def forward(self, alpha=1.0):
        # background
        mask = (self.input_uv[:,0:1,:,:] == INVALID_UV) & (self.input_uv[:,1:2,:,:] == INVALID_UV)
        mask = self.maskErosion(mask, self.opt.erosionFactor)
        mask = torch.cat([mask,mask,mask], 1)
        self.background = torch.where(mask, self.target, torch.zeros_like(self.target))

     
        # loop over batch elements
        batch_size = self.target.shape[0]
        self.features = []
        self.intermediate_fake = []
        self.fake = []
        for b in range(0,batch_size):
            feat =  self.texture(self.expressions[b:b+1], self.audio_features[b:b+1], self.input_uv[b:b+1])
            self.features.append(feat)

            intermediate_fake = self.texture_decoder(self.expressions[b:b+1], self.audio_features[b:b+1], feat, self.background[b:b+1])
            self.intermediate_fake.append(intermediate_fake)

            fake = self.inpainter(self.expressions[b:b+1], self.audio_features[b:b+1], intermediate_fake, self.background[b:b+1])
            self.fake.append(fake)


        self.features = torch.cat(self.features, dim=0)
        self.intermediate_fake = torch.cat(self.intermediate_fake, dim=0)
        self.fake = torch.cat(self.fake, dim=0)

        self.fake = torch.where(mask, self.background, self.fake)
    

    def backward_D(self):
        mask = ( (self.input_uv[:,0:1,:,:] != INVALID_UV) | (self.input_uv[:,1:2,:,:] != INVALID_UV) )
        mask = torch.cat([mask,mask,mask], 1)
        def masked(img):
            return torch.where(mask, img, torch.zeros_like(img))

        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.input_uv, masked(self.fake)), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.input_uv, masked(self.target)), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self, epoch):

        mask = ( (self.input_uv[:,0:1,:,:] != INVALID_UV) | (self.input_uv[:,1:2,:,:] != INVALID_UV) )
        sum_mask = torch.sum(mask)
        d = mask.shape[1]
        mask_weight = (d*d) / sum_mask
        mask = torch.cat([mask,mask,mask], 1)
        def masked(img):
            return torch.where(mask, img, torch.zeros_like(img))

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.input_uv, masked(self.fake)), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * 0.0 # disabled GAN


        # Second, G(A) = B
        self.loss_G_L1_Rendering = 0.0
        self.loss_G_L1_Rendering = 1.0 * self.criterionL1(masked(self.features[:,0:3,:,:]), masked(self.target) ) * mask_weight
        self.loss_G_L1_Rendering += 5.0 * self.criterionL1(masked(self.intermediate_fake), masked(self.target) ) * mask_weight
        self.loss_G_L1_Rendering += 10.0 * self.criterionL1(self.fake, self.target)

        self.loss_G_VGG_Rendering = 0.0
        if self.opt.lossType == 'VGG':
            self.loss_G_VGG_Rendering += 10.0 * self.vggloss(self.fake, self.target)

        self.loss_G_total = self.loss_G_L1_Rendering + self.loss_G_VGG_Rendering + self.loss_G_GAN

        self.loss_G_total.backward()

    def optimize_parameters(self, epoch_iter):
        alpha = (epoch_iter-5) / 50.0
        if alpha < 0.0: alpha = 0.0
        if alpha > 1.0: alpha = 1.0
        self.forward(alpha)


        updateDiscriminator = self.loss_G_GAN < 1.0#0.1

        # update Discriminator
        if updateDiscriminator:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        # update Generator
        self.set_requires_grad(self.netD, False)
        self.optimizer_texture_decoder.zero_grad()
        self.optimizer_inpainter.zero_grad()
        self.optimizer_T.zero_grad()

        self.backward_G(epoch_iter)

        self.optimizer_texture_decoder.step()
        self.optimizer_inpainter.step()
        self.optimizer_T.step()

