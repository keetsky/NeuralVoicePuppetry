import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import functools

from BaselModel.basel_model import *

################
###  HELPER  ###
################

INVALID_UV = -1.0


from torchvision import models
from collections import namedtuple


class ExpressionEstimator_Attention(nn.Module):
    def __init__(self, n_output_expressions, nIdentities, seq_len, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ExpressionEstimator_Attention, self).__init__()
        print('Estimator Attention')
        #################################
        ########   audio net   ##########
        #################################
        self.seq_len = seq_len

        dropout_rate = 0.0
        if use_dropout == True:
            #dropout_rate = 0.5
            dropout_rate = 0.25

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

        fullNet_input_size = 64

        self.subspace_dim = 32 # number of audio expressions
        print('fullNet_input_size: ', fullNet_input_size)
        self.fullNet = nn.Sequential(
            nn.Linear(in_features = fullNet_input_size, out_features=128, bias = True),
            nn.LeakyReLU(0.02),

            nn.Linear(in_features = 128, out_features=64, bias = True),
            nn.LeakyReLU(0.02),            

            nn.Linear(in_features = 64, out_features=self.subspace_dim, bias = True),          
            nn.Tanh()
            )


        # mapping from subspace to full expression space
        self.register_parameter('mapping', torch.nn.Parameter(torch.randn(1, nIdentities, N_EXPRESSIONS, self.subspace_dim, requires_grad=True)))

        # attention
        self.attentionConvNet = nn.Sequential( # b x subspace_dim x seq_len
            nn.Conv1d(self.subspace_dim, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features = self.seq_len, out_features=self.seq_len, bias = True),   
            nn.Softmax(dim=1)
            )
        #self.hidden2subspace = nn.Linear(self.subspace_dim,self.subspace_dim)

    def forward_internal(self, audio_features_sequence, identity_id):
        result_subspace, intermediate_expression = self.getAudioExpressions_internal(audio_features_sequence)
        mapping = torch.index_select(self.mapping[0], dim = 0, index = identity_id)
        result = 10.0 * torch.bmm(mapping, result_subspace)[:,:,0]
        result_intermediate = 10.0 * torch.bmm(mapping, intermediate_expression)[:,:,0]
        return result, result_intermediate

    def forward(self, audio_features_sequence, identity_id):
        result_subspace = self.getAudioExpressions(audio_features_sequence)
        mapping = torch.index_select(self.mapping[0], dim = 0, index = identity_id)
        result = torch.bmm(mapping, result_subspace)[:,:,0]
        return 10.0 * result

    def getAudioExpressions_internal(self, audio_features_sequence):
        # audio_features_sequence: b x seq_len x 16 x 29
        b = audio_features_sequence.shape[0] # batchsize
        audio_features_sequence = audio_features_sequence.view(b * self.seq_len, 1, 16, 29) # b * seq_len x 1 x 16 x 29
        audio_features_sequence = torch.transpose(audio_features_sequence, 1, 3) # b* seq_len  x 29 x 16 x 1
        conv_res = self.convNet( audio_features_sequence )
        conv_res = torch.reshape( conv_res, (b * self.seq_len, 1, -1))
        result_subspace = self.fullNet(conv_res)[:,0,:] # b * seq_len x subspace_dim
        result_subspace = result_subspace.view(b, self.seq_len, self.subspace_dim)# b x seq_len x subspace_dim

        #################
        ### attention ###
        ################# 
        result_subspace_T = torch.transpose(result_subspace, 1, 2) # b x subspace_dim x seq_len
        intermediate_expression = result_subspace_T[:,:,(self.seq_len // 2):(self.seq_len // 2) + 1]
        att_conv_res = self.attentionConvNet(result_subspace_T)
        #print('att_conv_res', att_conv_res.shape)
        attention = self.attentionNet(att_conv_res.view(b, self.seq_len)).view(b, self.seq_len, 1) # b x seq_len x 1
        #print('attention', attention.shape)
        # pooling along the sequence dimension
        result_subspace = torch.bmm(result_subspace_T, attention)
        #print('result_subspace', result_subspace.shape)
        ###

        return result_subspace.view(b, self.subspace_dim, 1), intermediate_expression

    def getAudioExpressions(self, audio_features_sequence):
        expr, _ = self.getAudioExpressions_internal(audio_features_sequence)
        return expr

    def regularizer(self):
        #reg = torch.norm(self.mapping)
        reg_mapping = torch.mean(torch.abs(self.mapping))

        # one could also enforce orthogonality here
        
        # s_browExpressions[] = { 32, 41, 71, 72, 73, 74, 75 };
        reg_eye_brow = torch.mean(torch.abs( self.mapping[0,:,[32, 41, 71, 72, 73, 74, 75],:] ))
        #return 0.01 * reg_mapping + 1.0 * reg_eye_brow
        return 0.0 * reg_mapping



def define_ExpressionEstimator(estimatorType='estimatorDefault', nIdentities=1, seq_len=1, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    print('EstimatorType: ', estimatorType)
    if estimatorType=='estimatorAttention': net = ExpressionEstimator_Attention(N_EXPRESSIONS,nIdentities, seq_len)

    return networks.init_net(net, init_type, init_gain, gpu_ids)


class Audio2ExpressionsAttentionTMP4Model(BaseModel):
    def name(self):
        return 'Audio2ExpressionsAttentionTMP4Model'

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
        self.loss_names = ['G_L1','G_L1_ABSOLUTE','G_L1_RELATIVE', 'G_Regularizer']

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        #self.visual_names = ['input_uv', 'fake', 'target']
        self.visual_names = ['zeros']
        self.zeros = torch.zeros(1,3,2,2)

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['netG']
        else:  # during test time, only load Gs
            self.model_names = ['netG']

        self.fake_expressions = None
        self.fake_expressions_prv = None
        self.fake_expressions_nxt = None

        self.morphable_model = MorphableModel()
        self.mask = self.morphable_model.LoadMask()

        nIdentities=opt.nTrainObjects

        # load/define networks
        self.netG = define_ExpressionEstimator(estimatorType=opt.rendererType, nIdentities=nIdentities, seq_len=opt.seq_len, gpu_ids=self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.fake_AB_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1Smooth = torch.nn.SmoothL1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0 )
            self.optimizers.append(self.optimizer_G)


    def set_input(self, input):
        self.image_paths = input['paths']
        
        self.expressions = input['expressions'].cuda()
        self.audio_features = input['audio_deepspeech'].cuda() # b x seq_len x 16 x 29
    
        if self.isTrain:
            self.expressions_prv = input['expressions_prv'].cuda()
            self.audio_features_prv = input['audio_deepspeech_prv'].cuda() # b x seq_len x 16 x 29
        
            self.expressions_nxt = input['expressions_nxt'].cuda()
            self.audio_features_nxt = input['audio_deepspeech_nxt'].cuda() # b x seq_len x 16 x 29

        self.target_id = input['target_id'].cuda()


    def forward(self):
        # estimate expressions
        if self.opt.output_audio_expressions: #self.opt.dataset_mode=='audio':
            self.fake_expressions = self.netG.getAudioExpressions(self.audio_features)
            if self.isTrain:
                self.fake_expressions_prv = self.netG.getAudioExpressions(self.audio_features_prv)
                self.fake_expressions_nxt = self.netG.getAudioExpressions(self.audio_features_nxt)
        else:
            self.fake_expressions, self.fake_expressions_intermediate = self.netG.forward_internal(self.audio_features, self.target_id)
            if self.isTrain:
                self.fake_expressions_prv = self.netG(self.audio_features_prv, self.target_id)
                self.fake_expressions_nxt = self.netG(self.audio_features_nxt, self.target_id)



    def backward_G(self, epoch):

        # Second, G(A) = B
        #self.loss_G_L1 = self.criterionL1(self.fake_expressions, self.expressions)

        # difference in vertex space
        mask = torch.cat([self.mask[:,None],self.mask[:,None],self.mask[:,None]], 1)
        mask = mask + 0.1 * torch.ones_like(mask) # priority for the mask region, but other region should also be constrained
        
        # absolute (single timesteps)
        diff_expression = self.fake_expressions - self.expressions
        diff_vertices = self.morphable_model.compute_expression_delta(diff_expression)

        
        diff_expression_intermediate = self.fake_expressions_intermediate - self.expressions
        diff_vertices_intermediate = self.morphable_model.compute_expression_delta(diff_expression_intermediate)

        
        diff_expression_prv = self.fake_expressions_prv - self.expressions_prv
        diff_vertices_prv = self.morphable_model.compute_expression_delta(diff_expression_prv)
        
        diff_expression_nxt = self.fake_expressions_nxt - self.expressions_nxt
        diff_vertices_nxt = self.morphable_model.compute_expression_delta(diff_expression_nxt)

        # relative (temporal 1 timestep) cur - nxt and prv - cur
        diff_expression_tmp_cur_nxt = (self.fake_expressions - self.fake_expressions_nxt) - (self.expressions - self.expressions_nxt)
        diff_vertices_tmp_cur_nxt = self.morphable_model.compute_expression_delta(diff_expression_tmp_cur_nxt)
        diff_expression_tmp_prv_cur = (self.fake_expressions_prv - self.fake_expressions) - (self.expressions_prv - self.expressions)
        diff_vertices_tmp_prv_cur = self.morphable_model.compute_expression_delta(diff_expression_tmp_prv_cur)

        # relative (temporal 2 timesteps)  nxt - prv
        diff_expression_tmp_nxt_prv = (self.fake_expressions_nxt - self.fake_expressions_prv) - (self.expressions_nxt - self.expressions_prv)
        diff_vertices_tmp_nxt_prv = self.morphable_model.compute_expression_delta(diff_expression_tmp_nxt_prv)

        #print('mask: ', mask.shape)
        #print('diff_vertices: ', diff_vertices.shape)
        
        self.loss_G_L1_ABSOLUTE = 0.0
        self.loss_G_L1_RELATIVE = 0.0
        if self.opt.lossType == 'L1':
            self.loss_G_L1_ABSOLUTE += 1000.0 * torch.mean(mask * torch.abs(diff_vertices)) # scale brings meters to millimeters
            self.loss_G_L1_ABSOLUTE += 1000.0 * torch.mean(mask * torch.abs(diff_vertices_prv))
            self.loss_G_L1_ABSOLUTE += 1000.0 * torch.mean(mask * torch.abs(diff_vertices_nxt))

            self.loss_G_L1_ABSOLUTE += 3000.0 * torch.mean(mask * torch.abs(diff_vertices_intermediate))
            
            self.loss_G_L1_RELATIVE += 20000.0 * torch.mean(mask * torch.abs(diff_vertices_tmp_cur_nxt))
            self.loss_G_L1_RELATIVE += 20000.0 * torch.mean(mask * torch.abs(diff_vertices_tmp_prv_cur))
            
            self.loss_G_L1_RELATIVE += 20000.0 * torch.mean(mask * torch.abs(diff_vertices_tmp_nxt_prv))

        elif self.opt.lossType == 'L2':
            self.loss_G_L1_ABSOLUTE += 1000.0 * torch.mean(mask * diff_vertices * diff_vertices)
            self.loss_G_L1_ABSOLUTE += 1000.0 * torch.mean(mask * diff_vertices_prv * diff_vertices_prv)
            self.loss_G_L1_ABSOLUTE += 1000.0 * torch.mean(mask * diff_vertices_nxt * diff_vertices_nxt)

            self.loss_G_L1_ABSOLUTE += 3000.0 * torch.mean(mask * diff_vertices_intermediate * diff_vertices_intermediate)

            self.loss_G_L1_RELATIVE += 20000.0 * torch.mean(mask * diff_vertices_tmp_cur_nxt * diff_vertices_tmp_cur_nxt)
            self.loss_G_L1_RELATIVE += 20000.0 * torch.mean(mask * diff_vertices_tmp_prv_cur * diff_vertices_tmp_prv_cur)

            self.loss_G_L1_RELATIVE += 20000.0 * torch.mean(mask * diff_vertices_tmp_nxt_prv * diff_vertices_tmp_nxt_prv)

        elif self.opt.lossType == 'RMS':
            self.loss_G_L1_ABSOLUTE += 1000.0 * torch.sqrt(torch.mean(mask * diff_vertices     * diff_vertices))
            self.loss_G_L1_ABSOLUTE += 1000.0 * torch.sqrt(torch.mean(mask * diff_vertices_prv * diff_vertices_prv))            
            self.loss_G_L1_ABSOLUTE += 1000.0 * torch.sqrt(torch.mean(mask * diff_vertices_nxt * diff_vertices_nxt))

            self.loss_G_L1_ABSOLUTE += 3000.0 * torch.sqrt(torch.mean(mask * diff_vertices_intermediate * diff_vertices_intermediate))
            
                     
            self.loss_G_L1_RELATIVE += 20000.0 * torch.sqrt(torch.mean(mask * diff_vertices_tmp_cur_nxt * diff_vertices_tmp_cur_nxt))         
            self.loss_G_L1_RELATIVE += 20000.0 * torch.sqrt(torch.mean(mask * diff_vertices_tmp_prv_cur * diff_vertices_tmp_prv_cur)) 

            self.loss_G_L1_RELATIVE += 20000.0 * torch.sqrt(torch.mean(mask * diff_vertices_tmp_nxt_prv * diff_vertices_tmp_nxt_prv))

        else: # L1
            self.loss_G_L1_ABSOLUTE += 1000.0 * torch.mean(mask * torch.abs(diff_vertices))
            self.loss_G_L1_ABSOLUTE += 1000.0 * torch.mean(mask * torch.abs(diff_vertices_prv))
            self.loss_G_L1_ABSOLUTE += 1000.0 * torch.mean(mask * torch.abs(diff_vertices_nxt))

            self.loss_G_L1_ABSOLUTE += 3000.0 * torch.mean(mask * torch.abs(diff_vertices_intermediate))
            
            self.loss_G_L1_RELATIVE += 20000.0 * torch.mean(mask * torch.abs(diff_vertices_tmp_cur_nxt))
            self.loss_G_L1_RELATIVE += 20000.0 * torch.mean(mask * torch.abs(diff_vertices_tmp_prv_cur))
            
            self.loss_G_L1_RELATIVE += 20000.0 * torch.mean(mask * torch.abs(diff_vertices_tmp_nxt_prv))

        self.loss_G_L1 = self.loss_G_L1_ABSOLUTE + self.loss_G_L1_RELATIVE
        self.loss_G_Regularizer = self.netG.regularizer()

        self.loss_G = self.loss_G_L1 + self.loss_G_Regularizer


        self.loss_G.backward()

    def optimize_parameters(self, epoch_iter):
        self.forward()

        # update Generator
        self.optimizer_G.zero_grad()
        self.backward_G(epoch_iter)
        self.optimizer_G.step()

