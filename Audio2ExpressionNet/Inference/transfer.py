import os
import os.path
from options.transfer_options import TransferOptions
from data import CreateDataLoader
from data.face_dataset import FaceDataset
from data.audio_dataset import AudioDataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import time
import random
import progressbar
import copy
from shutil import copyfile

from BaselModel.basel_model import *


def load_model(opt):
    opt.output_audio_expressions = True
    opt.nTrainObjects = 116

    print('#train objects = %d' % opt.nTrainObjects)

    print('>>> create model <<<')
    model = create_model(opt)
    print('>>> setup model <<<')
    model.setup(opt)

    return model

def load_target_sequence(opt):
    opt_target = copy.copy(opt) # create a clone
    opt_target.dataroot = opt.target_actor # overwrite root directory
    opt_target.dataset_mode = 'face'
    opt_target.phase = 'train'
    dataset_target = FaceDataset()
    dataset_target.initialize(opt_target)

    dataloader = torch.utils.data.DataLoader(
            dataset_target,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    return dataset_target, dataloader


def load_source_sequence(opt):
    opt_source = copy.copy(opt) # create a clone
    opt_source.dataroot = opt.source_actor # overwrite root directory
    opt_source.dataset_mode = 'audio'
    opt_source.phase = 'train'
    dataset_source = AudioDataset()
    dataset_source.initialize(opt_source)

    dataloader = torch.utils.data.DataLoader(
            dataset_source,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    return dataset_source, dataloader


if __name__ == '__main__':
    # read options
    opt = TransferOptions().parse()

    target_name = opt.target_actor.split("/")[-1]

    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_augmentation = True # no flip
    opt.display_id = -1   # no visdom display

    # load model
    model = load_model(opt)
    print('model version:', opt.name)


    # # load face model
    # morphable_model = MorphableModel()
    # mask = morphable_model.LoadMask()
    # mask = mask + 0.1 * torch.ones_like(mask)

    # read target sequence
    dataset_target, data_loader_target = load_target_sequence(opt)
    dataset_target_size = len(dataset_target)
    print('#target_actor  frames = %d' % dataset_target_size)

    ##################################
    #######   create mapping   #######
    ##################################
    os.makedirs('./mappings/'+opt.name, exist_ok=True)
    mapping_fn = './mappings/'+opt.name+'/'+'mapping_'+target_name
    #not_exists = True
    not_exists = not os.path.exists(mapping_fn+'.npy')
    if not_exists:
        # collect data
        print('collect data')
        audio_expressions = None
        gt_expressions = None
        with progressbar.ProgressBar(max_value=len(dataset_target)) as bar:
            for i, data in enumerate(data_loader_target):
                bar.update(i)
                model.set_input(data)
                model.test()
                
                ae = model.fake_expressions.data[:,:,0]
                if type(audio_expressions) == type(None):
                    audio_expressions = ae
                    e = model.expressions.data
                    gt_expressions = e
                else:
                    audio_expressions = torch.cat([audio_expressions,ae],dim=0)
                    e = model.expressions.data
                    gt_expressions = torch.cat([gt_expressions,e],dim=0)

        # solve for mapping
        print('solve for mapping')
        optimize_in_parameter_space = True #False
        if optimize_in_parameter_space:
#            A = audio_expressions
#            B = gt_expressions
#            # solve lstsq  ||AX - B||
#            X, _ = torch.gels(B, A, out=None)
#            #X, _ = torch.lstsq(B, A) # requires pytorch 1.2
#            X = X[0:A.shape[1],:]
#            mapping = X.t()

            # use gradient descent method
            n = audio_expressions.shape[0]
            subspace_dim = 32
            X = torch.nn.Parameter(torch.randn(N_EXPRESSIONS, subspace_dim, requires_grad=True).cuda())
            optimizer = torch.optim.Adam([X], lr=0.01)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            num_epochs = 90
            random_range = [k for k in range(0,n)]
            with progressbar.ProgressBar(max_value=num_epochs) as bar:
                for ep in range(0,num_epochs):
                    bar.update(ep)
                    random.shuffle(random_range)
                    for j in random_range:
                        expressions = gt_expressions[j]
                        fake_expressions =  10.0 * torch.matmul(X, audio_expressions[j])
                        diff_expression = fake_expressions - expressions
                        loss = torch.mean(diff_expression * diff_expression) # L2
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    lr_scheduler.step()
            mapping = X.data



#        else: # optimize in vertex space
#            # use gradient descent method
#            n = audio_expressions.shape[0]
#            subspace_dim = 32
#            X = torch.nn.Parameter(torch.randn(N_EXPRESSIONS, subspace_dim, requires_grad=True).cuda())
#            optimizer = torch.optim.Adam([X], lr=0.01)
#            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
#            num_epochs = 90
#            random_range = [k for k in range(0,n)]
#            with progressbar.ProgressBar(max_value=num_epochs) as bar:
#                for ep in range(0,num_epochs):
#                    bar.update(ep)
#                    random.shuffle(random_range)
#                    for j in random_range:
#                        expressions = gt_expressions[j]
#                        fake_expressions =  10.0 * torch.matmul(X, audio_expressions[j])
#                        diff_expression = fake_expressions - expressions
#                        diff_vertices = torch.matmul(morphable_model.expression_basis, diff_expression)
#                        #loss = torch.sqrt(torch.mean(mask * diff_vertices * diff_vertices)) # RMS
#                        loss = torch.mean(mask * diff_vertices * diff_vertices) # L2
#                        #
#                        optimizer.zero_grad()
#                        loss.backward()
#                        optimizer.step()
#                    lr_scheduler.step()
#
#            mapping = X.data

        map_cpu = mapping.data.cpu().numpy()
        file_out=open(mapping_fn+'.txt', 'w')
        np.savetxt(file_out, map_cpu, delimiter=' ')
        file_out.close()
        np.save(mapping_fn+'.npy', map_cpu)
    else:
        # load mapping from file
        map_cpu = np.load(mapping_fn+'.npy')
        mapping = torch.tensor(map_cpu.astype(np.float32)).cuda()
        print('loaded mapping from file', mapping.shape)



    # process source sequence (opt.source_actor)
    source_actors = [ 
                        './datasets/SOURCES/greta_1',
                    ]

    os.makedirs('./datasets/TRANSFERS/'+opt.name, exist_ok=True)
    list_transfer = open('./datasets/TRANSFERS/'+opt.name+'/list_transfer.txt', "a")

    target_actor_offset = 0 # default
    expression_multiplier = 1.0 # default


    if target_actor_offset != 0.0:
        target_name = target_name + '--offset'
    if expression_multiplier != 1.0:
        target_name = target_name + '-X'

    for source_actor in source_actors:
        opt.source_actor = source_actor
        source_name = opt.source_actor.split("/")[-1]
        # read source sequence
        dataset_source, data_loader_source = load_source_sequence(opt)
        dataset_source_size = len(dataset_source)
        print('#source_actor  frames = %d' % dataset_source_size)
        list_transfer.write(source_name+'--'+target_name+'\n')
        out_dir = './datasets/TRANSFERS/'+opt.name+'/'+source_name+'--'+target_name
        os.makedirs(out_dir, exist_ok=True)
        result_expressions_file=open(out_dir+'/'+'expression.txt', 'w')
        result_expressions = []
        with progressbar.ProgressBar(max_value=len(dataset_source)) as bar:
            for i, data in enumerate(data_loader_source):
                bar.update(i)
                model.set_input(data)
                model.test()
                audio_expression = model.fake_expressions.data[0,:,0]
                expression = expression_multiplier * 10.0 * torch.matmul(mapping, audio_expression)
                expression = expression[None,:]
                result_expressions.append(expression)
                np.savetxt(result_expressions_file, expression.cpu().numpy(), delimiter=' ')
        result_expressions_file.close()

    list_transfer.close()
    exit()
