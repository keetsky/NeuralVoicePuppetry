import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from data.base_dataset import BaseDataset
from data.audio import Audio
#from data.image_folder import make_dataset
from PIL import Image

#def make_dataset(dir):
#    images = []
#    assert os.path.isdir(dir), '%s is not a valid directory' % dir
#    for root, _, fnames in sorted(os.walk(dir)):
#        for fname in fnames:
#            if any(fname.endswith(extension) for extension in ['.bin', '.BIN']):
#                path = os.path.join(root, fname)
#                images.append(path)
#    return sorted(images)

def make_dataset(dir):
    images = []
    ids = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if any(fname.endswith(extension) for extension in ['.bin', '.BIN']):
                id_str = fname[:-4]
                i = int(id_str)
                ids.append(i)
    ids = sorted(ids)

    for id in ids:
        fname=str(id)+'.bin'
        path = os.path.join(root, fname)
        images.append(path)
    return images

def load_intrinsics(input_dir):
    file = open(input_dir+"/intrinsics.txt", "r")
    intrinsics = [[(float(x) for x in line.split())] for line in file]
    file.close()
    intrinsics = list(intrinsics[0][0])
    return intrinsics

def load_rigids(input_dir):
    file = open(input_dir+"/rigid.txt", "r")
    rigid_floats = [[float(x) for x in line.split()] for line in file] # note that it stores 5 lines per matrix (blank line)
    file.close()
    all_rigids = [ [rigid_floats[4*idx + 0],rigid_floats[4*idx + 1],rigid_floats[4*idx + 2],rigid_floats[4*idx + 3]] for idx in range(0, len(rigid_floats)//4) ]
    return all_rigids

def load_expressions(input_dir):
    file = open(input_dir+"/expression.txt", "r")
    expressions = [[float(x) for x in line.split()] for line in file]
    file.close()
    return expressions

def load_audio(input_dir):
    audio = Audio(input_dir + '/audio.mp3', write_mel_spectogram = False)
    #audio = Audio(input_dir + '/audio.mp3', write_mel_spectogram = True)

    return audio 

def make_ids(paths, root_dir):
    ids = []
    
    for fname in paths:
        l = fname.rfind('/')
        id_str = fname[l+1:-4]
        i = int(id_str)
        #print(fname, ': ', i)
        ids.append(i)
    return ids

class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.data_dir = os.path.join(opt.dataroot, opt.phase)
        self.frame_paths = make_dataset(self.data_dir)
        self.frame_ids = make_ids(self.frame_paths, self.root)
        self.intrinsics = load_intrinsics(self.data_dir)
        self.extrinsics = load_rigids(self.data_dir)
        self.expressions = load_expressions(self.data_dir)

        self.audio = load_audio(self.data_dir)
        self.audio_window_size = opt.audio_window_size

        opt.nObjects = 1
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):

        # get video data
        frame_id = index

        #print('GET ITEM: ', index)
        img_path = self.frame_paths[index]
        frame_id = self.frame_ids[index]

        # intrinsics and extrinsics
        intrinsics = self.intrinsics
        extrinsics = self.extrinsics[frame_id]

        # get audio mel sample window
        frame_rate = 24#29.97
        #frame_rate = len(self.expressions) / self.audio.time_total
        mel_frame_idx = int((frame_id / frame_rate) * self.audio.mel_sample_rate)
        mels = self.audio.getWindow(mel_frame_idx, self.audio_window_size)


        # expressions
        expressions = torch.tensor(self.expressions[frame_id])

        # default image dimensions
        IMG_DIM_X = 512
        IMG_DIM_Y = 512

        # load image data
        #assert(IMG_DIM == self.opt.fineSize)
        img_array = np.memmap(img_path, dtype='float32', mode='r').__array__()
        if img_array.size != IMG_DIM_X * IMG_DIM_Y * 5:
            IMG_DIM_X = int(img_array[0])
            IMG_DIM_Y = int(img_array[1])
            img_array = img_array[2:]
            intrinsics = img_array[0:4]
            img_array = img_array[4:]

        img_array = np.clip(img_array, 0.0, 1.0)
        img = np.resize(img_array,  (IMG_DIM_Y, IMG_DIM_X, 5))
        A = img[:,:,0:3]
        B = img[:,:,3:5]
        B = np.concatenate((B, np.zeros((IMG_DIM_Y, IMG_DIM_X, 1))), axis=2)

        TARGET = transforms.ToTensor()(A.astype(np.float32))
        UV = transforms.ToTensor()(B.astype(np.float32))

        TARGET = 2.0 * TARGET - 1.0
        UV = 2.0 * UV - 1.0


        # load deepspeech feature
        dsf_fname = img_path[:-4] + '.deepspeech.npy'
#        print('dsf_fname:', dsf_fname)
        feature_array = np.load(dsf_fname)
        #feature_array = np.memmap(dsf_fname, dtype='float32', mode='r').__array__()
#        print('feature_array shape: ', feature_array.shape)
        dsf_np = np.resize(feature_array,  (16,29,1))
        dsf = transforms.ToTensor()(dsf_np.astype(np.float32))

        #################################
        ####### apply augmentation ######
        #################################
        if not self.opt.no_augmentation:
            # random dimensions
            new_dim_x = np.random.randint(int(IMG_DIM_X * 0.75), IMG_DIM_X+1)
            new_dim_y = np.random.randint(int(IMG_DIM_Y * 0.75), IMG_DIM_Y+1)
            new_dim_x = int(np.floor(new_dim_x / 64.0) * 64 ) # << dependent on the network structure !! 64 => 6 layers
            new_dim_y = int(np.floor(new_dim_y / 64.0) * 64 )
            if new_dim_x > IMG_DIM_X: new_dim_x -= 64
            if new_dim_y > IMG_DIM_Y: new_dim_y -= 64

            # random pos
            if IMG_DIM_X == new_dim_x: offset_x = 0
            else: offset_x = np.random.randint(0, IMG_DIM_X-new_dim_x)
            if IMG_DIM_Y == new_dim_y: offset_y = 0
            else: offset_y = np.random.randint(0, IMG_DIM_Y-new_dim_y)

            # select subwindow
            TARGET = TARGET[:, offset_y:offset_y+new_dim_y, offset_x:offset_x+new_dim_x]
            UV = UV[:, offset_y:offset_y+new_dim_y, offset_x:offset_x+new_dim_x]

            # compute new intrinsics
            # TODO: atm not needed but maybe later

        else:
            new_dim_x = int(np.floor(IMG_DIM_X / 64.0) * 64 ) # << dependent on the network structure !! 64 => 6 layers
            new_dim_y = int(np.floor(IMG_DIM_Y / 64.0) * 64 )
            offset_x = 0
            offset_y = 0
            # select subwindow
            TARGET = TARGET[:, offset_y:offset_y+new_dim_y, offset_x:offset_x+new_dim_x]
            UV = UV[:, offset_y:offset_y+new_dim_y, offset_x:offset_x+new_dim_x]

            # compute new intrinsics
            # TODO: atm not needed but maybe later

        #################################

        return {'TARGET': TARGET, 'UV': UV,
                'paths': self.frame_paths[index],#img_path,
                'intrinsics': intrinsics,
                'extrinsics': extrinsics,
                'expressions': expressions,
                'audio_mels': mels,
                'audio_deepspeech': dsf, # deepspeech feature
                'object_id':0}

    def __len__(self):
        return len(self.frame_paths)

    def name(self):
        return 'AlignedDataset'
