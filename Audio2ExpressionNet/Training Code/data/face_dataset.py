import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from data.base_dataset import BaseDataset
from data.audio import Audio
#from data.image_folder import make_dataset
from PIL import Image
from util import util

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
            if any(fname.endswith(extension) for extension in ['.npy', '.NPY']):
                #.deepspeech.npy
                id_str = fname[:-15] #4]
                i = int(id_str)
                ids.append(i)
    ids = sorted(ids)

    for id in ids:
        fname=str(id)+'.deepspeech.npy'
        path = os.path.join(root, fname)
        images.append(path)
    return images

def make_ids(paths, root_dir):
    ids = []
    
    for fname in paths:
        l = fname.rfind('/')
        id_str = fname[l+1:-15]#4]
        i = int(id_str)
        #print(fname, ': ', i)
        ids.append(i)
    return ids

def make_dataset_png_ids(dir):
    images = []
    ids = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if any(fname.endswith(extension) for extension in ['.png', '.PNG']):
                id_str = fname[:-4]
                i = int(id_str)
                ids.append(i)
    ids = sorted(ids)
    return ids

def make_dataset_exr_ids(dir):
    images = []
    ids = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if any(fname.endswith(extension) for extension in ['.exr', '.EXR']):
                id_str = fname[:-4]
                i = int(id_str)
                ids.append(i)
    ids = sorted(ids)
    return ids


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

def load_identity(input_dir):
    file = open(input_dir+"/identities.txt", "r")
    identity = [[float(x) for x in line.split()] for line in file]
    file.close()
    return identity



class FaceDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt

        # directories
        self.dataroot = opt.dataroot
        self.audio_feature_dir = os.path.join(opt.dataroot, 'audio_feature')
        self.image_dir = os.path.join(opt.dataroot, 'images')
        self.uvs_dir = os.path.join(opt.dataroot, 'uvs')

        # debug print
        print('load sequence:', self.dataroot)
        print('\taudio_feature_dir:', self.audio_feature_dir)
        print('\timage_dir:', self.image_dir)
        print('\tuvs_dir:', self.uvs_dir)

        # generate index maps
        audio_ids = make_ids(make_dataset(self.audio_feature_dir), self.dataroot)
        image_ids = make_dataset_png_ids(self.image_dir)
        uvs_ids = make_dataset_exr_ids(self.uvs_dir)

        # get model parameters
        intrinsics = load_intrinsics(self.dataroot)
        extrinsics = load_rigids(self.dataroot)
        expressions = load_expressions(self.dataroot)
        identities = load_identity(self.dataroot)

        if opt.phase == 'test': # test overwrites the audio and uv files, as well as expressions
            print('Test mode. Overwriting audio, uv and expressions')
            print('source sequence:', opt.source_dir)
            dataroot = opt.source_dir
            self.audio_feature_dir = os.path.join(dataroot, 'audio_feature')
            self.uvs_dir = os.path.join(dataroot, 'uvs')
            audio_ids = make_ids(make_dataset(self.audio_feature_dir), dataroot)
            uvs_ids = make_dataset_exr_ids(self.uvs_dir)
            intrinsics = load_intrinsics(dataroot)
            extrinsics = load_rigids(dataroot)
            expressions = load_expressions(dataroot)
            identities = load_identity(dataroot)

        print('\tnum audio_ids:', len(audio_ids))
        print('\tnum image_ids:', len(image_ids))
        print('\tnum uvs_ids:', len(uvs_ids))


        # set data
        min_len = min(len(audio_ids), len(image_ids), len(uvs_ids), len(extrinsics), len(expressions))
        self.audio_ids = audio_ids[:min_len]
        self.image_ids = image_ids[:min_len]
        self.uvs_ids = uvs_ids[:min_len]
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics[:]   #extrinsics[:min_len]
        self.expressions = expressions[:] #expressions[:min_len]
        self.identities = identities[:]   #identities[:min_len]
        self.n_frames_total = min_len

        print('\tnum frames:', self.n_frames_total)


        opt.nTrainObjects = 1
        opt.nValObjects = 1
        opt.nTestObjects = 1

        
        opt.test_sequence_names = [[opt.dataroot.split("/")[-1], 'train']]
        if opt.phase == 'test':
            opt.test_sequence_names = [[opt.source_dir.split("/")[-1], 'test']]
            print('test:', opt.test_sequence_names)

        assert(opt.resize_or_crop == 'resize_and_crop')

    def getSampleWeights(self):
        weights = np.ones((self.n_frames_total))
        return weights

    def getExtrinsics(self, idx):
        return self.extrinsics[self.uvs_ids[idx % self.n_frames_total]]
    def getIntrinsics(self, idx):
        return self.intrinsics
    def getIdentities(self, idx):
        return self.identities[self.uvs_ids[idx % self.n_frames_total]]
    def getExpressions(self, idx):
        return self.expressions[self.uvs_ids[idx % self.n_frames_total]]

    def getAudioFilename(self):
        return os.path.join(self.dataroot, 'audio.wav')

    def getImageFilename(self, idx):
        image_id = self.image_ids[idx % self.n_frames_total]
        img_fname = os.path.join(self.image_dir, str(image_id).zfill(5) + '.png')
        return img_fname
        #img_numpy = np.asarray(Image.open(img_fname))
        #TARGET = 2.0 * transforms.ToTensor()(img_numpy.astype(np.float32))/255.0 - 1.0

    def getAudioFeatureFilename(self, idx):
        #return self.frame_paths[idx % len(self.frame_paths)]
        audio_id = self.audio_ids[idx]
        return os.path.join(self.audio_feature_dir, str(audio_id) + '.deepspeech.npy')


    def computeCrop(self, mask, MULTIPLE_OF=64, random_size=False):
        IMG_DIM_X = mask.shape[2]
        IMG_DIM_Y = mask.shape[1]
        if random_size:
            # random dimensions
            new_dim_x = np.random.randint(int(IMG_DIM_X * 0.75), IMG_DIM_X+1)
            new_dim_y = np.random.randint(int(IMG_DIM_Y * 0.75), IMG_DIM_Y+1)
            new_dim_x = int(np.floor(new_dim_x / float(MULTIPLE_OF)) * MULTIPLE_OF )
            new_dim_y = int(np.floor(new_dim_y / float(MULTIPLE_OF)) * MULTIPLE_OF )
        else:
            new_dim_x = 3 * MULTIPLE_OF
            new_dim_y = 3 * MULTIPLE_OF

        # check dims
        if new_dim_x > IMG_DIM_X: new_dim_x -= MULTIPLE_OF
        if new_dim_y > IMG_DIM_Y: new_dim_y -= MULTIPLE_OF
        
        # random pos
        mask_indices = torch.nonzero(mask)
        _, bb_mid_point_y, bb_mid_point_x = mask_indices[np.random.randint(0, mask_indices.shape[0])].data.cpu()
        #print('bb_mid_point', bb_mid_point_x, bb_mid_point_y)
        
        offset_x = bb_mid_point_x - new_dim_x/2
        offset_y = bb_mid_point_y - new_dim_y/2

  
        if IMG_DIM_X == new_dim_x: offset_x = 0
        if offset_x < 0: offset_x = 0
        if offset_x+new_dim_x >= IMG_DIM_X: offset_x = IMG_DIM_X-new_dim_x

        if IMG_DIM_Y == new_dim_y: offset_y = 0
        if offset_y < 0: offset_y = 0
        if offset_y+new_dim_y >= IMG_DIM_Y: offset_y = IMG_DIM_Y-new_dim_y

        return np.array([int(offset_x),int(offset_y),int(new_dim_x), int(new_dim_y)])


    def __getitem__(self, global_index):
        # select frame from sequence
        index = global_index

        # get data ids
        audio_id = self.audio_ids[index]
        image_id = self.image_ids[index]
        uv_id = self.uvs_ids[index]



        #print('GET ITEM: ', index)
        img_fname = os.path.join(self.image_dir, str(image_id).zfill(5) + '.png')
        img_numpy = np.asarray(Image.open(img_fname))
        TARGET = 2.0 * transforms.ToTensor()(img_numpy.astype(np.float32))/255.0 - 1.0

        uv_fname = os.path.join(self.uvs_dir, str(uv_id).zfill(5) + '.exr')
        uv_numpy = util.load_exr(uv_fname)
        UV = transforms.ToTensor()(uv_numpy.astype(np.float32))
        UV = torch.where(UV > 1.0, torch.zeros_like(UV), UV)
        UV = torch.where(UV < 0.0, torch.zeros_like(UV), UV)
        UV = 2.0 * UV - 1.0

        #print('img_fname:', img_fname)
        #print('uv_fname:', uv_fname)

        # intrinsics and extrinsics
        intrinsics = self.intrinsics
        extrinsics = self.extrinsics[uv_id]

        # expressions
        expressions = np.asarray(self.expressions[audio_id], dtype=np.float32)
        #print('expressions:', expressions.shape)
        expressions[32] *= 0.0         # remove eye brow movements
        expressions[41] *= 0.0         # remove eye brow movements
        expressions[71:75] *= 0.0         # remove eye brow movements
        expressions = torch.tensor(expressions)

        # identity
        identity = torch.tensor(self.identities[audio_id])

        # load deepspeech feature
        dsf_fname = os.path.join(self.audio_feature_dir, str(audio_id) + '.deepspeech.npy')
        feature_array = np.load(dsf_fname)
        dsf_np = np.resize(feature_array,  (16,29,1))
        dsf = transforms.ToTensor()(dsf_np.astype(np.float32))


        # load sequence data if necessary
        if self.opt.look_ahead:# use prev and following frame infos
            r = self.opt.seq_len//2
            last_valid_idx = audio_id
            for i in range(1,r): # prev frames
                index_seq = index - i
                if index_seq < 0: index_seq = 0
                audio_id_seq = self.audio_ids[index_seq]
                if audio_id_seq == audio_id - i: last_valid_idx = audio_id_seq
                else: audio_id_seq = last_valid_idx

                dsf_fname = os.path.join(self.audio_feature_dir, str(audio_id_seq) + '.deepspeech.npy')
                feature_array = np.load(dsf_fname)
                dsf_np = np.resize(feature_array,  (16,29,1))
                dsf_seq = transforms.ToTensor()(dsf_np.astype(np.float32)) # 1 x 16 x 29
                dsf = torch.cat([dsf_seq, dsf], 0)  # seq_len x 16 x 29
                # note the ordering [old ... current]
            
            last_valid_idx = audio_id
            for i in range(1,self.opt.seq_len - r + 1): # following frames
                index_seq = index + i
                max_idx = len(self.audio_ids)-1
                if index_seq > max_idx: index_seq = max_idx
                audio_id_seq = self.audio_ids[index_seq]
                if audio_id_seq == audio_id + i: last_valid_idx = audio_id_seq
                else: audio_id_seq = last_valid_idx

                dsf_fname = os.path.join(self.audio_feature_dir, str(audio_id_seq) + '.deepspeech.npy')
                feature_array = np.load(dsf_fname)
                dsf_np = np.resize(feature_array,  (16,29,1))
                dsf_seq = transforms.ToTensor()(dsf_np.astype(np.float32)) # 1 x 16 x 29
                dsf = torch.cat([dsf, dsf_seq], 0)  # seq_len x 16 x 29
                # note the ordering [old ... current ... future]
        else:
            last_valid_idx = audio_id
            for i in range(1,self.opt.seq_len):
                index_seq = index - i
                if index_seq < 0: index_seq = 0
                audio_id_seq = self.audio_ids[index_seq]
                if audio_id_seq == audio_id - i: last_valid_idx = audio_id_seq
                else: audio_id_seq = last_valid_idx

                dsf_fname = os.path.join(self.audio_feature_dir, str(audio_id_seq) + '.deepspeech.npy')
                feature_array = np.load(dsf_fname)
                dsf_np = np.resize(feature_array,  (16,29,1))
                dsf_seq = transforms.ToTensor()(dsf_np.astype(np.float32)) # 1 x 16 x 29
                dsf = torch.cat([dsf_seq, dsf], 0)  # seq_len x 16 x 29
                # note the ordering [old ... current]


        #################################
        ####### apply augmentation ######
        #################################
        crop = np.array([0,0,UV.shape[2],UV.shape[1]])
        if not self.opt.no_augmentation:
            INVALID_UV = -1
            mask = ( (UV[0:1,:,:] != INVALID_UV) | (UV[1:2,:,:] != INVALID_UV) )
            crop = self.computeCrop(mask, MULTIPLE_OF=64) # << dependent on the network structure !! 64 => 6 layers

            offset_x,offset_y,new_dim_x, new_dim_y = crop
            # select subwindow
            TARGET = TARGET[:, offset_y:offset_y+new_dim_y, offset_x:offset_x+new_dim_x]
            UV = UV[:, offset_y:offset_y+new_dim_y, offset_x:offset_x+new_dim_x]

            # compute new intrinsics
            # TODO: atm not needed but maybe later

        #################################
        weight = 1.0 / self.n_frames_total
        
        return {'TARGET': TARGET, 'UV': UV,
                'paths': dsf_fname, #img_path,
                'intrinsics': np.array(intrinsics),
                'extrinsics': np.array(extrinsics),
                'expressions': expressions,
                'identity': identity,
                'audio_deepspeech': dsf, # deepspeech feature
                'target_id': -1,
                'crop': crop,
                'internal_id': 0,
                'weight': np.array([weight]).astype(np.float32)}

    def __len__(self):
        return self.n_frames_total

    def name(self):
        return 'FaceDataset'
