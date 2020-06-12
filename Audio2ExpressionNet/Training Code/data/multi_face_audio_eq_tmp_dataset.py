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

def make_dataset_ids_png(dir):
    images = []
    ids = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if any(fname.endswith(extension) for extension in ['.png', '.png']):
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
    identities = [[float(x) for x in line.split()] for line in file]
    file.close()
    return identities


class MultiFaceAudioEQTmpDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        # read dataset file that contains the filenames for the train, val and test lists
        file = open(self.root+"/dataset.txt", "r")
        self.filename_train_list, self.filename_val_list, self.filename_test_list = [str(line) for line in file]
        file.close()
        if self.filename_train_list[-1] == '\n': self.filename_train_list = self.filename_train_list[:-1] 
        if self.filename_val_list[-1] == '\n':   self.filename_val_list   = self.filename_val_list[:-1] 
        if self.filename_test_list[-1] == '\n':  self.filename_test_list  = self.filename_test_list[:-1] 



        # get list of train sequences
        file = open(self.root+"/" + self.filename_train_list, "r")
        self.train_sequence_names = [str(line) for line in file]
        file.close()
        for i in range(0,len(self.train_sequence_names)):
            if self.train_sequence_names[i][-1] == '\n':
                self.train_sequence_names[i] = self.train_sequence_names[i][:-1]

        # get list of val sequences
        file = open(self.root+"/" + self.filename_val_list, "r")
        self.val_sequence_names = [[str(w) for w in line.split()] for line in file]
        file.close()
        for i in range(0,len(self.val_sequence_names)):
            if self.val_sequence_names[i][0][-1] == '\n': self.val_sequence_names[i][0] = self.val_sequence_names[i][0][:-1]
            if self.val_sequence_names[i][1][-1] == '\n': self.val_sequence_names[i][1] = self.val_sequence_names[i][1][:-1]

        # get list of test sequences
        file = open(self.root+"/" + self.filename_test_list, "r")
        self.test_sequence_names = [[str(w) for w in line.split()] for line in file]
        if opt.output_audio_expressions: self.test_sequence_names = self.test_sequence_names[0:1]
        file.close()
        for i in range(0,len(self.test_sequence_names)):
            if self.test_sequence_names[i][0][-1] == '\n': self.test_sequence_names[i][0] = self.test_sequence_names[i][0][:-1]
            if self.test_sequence_names[i][1][-1] == '\n': self.test_sequence_names[i][1] = self.test_sequence_names[i][1][:-1]

        # print some stats
        print('filename_train_list:', self.filename_train_list)
        print('\tnum_seq:', len(self.train_sequence_names))
        print('filename_val_list:  ', self.filename_val_list)
        print('\tnum_seq:', len(self.val_sequence_names))
        print('filename_test_list: ', self.filename_test_list)
        print('\tnum_seq:', len(self.test_sequence_names))

        opt.train_sequence_names = self.train_sequence_names
        opt.val_sequence_names = self.val_sequence_names
        opt.test_sequence_names = self.test_sequence_names

        # search mapping from val, test to train sequences that are used as targets
        self.val_sequence_targets = []
        for i in range(0,len(self.val_sequence_names)):
            target_name = self.val_sequence_names[i][1]
            target_id = -1
            for j in range(0,len(self.train_sequence_names)):
                if self.train_sequence_names[j] == target_name:
                    target_id = j
                    break
            if target_id == -1:
                print('Target sequence not in train set! ', target_name)
                exit()
            self.val_sequence_targets.append(target_id)

        self.test_sequence_targets = []
        for i in range(0,len(self.test_sequence_names)):
            target_name = self.test_sequence_names[i][1]
            target_id = -1
            for j in range(0,len(self.train_sequence_names)):
                if self.train_sequence_names[j] == target_name:
                    target_id = j
                    break
            if target_id == -1:
                print('Target sequence not in train set! ', target_name)
                exit()
            self.test_sequence_targets.append(target_id)
            print('test: ', self.test_sequence_names[i])
            print('\t target:', target_id)

        # store len values
        opt.nTrainObjects = len(self.train_sequence_names)
        opt.nValObjects = len(self.val_sequence_names)
        opt.nTestObjects = len(self.test_sequence_names)

        ################################################
        ################################################
        ################################################

        # prepare dataloader paths / data
        self.audio_feature_dir = []
        self.image_dir = []
        self.uvs_dir = []
        self.audio_ids = []
        self.image_ids = []
        self.intrinsics = []
        self.extrinsics = []
        self.expressions = []
        self.identities = []
        self.target_id = []
        self.n_frames_total = 0

        if opt.phase == 'train':
            self.sequence_names = self.train_sequence_names
            for i in range(0,len(self.train_sequence_names)):
                dataroot          = os.path.join(opt.dataroot, self.train_sequence_names[i])
                audio_feature_dir = os.path.join(opt.dataroot, self.train_sequence_names[i], 'audio_feature')
                image_dir         = os.path.join(opt.dataroot, self.train_sequence_names[i], 'images')
                uvs_dir           = os.path.join(opt.dataroot, self.train_sequence_names[i], 'uvs')
                print('load train sequence:', self.train_sequence_names[i])
                print('\tidentity_dir:', dataroot)
                print('\taudio_feature_dir:', audio_feature_dir)
                print('\timage_dir:', image_dir)
                print('\tuvs_dir:', uvs_dir)

                audio_ids = make_ids(make_dataset(audio_feature_dir), dataroot)
                image_ids = make_dataset_ids_png(image_dir) # [-1] * len(audio_ids) #make_ids(make_dataset(image_dir), dataroot)
                intrinsics = load_intrinsics(dataroot)
                extrinsics = load_rigids(dataroot)
                expressions = load_expressions(dataroot)
                identity = load_identity(dataroot)

                min_len = min(len(audio_ids), len(image_ids), len(extrinsics), len(expressions))

                self.audio_feature_dir.append(audio_feature_dir)
                self.image_dir.append(image_dir)
                self.uvs_dir.append(uvs_dir)
                self.audio_ids.append(audio_ids[:min_len])
                self.image_ids.append(image_ids[:min_len])
                self.intrinsics.append(intrinsics)
                self.extrinsics.append(extrinsics[:min_len])
                self.expressions.append(expressions[:min_len])
                self.identities.append(identity[:min_len])
                self.target_id.append(i)

                self.n_frames_total += min_len
        elif opt.phase == 'val':
            for i in range(0,len(self.val_sequence_names)):
                target_id = self.val_sequence_targets[i]
                dataroot          = os.path.join(opt.dataroot, self.train_sequence_names[target_id])
                audio_feature_dir = os.path.join(opt.dataroot, self.val_sequence_names[i][0], 'audio_feature')
                image_dir         = os.path.join(opt.dataroot, self.train_sequence_names[target_id], 'images')
                uvs_dir           = os.path.join(opt.dataroot, self.train_sequence_names[target_id], 'uvs')
                print('load val sequence:', self.val_sequence_names[i])
                print('\tidentity_dir:', dataroot)
                print('\taudio_feature_dir:', audio_feature_dir)
                print('\timage_dir:', image_dir)
                print('\tuvs_dir:', uvs_dir)
                audio_ids = make_ids(make_dataset(audio_feature_dir), os.path.join(opt.dataroot, self.val_sequence_names[i][0]))
                image_ids = make_dataset_ids_png(image_dir) # [-1] * len(audio_ids) #make_ids(make_dataset(image_dir), dataroot)
                intrinsics = load_intrinsics(dataroot)
                extrinsics = load_rigids(dataroot)
                expressions = load_expressions(os.path.join(opt.dataroot, self.val_sequence_names[i][0]))
                identity = load_identity(dataroot)

                min_len = min(len(audio_ids), len(image_ids), len(extrinsics), len(expressions))

                self.audio_feature_dir.append(audio_feature_dir)
                self.image_dir.append(image_dir)
                self.uvs_dir.append(uvs_dir)
                self.audio_ids.append(audio_ids[:min_len])
                self.image_ids.append(image_ids[:min_len])
                self.intrinsics.append(intrinsics)
                self.extrinsics.append(extrinsics[:min_len])
                self.expressions.append(expressions[:min_len])
                self.identities.append(identity[:min_len])
                self.target_id.append(target_id)

                self.n_frames_total += min_len               
        else: # test  
            for i in range(0,len(self.test_sequence_names)):
                target_id = self.test_sequence_targets[i]
                dataroot          = os.path.join(opt.dataroot, self.train_sequence_names[target_id])
                audio_feature_dir = os.path.join(opt.dataroot, self.test_sequence_names[i][0], 'audio_feature')
                image_dir         = os.path.join(opt.dataroot, self.train_sequence_names[target_id], 'images')
                uvs_dir           = os.path.join(opt.dataroot, self.train_sequence_names[target_id], 'uvs')
                print('load test sequence:', self.test_sequence_names[i])
                print('\tidentity_dir:', dataroot)
                print('\taudio_feature_dir:', audio_feature_dir)
                print('\timage_dir:', image_dir)
                print('\tuvs_dir:', uvs_dir)
                audio_ids = make_ids(make_dataset(audio_feature_dir), os.path.join(opt.dataroot, self.test_sequence_names[i][0]))
                image_ids = make_dataset_ids_png(image_dir) # [-1] * len(audio_ids) #make_ids(make_dataset(image_dir), dataroot)
                intrinsics = load_intrinsics(dataroot)
                extrinsics = load_rigids(dataroot)
                expressions = load_expressions(os.path.join(opt.dataroot, self.test_sequence_names[i][0]))
                identity = load_identity(dataroot)

                min_len = min(len(audio_ids), len(image_ids), len(extrinsics), len(expressions))

                self.audio_feature_dir.append(audio_feature_dir)
                self.image_dir.append(image_dir)
                self.uvs_dir.append(uvs_dir)
                self.audio_ids.append(audio_ids[:min_len])
                self.image_ids.append(image_ids[:min_len])
                self.intrinsics.append(intrinsics)
                self.extrinsics.append(extrinsics[:min_len])
                self.expressions.append(expressions[:min_len])
                self.identities.append(identity[:min_len])
                self.target_id.append(target_id)

                self.n_frames_total += min_len  


        print('frames_total:', self.n_frames_total)


        #global_target_ids = []
        #for i in range(0,len(self.audio_ids)):
        #    for j in range(0,len(self.audio_ids[i])):
        #        global_target_ids.append(self.target_id[i])
        #global_target_ids=np.array(global_target_ids)
        #self.weights = np.where(global_target_ids==2, 1.0 * np.ones((self.n_frames_total)),  0.01 * np.ones((self.n_frames_total)) )
        self.weights = []
        for i in range(0,len(self.audio_ids)):
            l = len(self.audio_ids[i])
            for j in range(0,l):
                self.weights.append(1.0 / l)
        self.weights = np.array(self.weights)

        assert(opt.resize_or_crop == 'resize_and_crop')

    def getSampleWeights(self):
        return self.weights

    def getitem(self, global_index):

        # select sequence
        internal_sequence_id = 0
        sum_frames = 0
        for i in range(0,len(self.audio_ids)):
            l = len(self.audio_ids[i])
            if (global_index-sum_frames) < l:
                internal_sequence_id = i
                break
            else:
                sum_frames += len(self.audio_ids[i])

        # select frame from sequence
        index = (global_index-sum_frames) % len(self.audio_ids[internal_sequence_id])

        # get data ids
        audio_id = self.audio_ids[internal_sequence_id][index]
        image_id = self.image_ids[internal_sequence_id][index]

        #print('GET ITEM: ', index)
        #img_path = self.frame_paths[sequence_id][index]

        # intrinsics and extrinsics
        intrinsics = self.intrinsics[internal_sequence_id]
        extrinsics = self.extrinsics[internal_sequence_id][image_id]

        # expressions
        expressions = np.asarray(self.expressions[internal_sequence_id][audio_id], dtype=np.float32)
        #print('expressions:', expressions.shape)
        expressions[32] *= 0.0         # remove eye brow movements
        expressions[41] *= 0.0         # remove eye brow movements
        expressions[71:75] *= 0.0         # remove eye brow movements
        expressions = torch.tensor(expressions)



        # identity
        identity = torch.tensor(self.identities[internal_sequence_id][image_id])
        target_id = self.target_id[internal_sequence_id] # sequence id refers to the target sequence (of the training corpus)

        # load deepspeech feature
        #print('audio_id', audio_id)    
        dsf_fname = os.path.join(self.audio_feature_dir[internal_sequence_id], str(audio_id) + '.deepspeech.npy')
        feature_array = np.load(dsf_fname)
        dsf_np = np.resize(feature_array,  (16,29,1))
        dsf = transforms.ToTensor()(dsf_np.astype(np.float32))

        # load sequence data if necessary
        last_valid_idx = audio_id
        for i in range(1,self.opt.seq_len):
            index_seq = index - i
            if index_seq < 0: index_seq = 0
            audio_id_seq = self.audio_ids[internal_sequence_id][index_seq]
            if audio_id_seq == audio_id - i: last_valid_idx = audio_id_seq
            else: audio_id_seq = last_valid_idx

            dsf_fname = os.path.join(self.audio_feature_dir[internal_sequence_id], str(audio_id_seq) + '.deepspeech.npy')
            feature_array = np.load(dsf_fname)
            dsf_np = np.resize(feature_array,  (16,29,1))
            dsf_seq = transforms.ToTensor()(dsf_np.astype(np.float32)) # 1 x 16 x 29
            dsf = torch.cat([dsf_seq, dsf], 0)  # seq_len x 16 x 29
            # note the ordering [old ... current]

 

        #weight = 1.0 / len(self.audio_feature_dir[internal_sequence_id])
        weight = self.weights[global_index]

        return {'paths': dsf_fname, #img_path,
                'intrinsics': np.array(intrinsics),
                'extrinsics': np.array(extrinsics),
                'expressions': expressions,
                'identity': identity,
                'audio_deepspeech': dsf, # deepspeech feature
                'target_id':target_id,
                'internal_id':internal_sequence_id,
                
                'weight': np.array([weight]).astype(np.float32)}


    def __getitem__(self, global_index):
        # select frame from sequence
        index = global_index
        current = self.getitem(index)
        prv = self.getitem(max(index-1, 0))
        nxt = self.getitem(min(index+1, self.n_frames_total-1))

        return {
                'paths':       current['paths'], #img_path,
                'target_id':   current['target_id'],
                'internal_id': current['internal_id'],
                'weight':      current['weight'],
                'identity':    current['identity'],
                'intrinsics':  current['intrinsics'],
                
                'extrinsics':  current['extrinsics'],
                'expressions': current['expressions'],
                'audio_deepspeech': current['audio_deepspeech'],

                'extrinsics_prv':  prv['extrinsics'],
                'expressions_prv': prv['expressions'],
                'audio_deepspeech_prv': prv['audio_deepspeech'],

                'extrinsics_nxt':  nxt['extrinsics'],
                'expressions_nxt': nxt['expressions'],
                'audio_deepspeech_nxt': nxt['audio_deepspeech'],
                }


    def __len__(self):
        return self.n_frames_total #len(self.frame_paths[0])

    def name(self):
        return 'MultiFaceAudioEQTmpDataset'
