import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from util import util
from scipy.misc import imresize

import torch
import numpy as np
from PIL import Image
import time
import cv2


def save_tensor_image(input_image, image_path):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_augmentation = True    # no flip
    opt.display_id = -1   # no visdom display

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#test images = %d' % dataset_size)
    print('#train objects = %d' % opt.nTrainObjects)
    print('#test objects = %d' % opt.nTestObjects)


    print('>>> create model <<<')
    model = create_model(opt)
    print('>>> setup model <<<')
    model.setup(opt)
    #save_tensor_image(model.texture.data[0:1,0:3,:,:], 'load_test1.png')


    sum_time = 0
    total_runs = dataset_size
    warm_up = 50

    # create a website
    web_dirs = []
    webpages = []
    file_expressions = []
    file_fake_expressions = []
    file_rigids = []
    file_intrinsics = []
    file_identities = []
    video_writer = []

    print('>>> create a websites and output directories <<<')
    for i in range(0, opt.nTestObjects):
        #web_dir = os.path.join(opt.results_dir, opt.name, '%s--%s__%s_%s' % (opt.test_sequence_names[i][0], opt.test_sequence_names[i][1], opt.phase, opt.epoch) )
        web_dir = os.path.join(opt.results_dir, opt.name, '%s--%s' % (opt.test_sequence_names[i][0], opt.test_sequence_names[i][1]) )
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

        ##
        output_file_expressions=None
        output_file_fake_expressions=None
        output_file_rigids = None
        output_file_intrinsics = None
        output_file_identities = None
        if hasattr(model, 'fake_expressions'):
            output_file_expressions=open(os.path.join(web_dir, 'gt_expressions.txt'), 'w')
            output_file_fake_expressions=open(os.path.join(web_dir, 'expression.txt'), 'w')
            output_file_rigids=open(os.path.join(web_dir, 'rigid.txt'), 'w')
            output_file_intrinsics=open(os.path.join(web_dir, 'intrinsics.txt'), 'w')
            output_file_identities=open(os.path.join(web_dir, 'identities.txt'), 'w')
        ##

        web_dirs.append(web_dir)
        webpages.append(webpage)
        file_expressions.append(output_file_expressions)
        file_fake_expressions.append(output_file_fake_expressions) 
        file_rigids.append(output_file_rigids)
        file_intrinsics.append(output_file_intrinsics)
        file_identities.append(output_file_identities)

        if opt.write_video:
            writer = cv2.VideoWriter(web_dir+'.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 25.0,(opt.display_winsize,opt.display_winsize))
            video_writer.append(writer)

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        #if i >= opt.num_test:
        #    break
        model.set_input(data)
        test_sequence_id = data['internal_id'].cpu()
        

        torch.cuda.synchronize()
        a = time.perf_counter()
        
        model.test()

        torch.cuda.synchronize() # added sync
        b = time.perf_counter()

        if i > warm_up:  # give torch some time to warm up
            sum_time += ((b-a) * 1000)

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        #if not hasattr(model, 'fake_expressions'):
        if not opt.write_no_images:
            save_images(webpages[test_sequence_id], visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

        if opt.write_video:
            fake = visuals['fake']
            im = util.tensor2im(fake)
            im = imresize(im, (opt.display_winsize,opt.display_winsize), interp='bicubic')
            im = np.concatenate([im[:,:,2:3], im[:,:,1:2], im[:,:,0:1]],axis=2)
            #video_writer[test_sequence_id].write(np.random.randint(0, 255, (opt.display_winsize,opt.display_winsize,3)).astype('uint8'))
            video_writer[test_sequence_id].write(im.astype('uint8'))

        if hasattr(model, 'fake_expressions'):
            #print('contains fake expressions')
            np.savetxt(file_fake_expressions[test_sequence_id], model.fake_expressions.data.cpu().numpy(), delimiter=' ')
            if hasattr(model, 'expressions'):
                np.savetxt(file_expressions[test_sequence_id], model.expressions.data.cpu().numpy(), delimiter=' ')

            np.savetxt(file_rigids[test_sequence_id], data['extrinsics'][0].data.cpu().numpy(), delimiter=' ')
            np.savetxt(file_intrinsics[test_sequence_id], data['intrinsics'].data.cpu().numpy(), delimiter=' ')
            np.savetxt(file_identities[test_sequence_id], data['identity'].data.cpu().numpy(), delimiter=' ')
            

        #if i < 50:
        #    if hasattr(model, 'face_model'):
        #        image_dir = webpage.get_image_dir()
        #        name = os.path.splitext(short_path)[0]
        #
        #        filename_mesh = name + '.obj'
        #        #filename_tex = name + '_audio_texture.png'
        #        #filename_mat = name + '.mtl'
        #        #model.face_model.save_model_to_obj_file(image_dir, filename_mesh, filename_mat, filename_tex)
        #        model.face_model.save_model_to_obj_file(image_dir, filename_mesh)

    print('mean eval time (ms): ', (sum_time / (total_runs - warm_up)))

    # save the website
    for webpage in webpages:
        webpage.save()

    if opt.write_video:
        for writer in video_writer:
            writer.release()
