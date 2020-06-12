from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import sys
import array
import OpenEXR
import Imath

def load_exr(image_path):
    # Open the input file
    file = OpenEXR.InputFile(image_path)

    # Compute the size
    dw = file.header()['dataWindow']
    w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    #(R,G,B) = [np.array(array.array('f', file.channel(Chan, FLOAT)).tolist()).reshape((w, h, 1)) for Chan in ("R", "G", "B") ]

    (r, g, b) = file.channels("RGB")
    R = np.array(array.array('f', r).tolist()).reshape((w, h, 1))
    G = np.array(array.array('f', g).tolist()).reshape((w, h, 1))
    B = np.array(array.array('f', b).tolist()).reshape((w, h, 1))

    return np.concatenate((R, G, B), axis=2)

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        input_image = torch.clamp(input_image, -1.0, 1.0) 
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
