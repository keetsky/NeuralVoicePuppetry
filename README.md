# NeuralVoicePuppetry

## Data / Preprocessing

This repository assumes that you have a running face tracker that can reconstruct a 3D face model based on the training RGB video sequences.
Based on this visual tracking the Audio2ExpressionNet network as well as the rendering network is trained.

To extract the corresponding per frame audio features, we use the DeepSpeech 0.1.0 pretrained model.
The resampling to the video fps is based on the code provided by the Voca repository.
[(Github repo)](https://github.com/TimoBolkart/voca/blob/9e2a759eed0a0e6a75ee0c22d2e09b819f3b420b/utils/inference.py#L32).

Note, because of legal issues, we are not allowed to share our 3D face model as well as the training video corpus.
The 3D face model is based on the Basel face model and can be downloaded here: https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads
The videos used for training are from the German public media, a download list is provided in the datasets subfolder.

## Audio2ExpressionNet

In the Audio2ExpressionNet subfolder you will find inference code with a pretrained model and the training code itself.
It allows you to map audio features to a blendshape model, by learning a linear mapping from the actual audio expression space to the blendshape model.
This should work with any face blendshape model that you have.
Note that there are two options to learn this mapping (one based on the actual vertex displacements (preferred) and one in parameter space (default)).

## Neural Rendering Network

The model for our neural rendering is provided in the 'Neural Rendering Network' subfolder.
It can be integrated into the Pix2Pix/CycleGan framework.
Note that you need a renderer that renders uv maps.

The code also contains implementations of neural textures that are conditioned e.g. on the audio feature inputs.
They are called dynamic neural textures.

## Ackowledgements

This code is based on the Pix2Pix/CycleGAN framework [(Github repo)](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).