import time
import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import torchaudio
import torchaudio.transforms

import librosa
import scipy.signal
import librosa.display
import matplotlib.pyplot as plt


class Audio():
    def name(self):
        return 'Audio'

    def __init__(self, filename, write_mel_spectogram = False):
        self.n_mels=128
        self.fmax=8000
        self.hop_length_ms = 20

        sound, sample_rate = librosa.load(filename)#torchaudio.load(filename)
        self.raw_audio = sound
        self.sample_rate = sample_rate
        print('sample_rate = %d' % self.sample_rate)
        self.n_samples = sound.shape[0]
        self.time_total = self.n_samples / self.sample_rate
        print('length = %ds' % self.time_total)

        print('compute mel spectrogram...')
        self.hop_length = int(sample_rate / 1000.0 * self.hop_length_ms)
        print('hop_length: ', self.hop_length)
        self.mel_spectrogram = librosa.feature.melspectrogram(y=self.raw_audio, sr=self.sample_rate, hop_length=self.hop_length, n_mels=self.n_mels, fmax=self.fmax)

        
        if write_mel_spectogram:
            print('write spectrogram to file')
            plt.figure(figsize=(100, 15))
            librosa.display.specshow(librosa.power_to_db(self.mel_spectrogram, ref=np.max), y_axis='mel', fmax=self.fmax, x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel spectrogram')
            plt.tight_layout()
            plt.savefig('mel_features.png', dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)

        print('mel: ', self.mel_spectrogram.shape) # (128, 18441)
        self.n_mel_frames = self.mel_spectrogram.shape[1]
        self.mel_sample_rate = self.mel_spectrogram.shape[1] / self.time_total
        print('n_mel_frames: ', self.n_mel_frames)
        print('mel_sample_rate: ', self.mel_sample_rate)

        # convert to torch
        self.mel_spectrogram = torch.FloatTensor(self.mel_spectrogram)

    def getWindow(self, mel_frame_idx, window_size):
        # get audio mel sample window
        audio_start = mel_frame_idx - (window_size//2)
        audio_end = mel_frame_idx + (window_size//2)
        if audio_start < 0:
            audio_input = self.mel_spectrogram[0:self.n_mels, 0:audio_end]
            zeros = torch.zeros((self.n_mels,-audio_start))
            audio_input = torch.cat([zeros, audio_input], 1)
        elif audio_end >= self.n_mel_frames:
            audio_input = self.mel_spectrogram[:, audio_start:-1]
            zeros = torch.zeros((self.n_mels,audio_end-self.n_mel_frames + 1))
            audio_input = torch.cat([audio_input, zeros], 1)
        else:
            audio_input = self.mel_spectrogram[:, audio_start:audio_end]

        return torch.reshape(audio_input, (1, 1, self.n_mels, window_size))