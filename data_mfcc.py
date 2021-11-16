import librosa
from glob import glob
import os
import torchaudio
import pdb
import numpy as np

filelist = glob('./*/*.wav')
for filename in filelist:
    route_name = filename.split('/')[1]
    file_n = filename.split('/')[2].split('.wav')[0]
    os.makedirs("./%s_result"%(route_name), exist_ok = True)

    waveform, sample_rate = torchaudio.load(filename)
    specgram = torchaudio.transforms.Spectrogram(n_fft = 1024, hop_length = 256)(waveform)
    numpy_specgram = specgram.cpu().numpy()
    np.save("./%s_result/spec_%s.npy"%(route_name, file_n), numpy_specgram)
    mfcc = torchaudio.transforms.MFCC(sample_rate = 16000, n_mfcc = 40)(waveform)
    #print("Shape of MFCC : {}".format(mfcc.size()))
    numpy_mfcc = mfcc.cpu().numpy()
    np.save("./%s_result/mfcc_%s.npy"%(route_name, file_n), numpy_mfcc)


    #melspec = torchaudio.transforms.MelSpectrogram(n_fft = 1024, hop_length = 256, sample_rate = 16000, n_mels = 40)
    #pdb.set_trace()
    #print("Shape of MFCC : {}".format(melspec.size())))