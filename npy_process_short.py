#from matplotlib import pyplot as plt
import json
import sys, time, os, pdb, argparse, pickle, subprocess
from glob import glob
import numpy as np
from shutil import rmtree
import pandas as pd
from collections import Counter
import copy
import csv

f = open("short_all.txt", 'r', encoding='UTF-8')
lines = f.readlines()
#lines
lines_list = [0 for i in range(len(lines))]

for i in range(len(lines)):
    lines_list[i] = lines[i].split('\n')[0]
speaker_list = []
speaker_dict = {}
for i in range(len(lines_list)):
    speaker = lines_list[i].split('/')[1] 
    if speaker not in speaker_list:
        speaker_list.append(speaker)
for i in range(len(speaker_list)):
    speaker_dict[speaker_list[i]] = i
#speaker_dict

wav_dict = {}
for lines in lines_list:
    #print(lines)
    if lines:
        wav_dict[lines] = speaker_dict[lines.split('/')[1]]
print(wav_dict)
np.save("voxceleb.npy", wav_dict)