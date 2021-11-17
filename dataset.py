import sys
sys.path.append('..')
import torch
from torch.utils.data.dataset import Dataset
from pathlib import Path
import pickle
import pdb
#import cv2
import numpy as np
import torchvision
import os
from glob import glob
# import albumentations as A

class seoulmal(Dataset):
    def __init__(self, data_dir, sample_stride=1, num_workers=24, image_size=256, norm_value=255,
                multiscale_crop=4, temporal_stride=1):
        super(seoulmal, self).__init__()
        self.video_path = Path(data_dir)
        #self.data_partition = data_partition    
        #self.clip_len = clip_len
        self.image_size = image_size
        self.norm_value = norm_value
        self.video_name_list = []
        self.video_list = []
        self.labels = []
        i = 0
        #Read data path
        self
        for label in os.path.basename(folder_dir):
            vid_list = os.path.join(self.video_path, label)
            np_list = glob(vid_list + '/*')   
            self.video_list.extend(pkl_list)
            self.labels.extend([i]*len(pkl_list))
            
            print('%s = %i complete %i'%(label, i, len(pkl_list)))
            i += 1
        
            pdb.set_trace()
        pdb.set_trace()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        image_list = []
        
        with open(str(self.video_list[idx]), 'rb') as f:
            #print(self.video_list[idx])
            #self.video_name_list.append(self.video_list[idx])
            video = pickle.load(f)
        
       
        data = self.stride_sampling(video, self.clip_len, self.temporal_stride)
        # data = video[start_idx:start_idx + self.clip_len]
        data = self.color_jitter(data)
        data = self.random_flip(data, prob=0.5)

        for image_ in data:
            image_list.append(torch.from_numpy(image_.transpose(-1,0,1).copy()))

        return self.normalize(torch.stack(image_list)), self.labels[idx], self.video_list[idx]


    def normalize(self, data):
        mean = torch.mean(data.float())
        std = torch.std(data.float())
        return (data-mean) / std
