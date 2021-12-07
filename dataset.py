import sys
sys.path.append('..')
import torch

from torch.utils.data.dataset import Dataset
#from torchvision.transforms import Normalize
from pathlib import Path
import pickle
import pdb
import numpy as np
import torchvision
import os
from glob import glob
from torch.autograd import Variable
class Voxceleb(Dataset):
    def __init__(self, data_folder,wav_lst,spk_lst, spk_lst_dict, mode, batch_size, wlen):
        super(Voxceleb, self).__init__()
        self.video_path = Path(data_folder)
        self.video_name_list = []
        self.video_list = []
        self.labels = []
        self.batch_size = batch_size
        self.wlen = wlen
        i = 0
        #Read data path
        
        if mode == "Train":
            for label in spk_lst:
                vid_list = os.path.join(self.video_path, "train", label)
                pkl_list = glob(vid_list + '/*')
                i = spk_lst_dict[label]
                self.video_list.extend(pkl_list)
                self.labels.extend([i]*len(pkl_list))
            
                print('%s = %i complete %i'%(label, i, len(pkl_list)))
                #i += 1
                #pdb.set_trace()
        elif mode == "Test":
            for label in spk_lst:
                vid_list = os.path.join(self.video_path, "test", label)
                pkl_list = glob(vid_list + '/*')
                i = spk_lst_dict[label]
                self.video_list.extend(pkl_list)
                self.labels.extend([i]*len(pkl_list))
            
                print('%s = %i complete %i'%(label, i, len(pkl_list)))
                #i += 1


        
        #pdb.set_trace()
    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        wlen = self.wlen
        batch_size = self.batch_size
        sig_batch=np.zeros([batch_size,wlen, wlen])
        lab_batch=np.zeros(batch_size)
        fact_amp = 0.2
        rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,self.batch_size)
        #snt_id_arr=np.random.randint(N_snt, size=self.batch_size)
        with open(str(self.video_list[idx]), 'rb') as f:
            video = np.load(f)
        video = video[:224, :]
        data = self.normalize(video)
        snt_len = data.shape[1]
        
        for i in range(self.batch_size):
            if snt_len > 225:
                snt_beg=np.random.randint(snt_len-wlen-1) #randint(0, snt_len-2*wlen-1)
                snt_end=snt_beg+wlen
                
                sig_batch[i,:,:]=data[:,snt_beg:snt_end]*rand_amp_arr[i]
                lab_batch[i]=self.labels[idx]
            else:
                sig_batch[i,:,:snt_len]=pase_data[:,:224]
                lab_batch[i]=lab_dict[wav_lst[snt_id_arr[i]]]
        #print(torch.max(output))
        #print(output.type()) # torch. DoubleTensor, 이전꺼는 torch.ByteTensor
        return sig_batch, lab_batch #, self.video_list[idx]


    def random_flip(self, video, prob):
        s = np.random.rand()
        if s < prob:
            video = np.flip(m=video, axis=2)
        return video    

    def stride_sampling(self, video, target_frames, stride):
        vid_len = len(video)

        if vid_len >= (target_frames-1)*stride + 1:
            start_idx = np.random.randint(vid_len - (target_frames-1)*stride)
            data = video[start_idx:start_idx+(target_frames-1)*stride+1:stride]
            

        elif vid_len >= target_frames:
            start_idx = np.random.randint(len(video) - target_frames)
            data = video[start_idx:start_idx + target_frames + 1]

        # Need Zero-pad
        else:
            sampled_video = []
            for i in range(0, vid_len):
                sampled_video.append(video[i])

            num_pad = target_frames - len(sampled_video)
            if num_pad>0:
                while num_pad > 0:
                    if num_pad > len(video):
                        padding = [video[i] for i in range(len(video)-1, -1, -1)]
                        sampled_video += padding
                        num_pad -= len(video)
                    else:
                        padding = [video[i] for i in range(num_pad-1, -1, -1)]
                        sampled_video += padding
                        num_pad = 0
            data = np.array(sampled_video, dtype=np.float32)
        
        return data
            
    def color_jitter(self,video):
        # range of s-component: 0-1
        # range of v component: 0-255
        s_jitter = np.random.uniform(-0.2,0.2)
        v_jitter = np.random.uniform(-30,30)
        for i in range(len(video)):
            hsv = cv2.cvtColor(video[i], cv2.COLOR_RGB2HSV)
            s = hsv[...,1] + s_jitter
            v = hsv[...,2] + v_jitter
            s[s<0] = 0
            s[s>1] = 1
            v[v<0] = 0
            v[v>255] = 255
            hsv[...,1] = s
            hsv[...,2] = v
            video[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return video


    def normalize(self, data):
        max_ = np.max(data)
        min_ = np.min(data)
        #data = data.float()
        data = (data - min_) / (max_ - min_)
        #print(data.shape, torch.max(data), torch.min(data))
        #mean = torch.mean(data.float())
        #std = torch.std(data.float())
        #print(mean, std)
        return data
