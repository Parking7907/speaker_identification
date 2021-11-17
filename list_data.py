import sys, time, os, pdb, argparse, pickle, subprocess
from glob import glob
import numpy as np
from shutil import rmtree
import re
i=0
#print("YAHO")
#pdb.set_trace()
#/home/jinyoung/share/SincNet/seoul_mal/20s_female_1/fv01/
#command = ("glob("/home/nas/DB/RWF-2000_Dataset/Validation/Fight")") 
#output = subprocess.call(command, shell=True, stdout=None)    
Train = glob("/home/data/jinyoung/classification/output_short_noisy_VAD/train/*/*")
Test = glob("/home/data/jinyoung/classification/output_short_noisy_VAD/test/*/*")
output = "/home/jinyoung/speaker_identification/voxceleb/"
# os.makedirs(new_file_out, exist_ok = True)
f = open("./short_train.txt", 'w')
f2 = open("./short_all.txt", 'w')
f3 = open("./short_test.txt", 'w')
#print(Test)
for name in Train:
    vid_n = os.path.basename(name)
    print(name.split('output_short_noisy_VAD/')[1])
    data = name.split('output_short_noisy_VAD/')[1]
    f.write(data+'\n')
    f2.write(data+'\n')
    
f.close()
for name in Test:
    vid_n = os.path.basename(name)
    print(name.split('output_short_noisy_VAD/')[1])
    data = name.split('output_short_noisy_VAD/')[1] 
    f3.write(data + '\n')
    f2.write(data + '\n')
    
f2.close()
f3.close()
    #expan = nn[1]
    #command = ("cp -r %s /home/jinyoung/share/SincNet/seoul_mal/"%(name))
    #output = subprocess.call(command, shell=True, stdout=None)
#for name in Test:
#    vid_n = os.path.basename(name)
#    print(vid_n)
    #nn = vid_n.split(".")
    #expan = nn[1]
#    command = ("cp -r %s /home/jinyoung/share/SincNet/seoul_mal/"%(name))
#    output = subprocess.call(command, shell=True, stdout=None)        

'''
for name in Test:
    vid_n = os.path.basename(name)
    print(vid_n)
    nn = vid_n.split(".")
    expan = nn[1]
    if expan != 'py':
        numbers = re.sub(r'[^0-9]', '', vid_n)
        Validation_name = numbers + '.' + expan
        print(Validation_name)
        command = ("mv /home/jinyoung/share/car_accident_dataset/Validation/%s /home/jinyoung/share/car_accident_dataset/Validation/%s"%(vid_n,Validation_name))
        output = subprocess.call(command, shell=True, stdout=None)    
'''


