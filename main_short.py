import torch
import os
import math
from glob import glob
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys
import numpy as np
from model import MyNet
import argparse
import sys
import yaml
from torch.utils.data import DataLoader
from data_io import ReadList,read_conf,str_to_bool
from loss_functions import AngularPenaltySMLoss
import pdb
import logging
#from trainer import ModelTrainer
#from tester import ModelTester
#from dataset import seoulmal

#create_batches_rnd(batch_size,data_folder,wav_lst_tr,snt_tr,wlen,lab_dict,0.2)
def create_batches_rnd(batch_size,data_folder,wav_lst,N_snt,wlen,lab_dict,fact_amp, normalize_size):
 # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    sig_batch=np.zeros([batch_size,wlen, wlen])
    lab_batch=np.zeros(batch_size)
    
    snt_id_arr=np.random.randint(N_snt, size=batch_size)
    rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,batch_size)
    p = 0
    for i in range(batch_size):
        #pdb.set_trace()
        pase_data = np.load(data_folder+wav_lst[snt_id_arr[i]])
        pase_data = pase_data[:224, :] / np.max(pase_data)

        # accesing to a random chunk
        snt_len=pase_data.shape[1]
        if snt_len > 225:
            snt_beg=np.random.randint(snt_len-wlen-1) #randint(0, snt_len-2*wlen-1)
            snt_end=snt_beg+wlen
            
            sig_batch[i,:,:]=pase_data[:,snt_beg:snt_end]*rand_amp_arr[i]
            lab_batch[i]=lab_dict[wav_lst[snt_id_arr[i]]]
            #p+=1
        #elif snt_len > 111:
        #    sig_batch[i,:,:snt_len]=pase_data[:,:224]
        else:
            sig_batch[i,:,:snt_len]=pase_data[:,:224]
            lab_batch[i]=lab_dict[wav_lst[snt_id_arr[i]]]
            #p+=1
            #print("warning : ", data_folder+wav_lst[snt_id_arr[i]])
            #print("length : ", str(snt_len))
  
    batch=Variable(torch.from_numpy(sig_batch).float().cuda().contiguous())
    label=Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())
    #pdb.set_trace()
    return batch,label
    


if __name__ == '__main__':
    #####################################################################Setup#######################################################################
    options=read_conf()
    tr_lst=options.tr_lst
    te_lst=options.te_lst
    pt_file=options.pt_file
    class_dict_file=options.lab_dict
    lab_dict=np.load(class_dict_file, allow_pickle=True).item()
    #pdb.set_trace()
    data_folder=options.data_folder+'/'
    output_folder=options.output_folder

    #[windowing]
    fs=int(options.fs)
    cw_len=int(options.cw_len)
    cw_shift=int(options.cw_shift)
    class_lay=list(map(int, options.class_lay.split(',')))
    #[optimization]
    lr=float(options.lr)
    batch_size=int(options.batch_size)
    N_epochs=int(options.N_epochs)
    N_batches=int(options.N_batches)
    N_eval_epoch=int(options.N_eval_epoch)
    seed=int(options.seed)


    # training list
    wav_lst_tr=ReadList(tr_lst)
    snt_tr=len(wav_lst_tr)

    # test list
    wav_lst_te=ReadList(te_lst)
    snt_te=len(wav_lst_te)


    # Folder creationt
    try:
        os.stat(output_folder)
    except:
        os.mkdir(output_folder) 
        
        
    # setting seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # loss function
    # in_features = input feature dimension, out_features = classes갯수
    in_features = 512
    out_features = int(options.class_lay)
    cost = AngularPenaltySMLoss(in_features, out_features, loss_type='arcface')
    
    
    # Converting context and shift in samples
    wlen=int(cw_len)
    wshift=int(cw_shift)
    
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(output_folder, 'training.log'))
    sh = logging.StreamHandler(sys.stdout)
    logger.addHandler(fh)
    logger.addHandler(sh)






    ##################################################TRAIN###################################
    pdb.set_trace()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    cost = cost.to(device)
    model = MyNet(out_features).to(device)
    best_error = 1
    wav_lst_te.sort()
    for epoch in range(N_epochs):
        test_flag=0
        model.train()
        loss_sum=0
        err_sum=0
        #MyNet_net = MLP(MyNet_net)
        optimizer_MyNet = optim.RMSprop(model.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
        # Batch_dev
        Batch_dev=128
        normalize_size = 12.5
        for i in range(N_batches):
            [inp,lab]=create_batches_rnd(batch_size,data_folder,wav_lst_tr,snt_tr,wlen,lab_dict,0.2,normalize_size)
            #inp = inp.to(device=cuda)
            pout=model(inp)
            pout = pout.to(device)
            lab = lab.long().to(device)
            wf, loss = cost(pout, lab)
            pred=torch.max(wf,dim=1)[1]
            #loss = cost(pout, lab.long())
            err = torch.mean((pred!=lab.long()).float())
    
            optimizer_MyNet.zero_grad()
        
            loss.backward()
            optimizer_MyNet.step()
            
            loss_sum=loss_sum+loss.detach()
            err_sum=err_sum+err.detach()
            if i % 400 ==0:
                logger.info("epoch: {}, batch {} / {} --- t_loss : {:0.3f}".format(epoch, i, N_batches, loss))

        loss_tot=loss_sum/N_batches
        err_tot=err_sum/N_batches
        if epoch % N_eval_epoch==0:
            model.eval()
            cost.eval()
            test_flag=1 
            loss_sum=0
            err_sum=0
            err_sum_snt=0
            with torch.no_grad():
                for i in range(snt_te):
                    pase_data = np.load(data_folder+wav_lst_te[i]) # 256 X length
                    #print(wav_lst_te[i])
                    pase_data = torch.from_numpy(pase_data[:224, :]).cuda().contiguous() # 224 * length
                    lab_batch = lab_dict[wav_lst_te[i]] # 0번화자 1번화자..(int)
                    beg_samp = 0
                    end_samp = wlen
                    
                    count_fr = 0
                    count_fr_tot = 0
                    if pase_data.shape[1]-wlen >= 0:
                        N_fr = int((pase_data.shape[1]-wlen-1)/(wshift))
                        sig_arr = torch.zeros([Batch_dev, wlen, wlen]).float().cuda().contiguous()
                        lab=Variable((torch.zeros(N_fr+1)+lab_batch).cuda().contiguous().long())
                        pout=Variable(torch.zeros(N_fr+1,class_lay[-1]).float().cuda().contiguous())
                        wf = Variable(torch.zeros(N_fr+1,out_features).float().cuda().contiguous())
                    else:
                        N_fr = 0
                        sig_arr = torch.zeros([Batch_dev, wlen, wlen]).float().cuda().contiguous()
                        lab=Variable((torch.zeros(N_fr+1)+lab_batch).cuda().contiguous().long())
                        pout=Variable(torch.zeros(N_fr+1,class_lay[-1]).float().cuda().contiguous())
                        wf = Variable(torch.zeros(N_fr+1,out_features).float().cuda().contiguous())
                        sig_arr[count_fr,:,:pase_data.shape[1]] = pase_data[:,:224]
                        count_fr=count_fr+1
                        count_fr_tot=count_fr_tot+1
                    # accesing to a random chunk
                    #pout = lab size X 117
                    while end_samp<pase_data.shape[1]:
                        sig_arr[count_fr,:,:]=pase_data[:,beg_samp:end_samp]
                        #sig_arr = 128 X 224 X 224, pase data 1개씩 넣기
                        beg_samp=beg_samp+wshift
                        end_samp=beg_samp+wlen
                        count_fr=count_fr+1
                        count_fr_tot=count_fr_tot+1
                        if count_fr==Batch_dev:
                            #pdb.set_trace()
                            lab_ = lab[count_fr_tot-Batch_dev:count_fr].long().to(device)
                            inp_=Variable(sig_arr)
                            p_2 = model(inp_)
                            wf, loss = cost(p_2, lab_)
                            pout[count_fr_tot-Batch_dev:count_fr_tot,:]=wf
                            pout = pout.to(device)
                            #pred=torch.max(wf,dim=1)[1]
                            count_fr = 0
                            sig_arr=torch.zeros([Batch_dev,wlen,wlen]).float().cuda().contiguous()
                    if count_fr > 0:
                        inp = Variable(sig_arr[0:count_fr]) # countfr X 224 X 224
                        p_ = model(inp) # p_ = 10 X 512
                        b_len = p_.shape[0]
                        lab_ = lab[count_fr_tot-b_len:count_fr_tot].long().to(device)
                        wf, loss = cost(p_, lab_) # wf = 10 X 117, p_ 1 X 512, 
                        #pdb.set_trace()
                        pout[count_fr_tot-count_fr:count_fr_tot,:] = wf # pout.shape = 10 X 117, 
                    #wf, loss = cost(pout, lab)
                    pred = torch.max(pout, dim=1)[1] # 10, pred = [26, 26, 26...]
                    err = torch.mean((pred!=lab.long()).float())

                    [val,best_class]=torch.max(torch.sum(pout,dim=0),0) # val = 0.7708, best class => snt 결과
                    err_sum_snt =err_sum_snt+(best_class!=lab[0]).float()

                    loss_sum=loss_sum+loss.detach()
                    err_sum=err_sum+err.detach()
                err_tot_dev_snt=err_sum_snt/snt_te
                loss_tot_dev=loss_sum/snt_te
                err_tot_dev=err_sum/snt_te
            print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f" % (epoch, loss_tot,err_tot,loss_tot_dev,err_tot_dev,err_tot_dev_snt))

            with open(output_folder+"/res.res", "a") as res_file:
                res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f\n" % (epoch, loss_tot,err_tot,loss_tot_dev,err_tot_dev,err_tot_dev_snt))   
            checkpoint = {'model_par' : model.state_dict(),
                        'loss_par' : cost.state_dict(),}
            if err_tot_dev_snt < best_error:
                torch.save(checkpoint,output_folder+'/model_raw.pkl')
                print("New best model", str(err_tot_dev_snt))
                best_error = err_tot_dev_snt
    
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type=str, default='.', help='Root directory')
    parser.add_argument('-c', '--config', type=str, help='Path to option YAML file.')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset')
    parser.add_argument('-m', '--mode', type=str, help='Train or Test')
    args = parser.parse_args()
    '''
    #data_folder = setup()
    #data = seoulmal(data_folder, 'train')
    #print(data)