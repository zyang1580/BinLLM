from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd


from minigpt4.models.rec_model import MatrixFactorization
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import numpy as np
import torch.optim
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
import random 

import os
# os.environ["CUDA_VISIBLE_DEVICES"]='2'



from minigpt4.tasks import base_task
import time
import numpy as np
# import ray

class my_sign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        re = torch.ones_like(inputs)
        re[inputs<0]= -1.0
        return re
        # return (ctx>0).float() 
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

class binarize(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self,x):
        re = my_sign.apply(x)
        return re 



class hashGNN(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rec_model = MatrixFactorization(config=config)
        self.emb_dim = config.embedding_size
        self.proj_u = nn.Linear(self.emb_dim,self.emb_dim)
        self.proj_i = nn.Linear(self.emb_dim,self.emb_dim)
        self.act = nn.Tanh()
        self.binarize = binarize()
    
    def beta_init(self):
        self.beta = 0.0
        self.beta_updatable = True

    def update_beta(self):
        if self.beta_updatable and self.beta < 1.01:
            self.beta += 0.02
        else:
            self.beta_updatable = False
            self.beta = 0
    
    def computer(self):
        u_emb = self.rec_model.user_embedding.weight
        i_emb = self.rec_model.item_embedding.weight
        u_z = self.act(self.proj_u(u_emb))
        i_z = self.act(self.proj_i(i_emb))
        self.hash_u = self.binarize(u_z)
        self.hash_i = self.binarize(i_z)
    
    def user_encoder(self,idx):
        return self.hash_u[idx]
    def item_encoder(self,idx):
        return self.hash_u[idx]


    def forward(self,user,item):
        u_emb = self.rec_model.user_encoder(user)
        i_emb = self.rec_model.item_encoder(item)

        z_u = self.act(self.proj_u(u_emb))
        z_i = self.act(self.proj_i(i_emb))

        h_u = self.binarize(z_u)
        h_i = self.binarize(z_i)

        h_u_shape = h_u.shape

        # selector = self.beta < torch.rand_like(h_u) 
        # o_u = torch.where(selector, z_u, h_u)
        # o_i = torch.where(selector, z_i, h_i)
        if self.training:
            # selector = self.beta < torch.rand([np.prod(h_u_shape[:-1])]) 
            # selector = selector.reshape(-1,1).repeat(1,h_u_shape[-1]).cuda()
            selector = self.beta < torch.rand_like(h_u) 
            o_u = torch.where(selector, z_u, h_u)
            o_i = torch.where(selector, z_i, h_i)
        else:
            o_u = h_u
            o_i = h_i
        matching = torch.mul(o_u,o_i).sum(dim=-1)
        return matching


def uAUC_me(user, predict, label):
    if not isinstance(predict,np.ndarray):
        predict = np.array(predict)
    if not isinstance(label,np.ndarray):
        label = np.array(label)
    predict = predict.squeeze()
    label = label.squeeze()

    start_time = time.time()
    u, inverse, counts = np.unique(user,return_inverse=True,return_counts=True) # sort in increasing
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    for u_i in u:
        start_id,end_id = total_num, total_num+counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts ==1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        # print(index_ui, predict.shape)
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]
        
        k+=1
    print("only one interaction users:",only_one_interaction)
    auc=[]
    only_one_class = 0

    for ui,pre_and_true in candidates_dict.items():
        pre_i,label_i = pre_and_true
        try:
            ui_auc = roc_auc_score(label_i,pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except:
            only_one_class += 1
            # print("only one class")
        
    auc_for_user = np.array(auc)
    print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    print("uauc for validation Cost:", time.time()-start_time,'uauc:', uauc)
    return uauc, computed_u, auc_for_user


class early_stoper(object):
    def __init__(self,ref_metric='valid_auc', incerase =True,patience=20) -> None:
        self.ref_metric = ref_metric
        self.best_metric = None
        self.increase = incerase
        self.reach_count = 0
        self.patience= patience
        # self.metrics = None
    
    def _registry(self,metrics):
        self.best_metric = metrics

    def update(self, metrics):
        if self.best_metric is None:
            self._registry(metrics)
            return True
        else:
            if self.increase and metrics[self.ref_metric] > self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True
            elif not self.increase and metrics[self.ref_metric] < self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True 
            else:
                self.reach_count += 1
                return False

    def is_stop(self):
        if self.reach_count>=self.patience:
            return True
        else:
            return False

# set random seed   
def run_a_trail(train_config,log_file=None, save_mode=False,save_file=None,need_train=True,warm_or_cold=None,need_uauc=False):
    seed=2023
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # load dataset
  
    data_dir = "/data/datasets/book/"
    print("data dir:", data_dir, file=log_file)
    train_data = pd.read_pickle(data_dir+"train_ood2.pkl")[['uid','iid','label']].values
    valid_data = pd.read_pickle(data_dir+"valid_ood2.pkl")[['uid','iid','label']].values
    test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label']].values

    # train_config={
    #     "lr": 1e-2,
    #     "wd": 1e-4,
    #     "epoch": 5000,
    #     "eval_epoch":1,
    #     "patience":50,
    #     "batch_size":1024
    # }

    user_num = max(train_data[:,0].max(), valid_data[:,0].max(), test_data[:,0].max()) + 1
    item_num =  max(train_data[:,1].max(), valid_data[:,1].max(), test_data[:,1].max()) + 1

    # if warm_or_cold is not None:
    #     if warm_or_cold == 'warm':
    #         test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label', 'not_cold']]
    #         test_data = test_data[test_data['not_cold'].isin([1])][['uid','iid','label']].values
    #         print("warm data size:", test_data.shape[0])
    #         # pass
    #     else:
    #         test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label', 'not_cold']]
    #         test_data = test_data[test_data['not_cold'].isin([0])][['uid','iid','label']].values
    #         print("cold data size:", test_data.shape[0])
    #         # pass
    if warm_or_cold is not None:
        if warm_or_cold == 'warm':
            test_data = pd.read_pickle(data_dir+"test_warm_cold_ood2.pkl")[['uid','iid','label', 'warm']]
            test_data = test_data[test_data['warm'].isin([1])][['uid','iid','label']].values
            print("warm data size:", test_data.shape[0])
            # pass
        else:
            test_data = pd.read_pickle(data_dir+"test_warm_cold_ood2.pkl")[['uid','iid','label', 'cold']]
            test_data = test_data[test_data['cold'].isin([1])][['uid','iid','label']].values
            print("cold data size:", test_data.shape[0])
            # pass
    

    print("user nums:", user_num, "item nums:", item_num)

    mf_config={
        "user_num": int(user_num),
        "item_num": int(item_num),
        "embedding_size": int(train_config['embedding_size'])
        }
    mf_config = omegaconf.OmegaConf.create(mf_config)

    train_data_loader = DataLoader(train_data, batch_size = train_config['batch_size'], shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size = train_config['batch_size'], shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size = train_config['batch_size'], shuffle=False)





    model = hashGNN(mf_config).cuda()
    opt = torch.optim.Adam(model.parameters(),lr=train_config['lr'],weight_decay=train_config['wd'])
    early_stop = early_stoper(ref_metric='valid_auc',incerase=True,patience=train_config['patience'])
    # trainig part
    criterion = nn.BCEWithLogitsLoss()

    if not need_train:
        model.load_state_dict(torch.load(save_file))
        model.eval()
        pre=[]
        label = []
        users = []
        for batch_id,batch_data in enumerate(valid_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
            users.extend(batch_data[:,0].cpu().numpy())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[:,-1].cpu().numpy())
        valid_auc = roc_auc_score(label,pre)
        valid_uauc = 0
        if need_uauc:
            valid_uauc, _, _ = uAUC_me(users, pre, label)
        label = np.array(label)
        pre = np.array(pre)
        thre = 0.1
        pre[pre>=thre] =  1
        pre[pre<thre]  =0
        val_acc = (label==pre).mean()

        pre=[]
        label = []
        users = []
        for batch_id,batch_data in enumerate(test_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[:,-1].cpu().numpy())
            users.extend(batch_data[:,0].cpu().numpy())
        test_auc = roc_auc_score(label,pre)
        test_uauc = 0
        if need_uauc:
            test_uauc, _, _ = uAUC_me(users, pre, label)
        
            

        print("valid_auc:{}, valid_uauc:{}, test_auc:{}, test_uauc:{}, acc: {}".format(valid_auc, valid_uauc, test_auc, test_uauc, val_acc))
        return
    
    model.beta_init()
    for epoch in range(train_config['epoch']):
        model.train()
        for bacth_id, batch_data in enumerate(train_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
            loss = criterion(ui_matching, batch_data[:,-1].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        if epoch% train_config['eval_epoch']==0:
            model.eval()
            pre=[]
            label = []
            users = []
            for batch_id,batch_data in enumerate(valid_data_loader):
                batch_data = batch_data.cuda()
                ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
                users.extend(batch_data[:,0].cpu().numpy())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:,-1].cpu().numpy())
            valid_auc = roc_auc_score(label,pre)
            valid_uauc = 0
            if need_uauc:
                valid_uauc, _, _ = uAUC_me(users, pre, label)

            pre=[]
            label = []
            users = []
            for batch_id,batch_data in enumerate(test_data_loader):
                batch_data = batch_data.cuda()
                ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
                users.extend(batch_data[:,0].cpu().numpy())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:,-1].cpu().numpy())
            test_auc = roc_auc_score(label,pre)
            
            test_uauc = 0
            if need_uauc:
                test_uauc, _, _ = uAUC_me(users, pre, label)

            updated = early_stop.update({'valid_auc':valid_auc, 'valid_uauc':valid_uauc,'test_auc':test_auc, 'test_uauc':test_uauc, 'epoch':epoch})
            if updated and save_mode:
                torch.save(model.state_dict(),save_file)


            print("epoch:{}, valid_auc:{}, test_auc:{}, early_count:{}".format(epoch, valid_auc, test_auc, early_stop.reach_count))
            if early_stop.is_stop():
                print("early stop is reached....!")
                # print("best results:", early_stop.best_metric)
                break
            if epoch>500 and early_stop.best_metric[early_stop.ref_metric] < 0.52:
                print("training reaches to 500 epoch but the valid_auc is still less than 0.55")
                break
        if epoch%2 ==0:
            model.update_beta()
    print("train_config:", train_config,"\nbest result:",early_stop.best_metric) 
    if log_file is not None:
        print("train_config:", train_config, "best result:", early_stop.best_metric, file=log_file)
        log_file.flush()
    

# if __name__=='__main__':
#     # lr_ = [1e-1,1e-2,1e-3]
#     lr_=[1e-1]
#     dw_ = [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]
#     # embedding_size_ = [32, 64, 128, 156, 512]
#     embedding_size_ = [32,64,128,256]
#     try:
#         f = open("0103_hash_mf_lr"+str(lr_[0])+".log",'rw+')
#     except:
#         f = open("0103_hash_mf_lr"+str(lr_[0])+".log",'w+')
#     for lr in lr_:
#         for wd in dw_:
#             for embedding_size in embedding_size_:
#                 train_config={
#                     'lr': lr,
#                     'wd': wd,
#                     'embedding_size': embedding_size,
#                     "epoch": 5000,
#                     "eval_epoch":1,
#                     "patience":100,
#                     "batch_size":2048
#                 }
#                 print(train_config)
#                 run_a_trail(train_config=train_config, log_file=f, save_mode=False,need_uauc=False)
#     f.close()



# {'lr': 0.001, 'wd': 0.0001, 'embedding_size': 256, 'epoch': 5000, 'eval_epoch': 1, 'patience': 100, 'batch_size': 2048},
#  {'valid_auc': 0.6760080227104877, 'valid_uauc': 0.6191863368703151, 'test_auc': 0.6482002627476354, 'test_uauc': 0.636100123360848, 'epoch': 465}
# save version....
# if __name__=='__main__':
#     # lr_ = [1e-1,1e-2,1e-3]
#     lr_=[1e-3] #1e-2
#     dw_ = [1e-4]
#     # embedding_size_ = [32, 64, 128, 156, 512]
#     embedding_size_ = [32]
#     # save_path = "/data/LLM/PretrainedModels/mf/"
#     save_path = "/home/sist/LLM/PretrainedModels/hash/"
#     # try:
#     #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'rw+')
#     # except:
#     #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'w+')
#     f=None
#     for lr in lr_:
#         for wd in dw_:
#             for embedding_size in embedding_size_:
#                 train_config={
#                     'lr': lr,
#                     'wd': wd,
#                     'embedding_size': embedding_size,
#                     "epoch": 5000,
#                     "eval_epoch":1,
#                     "patience":100,
#                     "batch_size":2048
#                 }
#                 print(train_config)
#                 save_path += "0104hashGNN" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"
#                 # save_path = None
#                 print("save path: ", save_path)
#                 run_a_trail(train_config=train_config, log_file=f, save_mode=True,save_file=save_path)
#     f.close()




#### /data/LLM/PretrainedModels/mf/best_model_d128.pth
# with prtrain version:
if __name__=='__main__':
    # lr_ = [1e-1,1e-2,1e-3]
    lr_=[1e-3] #1e-2
    dw_ = [1e-4]
    # embedding_size_ = [32, 64, 128, 156, 512]
    embedding_size_ = [32]
    save_path = "/data/LLM/PretrainedModels/mf/"
    # try:
    #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'rw+')
    # except:
    #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'w+')
    f=None
    for lr in lr_:
        for wd in dw_:
            for embedding_size in embedding_size_:
                train_config={
                    'lr': lr,
                    'wd': wd,
                    'embedding_size': embedding_size,
                    "epoch": 5000,
                    "eval_epoch":1,
                    "patience":50,
                    "batch_size":1024
                }
                print(train_config)
                # save_path = "/data/LLM/PretrainedModels/mf/0912_ml100k_oodv2_best_model_d64lr-0.001wd0.0001.pth"
                # save_path = "/data/LLM/PretrainedModels/mf/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth"
                save_path = "/data1/cf4reclog/0104hashGNN32lr-0.001wd0.0001.pth"
                # if os.path.exists(save_path + "0912_ml100k_oodv2_best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"):
                #     save_path += "0912_ml100k_oodv2_best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"
                #     print(save_path)
                # else:
                #     save_path += "best_model_d" + str(embedding_size) + ".pth"
                
                run_a_trail(train_config=train_config, log_file=f, save_mode=False,save_file=save_path,need_train=False,warm_or_cold=None,need_uauc=True)
    if f is not None:
        f.close()
        





        
            






