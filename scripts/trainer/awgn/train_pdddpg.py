import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import random
import gym
import copy
import pandas as pd
from method.awgn.pdddpg_awgn import PD_DDPG

def trainer( env, num_exp, num_iterations, num_users, pow_max, priority_weights, lr_x, lr_actor,lr_critic, lr_lr, lr_lri, lr_decay):
    plot_g_vec_f = np.zeros(shape=(num_exp,num_iterations))
    plot_fi = np.zeros_like(plot_g_vec_f)
    
    #dataframes
    ddpg_g_vec_f_df = pd.DataFrame()
    ddpg_g_vec_f_df['Iteration'] = np.arange(num_iterations)
    ddpg_fi_df = ddpg_g_vec_f_df.copy(deep=True)   
    
    for exp in range(num_exp):
        ddpgpdl = PD_DDPG(env, num_users, pow_max, priority_weights, lr_x, lr_actor,lr_critic, lr_lr, lr_lri, lr_decay)
        ddpgpdl.reset_env()
        for i in range(num_iterations):
            plot_g_vec_f[exp,i], g_x, plot_fi[exp,i] = ddpgpdl.step()
            if i%10000==0: print("Exp ",exp+1," ",g_x)
        print("Saving DataFrames for experiment ",exp+1)      
        ddpg_g_vec_f_df["exp_"+str(exp+1)] = plot_g_vec_f[exp]
        ddpg_fi_df["exp_"+str(exp+1)] = plot_fi[exp]
        
    print("Saving Data to Google Drive Now")    
    ddpg_g_vec_f_df.to_pickle("./data/awgn/ddpg/ddpg_g_vec_f_df.pkl")
    ddpg_fi_df.to_pickle("./data/awgn/ddpg/ddpg_fi_df.pkl")