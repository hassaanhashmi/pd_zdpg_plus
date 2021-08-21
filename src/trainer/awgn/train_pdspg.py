import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.distributions import Normal
import numpy as np
import random
import gym
import copy
import pandas as pd
from method.awgn.pdspg_awgn import PD_SPG

#CORRECT THIS SCRIPT
def trainer( env, num_exp, num_iterations, num_users, pow_max, priority_weights, lr_x, lr_th, lr_lr, lr_lri, batch_size):
    plot_g_vec_f = np.zeros(shape=(num_exp,num_iterations))
    plot_fi = np.zeros_like(plot_g_vec_f)
    #dataframes
    v1_g_vec_f_df = pd.DataFrame()
    v1_g_vec_f_df['Iteration'] = np.arange(num_iterations)
    v1_fi_df = v1_g_vec_f_df.copy(deep=True)

    for exp in range(num_exp):
        pd_spg = PD_SPG(env, num_users, pow_max, priority_weights, lr_x, lr_th, lr_lr, lr_lri, batch_size)
        pd_spg.reset_env()
        for i in range(num_iterations):
            plot_g_vec_f[exp,i], g_x, plot_fi[exp,i] = pd_spg.step()
            if i%10000==0: print("Exp ",exp+1," ",g_x)
        print("Saving DataFrames for experiment ",exp+1)      
        v1_g_vec_f_df["exp_"+str(exp+1)] = plot_g_vec_f[exp]
        v1_fi_df["exp_"+str(exp+1)] = plot_fi[exp]
    print("Saving Data to Google Drive Now")    
    v1_g_vec_f_df.to_pickle("./data/awgn/v1/v1_g_vec_f_df.pkl")
    v1_fi_df.to_pickle("./data/awgn/v1/v1_fi_df.pkl")