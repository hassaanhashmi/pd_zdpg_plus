import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
import gym
import copy
import pandas as pd
from method.awgn.pdzdpg_p_awgn import PD_ZDPG_Plus

def trainer( env, num_exp, num_iterations, num_users, pow_max, priority_weights, mu_r, lr_x, lr_th, lr_lr, lr_lri, c):
    plot_g_vec_f = np.zeros(shape=(num_exp,num_iterations))
    plot_g_x = np.zeros_like(plot_g_vec_f)
    plot_fi = np.zeros_like(plot_g_vec_f)
    plot_rate_cstr = np.zeros(shape=(num_exp,num_users, num_iterations))

    #dataframes
    v3_g_vec_f_df = pd.DataFrame()
    v3_g_vec_f_df['Iteration'] = np.arange(num_iterations)
    v3_g_x_df = v3_g_vec_f_df.copy(deep=True)
    v3_fi_df = v3_g_vec_f_df.copy(deep=True)
    v3_rate_cstr_df = []
    for i in range(num_users):
        v3_rate_cstr_df.append(v3_g_vec_f_df.copy(deep=True))
        
    for exp in range(num_exp):
        pd_zdpg_p = PD_ZDPG_Plus(env, num_users, pow_max, priority_weights, mu_r, lr_x, lr_th, lr_lr, lr_lri, c)
        pd_zdpg_p.reset_env()
        for i in range(num_iterations):
            o1, o2, o3, o4 = pd_zdpg_p.step()
            plot_g_vec_f[exp,i] = o1
            plot_g_x[exp,i] = o2
            plot_fi[exp,i] = o3
            plot_rate_cstr[exp,:,i] = o4
            if i%10000==0: print("Exp ",exp+1," ",plot_g_x[exp, i])
        print("Saving DataFrames for experiment ",exp+1)      
        v3_g_vec_f_df["exp_"+str(exp+1)] = plot_g_vec_f[exp]
        v3_g_x_df["exp_"+str(exp+1)] = plot_g_x[exp]
        v3_fi_df["exp_"+str(exp+1)] = plot_fi[exp]
        for n in range(num_users):
            v3_rate_cstr_df[n]["exp_"+str(exp+1)] = plot_rate_cstr[exp,n]
    
    print("Saving Data to Google Drive Now")    
    v3_g_vec_f_df.to_pickle("./data/awgn/v3/v3_g_vec_f_df.pkl")
    v3_g_x_df.to_pickle("./data/awgn/v3/v3_g_x_df.pkl")
    v3_fi_df.to_pickle("./data/awgn/v3/v3_fi_df.pkl")
    for n in range(num_users):
        v3_rate_cstr_df[n].to_pickle("./data/awgn/v3/v3_rate_cstr"+str(n+1)+"_df.pkl")