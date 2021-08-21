import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import copy
import pandas as pd
from method.awgn.pdzdpg_awgn import PD_ZDPG

def trainer( env, num_exp, num_iterations, num_users, pow_max, priority_weights, mu_r, lr_x, lr_th, lr_lr, lr_lri, c):
    plot_g_vec_f = np.zeros(shape=(num_exp,num_iterations))
    plot_fi = np.zeros_like(plot_g_vec_f)

    #dataframes
    v2_g_vec_f_df = pd.DataFrame()
    v2_g_vec_f_df['Iteration'] = np.arange(num_iterations)
    v2_fi_df = v2_g_vec_f_df.copy(deep=True)

    for exp in range(num_exp):
        pd_zdpg = PD_ZDPG(env, num_users, pow_max, priority_weights, mu_r, lr_x, lr_th, lr_lr, lr_lri, c)
        pd_zdpg.reset_env()
        for i in range(num_iterations):
            plot_g_vec_f[exp,i], g_x, plot_fi[exp,i] = pd_zdpg.step()
            if i%10000==0: print("Exp ",exp+1," ", g_x)
        print("Saving DataFrame for experiment ",exp+1)      
        v2_g_vec_f_df["exp_"+str(exp+1)] = plot_g_vec_f[exp]
        v2_fi_df["exp_"+str(exp+1)] = plot_fi[exp]
    print("Saving Data to Google Drive Now")    
    v2_g_vec_f_df.to_pickle("./data/awgn/v2/v2_g_vec_f_df.pkl")
    v2_fi_df.to_pickle("./data/awgn/v2/v2_fi_df.pkl")
