import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import copy
import pandas as pd
from method.mai.pdzdpg_mai import PD_ZDPG

def trainer( env, num_exp, num_iterations, num_users, pow_max, priority_weights, mu_r, lr_x, lr_th, lr_lr, lr_lri, c):
    plot_g_vec_f = np.zeros(shape=(num_exp,num_iterations))

    #dataframes
    v2_g_vec_f_df = pd.DataFrame()
    v2_g_vec_f_df['Iteration'] = np.arange(num_iterations)

    for exp in range(num_exp):
        pdzdpg = PD_ZDPG(env, num_users, pow_max, priority_weights, mu_r, lr_x, lr_th, lr_lr, lr_lri, c)
        pdzdpg.reset_env()
        for i in range(num_iterations):
            plot_g_vec_f[exp,i], g_x = pdzdpg.step()
            if i%10000==0: print("Exp ",exp+1," ",g_x)
        print("Saving DataFrames for experiment ",exp+1)      
        v2_g_vec_f_df["exp_"+str(exp+1)] = plot_g_vec_f[exp]
    
    print("Saving Data to Google Drive Now")    
    v2_g_vec_f_df.to_pickle("./data/mai/v2"+str(num_users)+"/v2_g_vec_f_df.pkl")
