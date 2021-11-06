import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
import random
import gym
import copy
import pandas as pd
from method.mai.pdzdpg_p_mai import PD_ZDPG_Plus



def trainer( env, num_exp, num_iterations, num_users, pow_max, priority_weights, mu_r, lr_x, lr_th, lr_lr, lr_lri, c):
    plot_g_vec_f = np.zeros(shape=(num_exp,num_iterations))

    #dataframes
    v3_g_vec_f_df = pd.DataFrame()
    v3_g_vec_f_df['Iteration'] = np.arange(num_iterations)
    
    for exp in range(num_exp):
        pd_zdpg_p = PD_ZDPG_Plus(env, num_users, pow_max, priority_weights, mu_r, lr_x, lr_th, lr_lr, lr_lri, c)
        pd_zdpg_p.reset_env()
        for i in range(num_iterations):
            plot_g_vec_f[exp,i], g_x = pd_zdpg_p.step()
            if i%10000==0: print("Exp ",exp+1," ",g_x)
        print("Saving DataFrames for experiment ",exp+1)      
        v3_g_vec_f_df["exp_"+str(exp+1)] = plot_g_vec_f[exp]
    
        print("Saving Data to Google Drive Now")    
        v3_g_vec_f_df.to_pickle("./data/mai/v3"+str(num_users)+"/v3_g_vec_f_df.pkl")