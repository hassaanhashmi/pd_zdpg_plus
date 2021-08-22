import os
import sys
import time
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
import random
import gym
import copy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from method.mai.pdzdpg_mai_scale import PD_ZDPG
from method.mai.pdzdpg_p_mai_scale import PD_ZDPG_Plus
from plotter.scale_plotter import plotter

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
sns.set_theme(context="paper", style="whitegrid")
sns.set_context("paper", rc={"lines.line_width":1})
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family":"Times New Roman",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.titlesize": 7,
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 4,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    # reduce padding of x/y label ticks with plots
    "xtick.major.pad":0,
    "ytick.major.pad":0,
    #set figure size and dpi
    'figure.figsize': (4.875, 2),
    'figure.dpi': 600
}
plt.rcParams.update(tex_fonts)

#arguments
channel_mu=2,
noise_var=1
pow_max = 20
num_exp = 11
repeat = 10
iterations = 1000
priority_weights = np.array([[0.231508653774241,
                              0.174428832659975,
                              0.204576467124592,
                              0.154505309166867,
                              0.234980737274325]]).T
num_users = priority_weights.shape[0]
env = gym.make('gym_cstr_optim:mai-v0', 
                num_users=num_users,
                priority_weights=priority_weights,
                pow_max=pow_max,
                channel_mu=channel_mu,
                noise_var=noise_var)
dir_path = "./data/scale/"



def trainer_pd_zdpg(env, num_exp, repeat, num_iterations, num_users, pow_max, priority_weights, mu_r, lr_x, lr_th, lr_lr, lr_lri, c):
    v2_iter_time_df = pd.DataFrame()
    v2_iter_time_df['Best execution time per iteration (ms)'] = np.zeros(num_exp)
    v2_iter_time_df['Neural Network sizes'] = 'empty'
    for exp in range(num_exp):
        nn_scale = exp*10
        if nn_scale <=0: nn_scale =1
        v2_iter_time_df['Neural Network sizes'][exp] = '[5,'+str(32*nn_scale)+', '+str(16*nn_scale)+', 5]'  
        pdzdpg = PD_ZDPG(env, num_users, pow_max, priority_weights, mu_r, lr_x, lr_th, lr_lr, lr_lri, c, nn_scale)
        pdzdpg.reset_env()
        avg_time = 1e3*min(timeit.repeat(stmt=pdzdpg.step, repeat=repeat, number=num_iterations))/num_iterations
        v2_iter_time_df["Best execution time per iteration (ms)"][exp] = avg_time
        print('[5, '+str(32*nn_scale)+', '+str(16*nn_scale)+', 5]: ',avg_time,"ms")
        v2_iter_time_df.to_pickle(dir+"v2_iter_time_df.pkl")

def trainer_pd_zdpg_p(env, num_exp, repeat, num_iterations, num_users, pow_max, priority_weights, mu_r, lr_x, lr_th, lr_lr, lr_lri, c):
    v3_iter_time_df = pd.DataFrame()
    v3_iter_time_df['Best execution time per iteration (ms)'] = np.zeros(num_exp)
    v3_iter_time_df['Neural Network sizes'] = 'empty'
    for exp in range(num_exp):
        nn_scale = exp*10
        if nn_scale <=0: nn_scale =1
        v3_iter_time_df['Neural Network sizes'][exp] = '[5,'+str(32*nn_scale)+', '+str(16*nn_scale)+', 5]'
        pdzdpg_p = PD_ZDPG_Plus(env, num_users, pow_max, priority_weights, mu_r, lr_x, lr_th, lr_lr, lr_lri, c, nn_scale)
        pdzdpg_p.reset_env()
        avg_time = 1e3*min(timeit.repeat(stmt=pdzdpg_p.step, repeat=repeat, number=num_iterations))/num_iterations
        v3_iter_time_df["Best execution time per iteration (ms)"][exp] = avg_time
        print('[5, '+str(32*nn_scale)+', '+str(16*nn_scale)+', 5]: ',avg_time,"ms")
        print("Saving Data Now")    
        v3_iter_time_df.to_pickle(dir+"v3_iter_time_df.pkl")

if __name__ == "__main__":

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    trainer_pd_zdpg(env=env,
        num_exp = num_exp,
        repeat=repeat,
        num_iterations=iterations,
        num_users=num_users,
        pow_max=pow_max,
        priority_weights=priority_weights,
        mu_r=1e-9,
        lr_x=0.0006,
        lr_th=0.0005,
        lr_lr=0.001,
        lr_lri=0.0001,
        c=0)

    trainer_pd_zdpg_p(env=env,
            num_exp = num_exp,
            repeat=repeat,
            num_iterations=iterations,
            num_users=num_users,
            pow_max=pow_max,
            priority_weights=priority_weights,
            mu_r=1e-9,
            lr_x=0.0006,
            lr_th=0.0005,
            lr_lr=0.001,
            lr_lri=0.0001,
            c=0)

    plotter(dir_path=dir_path)