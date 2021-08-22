import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import SGD
import gym
import copy
from collections import Counter
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from utils.movmean import movmean
from trainer.awgn.train_pdspg import trainer as trainer_pdspg
from trainer.awgn.train_pdzdpg import trainer as trainer_pdzdpg
from trainer.awgn.train_pdzdpg_p import trainer as trainer_pdzdpg_p
from trainer.awgn.train_pdddpg import trainer as trainer_pdddpg
from trainer.awgn.train_clairvoyant import trainer as trainer_clairvoyant
from plotter.awgn_plotter_1 import plotter as plotter_1
from plotter.awgn_plotter_2 import plotter as plotter_2

#config
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
    'figure.figsize': (4.875, 3.69),
    'figure.dpi': 600
}
plt.rcParams.update(tex_fonts)

#arguments
batch_size = 32
pow_max = 20
channel_mu=2,
noise_var=1
num_exp = 5
iterations = 100001
priority_weights = np.array([[0.195908404155517, 0.098682331155082, 0.010756919947350,
                              0.038432374846126, 0.001222761059510, 0.193806914053259,
                              0.068729218419940, 0.037385433953708, 0.189930555148395,
                              0.165145087261113]]).T
num_users = priority_weights.shape[0]

dir_path = './awgn/'
file_name = '_g_vec_f_df.pkl'
num_exp = 5
window = 1000
dfactor = 55
dir_list = ['v1', 'v2','v3','ddpg', 'clair']
method_list = ['Ergodic: PD-SPG [9]', 'Ergodic: PD-ZDPG [2]', 'Ergodic: PD-ZDPG+ (Proposed)',\
                'Ergodic: PD-DDPG [12, 15]', 'Ergodic: Clairvoyant [17]',\
                'Objective $\mathbf{w}^{T}\mathbf{x}$: PD-ZDPG+ (Proposed)']

#environment
env = gym.make('gym_cstr_optim:awgn-v0', 
                    num_users=num_users,
                    priority_weights=priority_weights,
                    pow_max=pow_max,
                    channel_mu=channel_mu,
                    noise_var=noise_var)

if __name__ == "__main__":
    #creat folder and subfolders
    if not os.path.exists(dir_path):
        for dir in dir_list:
                path = os.path.join(dir_path, dir)
                os.makedirs(path)

    trainer_pdspg(env=env,
                num_exp = num_exp,
                num_iterations=iterations,
                num_users=num_users,
                batch_size=batch_size,
                pow_max=pow_max,
                priority_weights=priority_weights,
                lr_x=0.01,
                lr_th=0.01,
                lr_lr=0.08,
                lr_lri=0.0001)
    
    trainer_pdzdpg(env=env,
                num_exp=num_exp,
                num_iterations=iterations,
                num_users=num_users,
                pow_max=pow_max,
                priority_weights=priority_weights,
                mu_r=1e-9,
                lr_x=0.001,
                lr_th=0.0008,
                lr_lr=0.008,
                lr_lri=0.0001,
                c=0)
    
    trainer_pdzdpg_p(env=env,
                num_exp=num_exp,
                num_iterations=100000,
                num_users=num_users,
                mu_r=1e-9,
                pow_max=pow_max,
                priority_weights=priority_weights,
                lr_x=0.001,
                lr_th=0.02,
                lr_lr=0.008,
                lr_lri=0.0001,
                c=0)
    
    trainer_pdddpg(env=env,
                num_exp=num_exp,
                num_iterations=iterations,
                num_users=num_users,
                pow_max=pow_max,
                priority_weights=priority_weights,
                lr_x=0.001,
                lr_actor=0.002,
                lr_critic=0.001,
                lr_lr=0.01,
                lr_lri=0.0001,
                lr_decay=1)
    
    trainer_clairvoyant(env=env,
                num_exp=num_exp,
                num_iterations=iterations,
                num_users=num_users,
                pow_max=pow_max,
                priority_weights=priority_weights,
                noise_var=noise_var,
                lr_lu=0.0005)

    plotter_1(dir_path, file_name, num_exp, 
            window, dfactor, dir_list, method_list)
    
    plotter_2(dir_path, file_name, num_exp, 
            window, dfactor, dir_list, method_list)