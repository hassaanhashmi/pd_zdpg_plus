import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import gym
import copy
from collections import Counter
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from utils.movmean import movmean
from trainer.mai.train_pdzdpg import trainer as trainer_pdzdpg
from trainer.mai.train_pdzdpg_p import trainer as trainer_pdzdpg_p
from trainer.mai.train_wmmse import trainer as trainer_wmmse
from plotter.mai_plotter import plotter

#config
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
sns.set_theme(context="paper", style="whitegrid")
sns.set_style({'font.family':'Times New Roman'})
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
    "legend.title_fontsize":8,
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
channel_mu=2,
noise_var=1
pow_max = 20
num_exp = 5
wmmse_iter = 200

dir_path = './data/mai/'
file_name = '_g_vec_f_df.pkl'
num_exp = 5
window = 1000
dfactor = [150, 300, 400]
dir_list = ['v2','v3', 'wmmse']
method_list = ['Random: PD-ZDPG [2]', 'Ergodic: PD-ZPDG [2]', 'Ergodic: PD-ZDPG+ (Proposed)',\
               'Ergodic: WMMSE [4]', 'Objective $\mathbf{w}^{T} \mathbf{x}$: PD-ZDPG+ (Proposed)']
list_num_users = [10, 25, 50]


if __name__ == "__main__":
    #create folder and subfolders
    if not os.path.exists(dir_path):
        for dir in dir_list:
            for user in list_num_users:
                path = os.path.join(dir_path, dir+str(user))
                os.makedirs(path)
    #10 users
    iterations = 300001
    priority_weights = np.array([[0.195908404155517,
                                0.098682331155082,
                                0.010756919947350,
                                0.038432374846126,
                                0.001222761059510,
                                0.193806914053259,
                                0.068729218419940,
                                0.037385433953708,
                                0.189930555148395,
                                0.165145087261113]]).T
    num_users = priority_weights.shape[0]
    env = gym.make('gym_cstr_optim:mai-v0', 
                num_users=num_users,
                priority_weights=priority_weights,
                pow_max=pow_max,
                channel_mu=channel_mu,
                noise_var=noise_var)
    
    trainer_pdzdpg(env=env,
                num_exp = num_exp,
                num_iterations=iterations,
                num_users=num_users,
                pow_max=pow_max,
                priority_weights=priority_weights,
                mu_r=1e-9,
                lr_x=0.001,
                lr_th=0.00005,
                lr_lr=0.004,
                lr_lri=0.0001,
                c=0)
    
    trainer_pdzdpg_p(env=env,
                num_exp=num_exp,
                num_iterations=iterations,
                num_users=num_users,
                pow_max=pow_max,
                priority_weights=priority_weights,
                mu_r=1e-9,
                lr_x=0.001,
                lr_th=0.04,
                lr_lr=0.008,
                lr_lri=0.0001,
                c=0)
    
    trainer_wmmse(env=env,
                num_exp=num_exp,
                num_iterations=iterations,
                num_users=num_users,
                pow_max=pow_max,
                priority_weights=priority_weights,
                noise_var=noise_var,
                wmmse_iter=wmmse_iter)


    #25 users
    iterations = 600001
    priority_weights = np.array([[0.06207456, 0.06134886, 0.0261264 , 0.02364688, 0.04468602,
                              0.06308199, 0.05463633, 0.07351197, 0.01660868, 0.07697788,
                              0.02463234, 0.04993269, 0.06615327, 0.03538398, 0.00264264,
                              0.04450962, 0.05283343, 0.03816324, 0.01560540, 0.00606673,
                              0.07713794, 0.00324537, 0.02104770, 0.03826059, 0.02168548]]).T
    num_users = priority_weights.shape[0]
    env = gym.make('gym_cstr_optim:mai-v0', 
                num_users=num_users,
                priority_weights=priority_weights,
                pow_max=pow_max,
                channel_mu=channel_mu,
                noise_var=noise_var)

    trainer_pdzdpg(env=env,
                num_exp = num_exp,
                num_iterations=iterations,
                num_users=num_users,
                pow_max=pow_max,
                priority_weights=priority_weights,
                mu_r=1e-9,
                lr_x=0.001,
                lr_th=0.00005,
                lr_lr=0.004,
                lr_lri=0.0001,
                c=0)
    
    trainer_pdzdpg_p(env=env,
                num_exp=num_exp,
                num_iterations=iterations,
                num_users=num_users,
                pow_max=pow_max,
                priority_weights=priority_weights,
                mu_r=1e-9,
                lr_x=0.001,
                lr_th=0.04,
                lr_lr=0.008,
                lr_lri=0.0001,
                c=0)
    
    trainer_wmmse(env=env,
                num_exp=num_exp,
                num_iterations=iterations,
                num_users=num_users,
                pow_max=pow_max,
                priority_weights=priority_weights,
                noise_var=noise_var,
                wmmse_iter=wmmse_iter)


    #50 users
    iterations = 800001
    priority_weights = np.array([[0.0226765277970221300, 0.0171004333173375050, 0.0084004796566064870, 0.0297334186274717900,
                                0.0153726300754585070, 0.0003572417397355403, 0.0176968648889887680, 0.0110649364077487100,
                                0.0258826648714543000, 0.0328449694379896200, 0.0250796701467405520, 0.0296135583374454660,
                                0.0354395380659633250, 0.0235400570355920150, 0.0279524147282580080, 0.0173642323911879800,
                                0.0036001068052683120, 0.0201906014505511060, 0.0092332743954706330, 0.0123008827960149340,
                                0.0325039419430143200, 0.0125746730291091470, 0.0211725950026849170, 0.0080294980518835110,
                                0.0266191647690334800, 0.0128833648660391760, 0.0275025237506026600, 0.0371808809581958200,
                                0.0118740052822522720, 0.0211829196716128640, 0.0175521481412721850, 0.0083260321087246220,
                                0.0342344560650417900, 0.0335927899309193260, 0.0337436114652624200, 0.0121924776692405730,
                                0.0180844820336568120, 0.0260724781927306230, 0.0025311952937090260, 0.0194336023352554220,
                                0.0335753721387821300, 0.0082480679403871930, 0.0158653604903985200, 0.0316936346745223700,
                                0.0366861356942236500, 0.0005202801894925929, 0.0177391919984798730, 0.0153038587891118350,
                                0.0289476164328770980, 0.0086891381191781070]]).T
    num_users = priority_weights.shape[0]
    env = gym.make('gym_cstr_optim:mai-v0', 
                num_users=num_users,
                priority_weights=priority_weights,
                pow_max=pow_max,
                channel_mu=channel_mu,
                noise_var=noise_var)
    
    trainer_pdzdpg(env=env,
                num_exp = num_exp,
                num_iterations=iterations,
                num_users=num_users,
                pow_max=pow_max,
                priority_weights=priority_weights,
                mu_r=1e-9,
                lr_x=0.001,
                lr_th=0.00005,
                lr_lr=0.004,
                lr_lri=0.0001,
                c=0)
    
    trainer_pdzdpg_p(env=env,
                num_exp=num_exp,
                num_iterations=iterations,
                num_users=num_users,
                pow_max=pow_max,
                priority_weights=priority_weights,
                mu_r=1e-9,
                lr_x=0.001,
                lr_th=0.04,
                lr_lr=0.008,
                lr_lri=0.0001,
                c=0)
    
    trainer_wmmse(env=env,
                num_exp=num_exp,
                num_iterations=iterations,
                num_users=num_users,
                pow_max=pow_max,
                priority_weights=priority_weights,
                noise_var=noise_var,
                wmmse_iter=wmmse_iter)
    
    #plotter
    plotter(dir_path, file_name, num_exp, window, dfactor, dir_list, method_list, list_num_users)