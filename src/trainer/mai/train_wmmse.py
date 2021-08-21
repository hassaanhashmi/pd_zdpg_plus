import os
import numpy as np
import gym
import pandas as pd
from method.mai.wmmse_mai import WMMSE_Policy

def trainer(env, num_exp, num_iterations, num_users, pow_max, priority_weights, noise_var, wmmse_iter):
    plot_sumrate = np.zeros(shape=(num_exp, num_iterations))

    wmmse_sumrate_df = pd.DataFrame()
    wmmse_sumrate_df["Iteration"] = np.arange(num_iterations)

    #algorithm    
    for exp in range(num_exp):
      wmmse = WMMSE_Policy(env, num_users, pow_max, priority_weights, noise_var, wmmse_iter)
      wmmse.reset_env()
      for i in range(num_iterations):
          plot_sumrate[exp,i] = wmmse.step()
          if i%10000==0:print("Exp ",exp+1," ",plot_sumrate[exp, i])
      wmmse_sumrate_df["exp_"+str(exp+1)] = plot_sumrate[exp]
      
    print("Saving Data Now")    
    wmmse_sumrate_df.to_pickle("./data/mai/wmmse"+str(num_users)+"/wmmse_sumrate_df.pkl")