
import numpy as np
import gym
import pandas as pd
from method.awgn.clairvoyant_awgn import ClairvoyantPolicy

def trainer(env, num_exp, num_iterations, num_users, pow_max, priority_weights, noise_var, lr_lu):
    plot_sumrate = np.zeros(shape=(num_exp, num_iterations))
    clair_sumrate_df = pd.DataFrame()
    clair_sumrate_df["Iteration"] = np.arange(num_iterations)
    for exp in range(num_exp):
        clair = ClairvoyantPolicy(env, num_users, pow_max, priority_weights, noise_var, lr_lu)
        clair.reset_env()
        for i in range(num_iterations):
            plot_sumrate[exp,i] = clair.step()
            if i%10000==0:print("Exp ",exp+1," ",plot_sumrate[exp, i])
        clair_sumrate_df["exp_"+str(exp)] = plot_sumrate[exp]
    print("Saving Data Now")    
    clair_sumrate_df.to_pickle("./data/awgn/clair/clair_sumrate_df.pkl")