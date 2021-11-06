import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.movmean import movmean

def plotter(dir_path, file_name, num_exp, window, dfactor, dir_list, method_list):
    df = {}
    df['v3gx_df'] = pd.read_pickle(os.path.join(dir_path,'v3/v3_g_x_df.pkl'))
    drop_b = list(df['v3gx_df'].index)
    drop_a = list(df['v3gx_df'].index)[::dfactor]
    drop_list = list((Counter(drop_b) - Counter(drop_a)).elements())
    df['v3gx_df'] = df['v3gx_df'].drop(drop_list)
    df['v3gx_df'] = pd.melt(df['v3gx_df'], id_vars='Iteration', var_name='Experiment', value_name='Sumrate')
    df['v3gx_df']['method'] = method_list[-1]
    for i, d in enumerate(dir_list):
      df[d+'_df'] = pd.read_pickle(os.path.join(dir_path,d+'/'+d+file_name))
      for j in range(num_exp):
        df[d+'_df']['exp_'+str(j+1)] = movmean(df[d+'_df']['exp_'+str(j+1)].to_numpy(), window)
      df[d+'_df'] = df[d+'_df'].drop(drop_list)
      df[d+'_df'] = pd.melt(df[d+'_df'], id_vars='Iteration', var_name='Experiment', value_name='Sumrate')
      df[d+'_df']['method'] = method_list[i]
    df_final = pd.concat(list(df.values()))

    plt.ylim(0,3)
    plt.xlim(0, 100000)
    b = sns.lineplot(x=r'Iteration', y=r'Sumrate', hue=r'method', data=df_final, hue_order=method_list)
    plt.xlabel(r'Iteration')
    plt.ylabel(r'Sumrate')
    plt.legend(loc='lower right')
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(top=0.95)
    plt.savefig('awgn_multi_plot.pdf')
