import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.movmean import movmean

def plotter(dir_path, file_name, num_exp, window, dfactor, dir_list, method_list):
    df = {}
    df_temp = pd.read_pickle(os.path.join(dir_path,'v3/v3_g_x_df.pkl'))
    drop_b = list(df_temp.index)
    drop_a = list(df_temp.index)[::dfactor]
    drop_list = list((Counter(drop_b) - Counter(drop_a)).elements())
    for i, d in enumerate(dir_list[:-1]):
        df[d+'_df'] = pd.read_pickle(os.path.join(dir_path,d+'/'+d+file_name))
        for j in range(num_exp):
            df[d+'_df']['exp_'+str(j+1)] = movmean(df[d+'_df']['exp_'+str(j+1)].to_numpy(), window)
        df[d+'_df'] = df[d+'_df'].drop(drop_list)
        df[d+'_df'] = pd.melt(df[d+'_df'], id_vars='Iteration', var_name='Experiment', value_name='Power Constraint Violation')
        df[d+'_df']['method'] = method_list[i]
    df_pcv = pd.concat(list(df.values()))
    df_pcv

    df = {}
    df_temp = pd.read_pickle(os.path.join(dir_path,'v3/v3_g_x_df.pkl'))
    drop_b = list(df_temp.index)
    drop_a = list(df_temp.index)[::dfactor]
    drop_list = list((Counter(drop_b) - Counter(drop_a)).elements())
    for n in range(10):
        df[str(n+1)+'_df'] = pd.read_pickle(os.path.join(dir_path,'v3/v3_rate_cstr'+str(n+1)+'_df.pkl'))
    for j in range(num_exp):
        df[str(n+1)+'_df']['exp_'+str(j+1)] = movmean(df[str(n+1)+'_df']['exp_'+str(j+1)].to_numpy(), window)
    df[str(n+1)+'_df'] = df[str(n+1)+'_df'].drop(drop_list)
    df[str(n+1)+'_df'] = pd.melt(df[str(n+1)+'_df'], id_vars='Iteration', var_name='Experiment', value_name='Rate Constraint Violation')
    df[str(n+1)+'_df']['number'] = str(n+1)
    df_rcv = pd.concat(list(df.values()))
    df_rcv

    fig, axes = plt.subplots(2,1)
    axes[0].set_xlim([0,100000])
    axes[0].set_ylim([-10,20])
    axes[0].set_title(r"(a) Power constraint violations in all considered methods", pad=3)
    b1 = sns.lineplot(ax = axes[0], x=r'Iteration', y=r'Power Constraint Violation', hue=r'method', data=df_pcv, hue_order=method_list)
    axes[0].legend(loc='upper right')
    axes[0].set_xlabel(None)
    axes[1].set_xlim([0,100000])
    axes[1].set_ylim([-0.1,1])
    axes[1].set_title(r"(b) Rate constraint violations in proposed method (Ergodic)", pad=3)
    b2 = sns.lineplot(ax = axes[1], x=r'Iteration', y=r'Rate Constraint Violation', hue=r'number', data=df_rcv, legend=False)
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(top=0.95, hspace=0.4)
    plt.savefig('pcv_rcv_multi_plot.pdf')