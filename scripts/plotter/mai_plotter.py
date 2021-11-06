import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.movmean import movmean

def plotter(dir_path, file_name, num_exp, window, dfactor, dir_list, method_list, list_num_users):
    #10 users
    df = {}
    df['v3_gx'+str(list_num_users[0])+'_df'] = pd.read_pickle(os.path.join(dir_path,'v3'+str(list_num_users[0])+'/v3_g_x_df.pkl'))
    drop_b = list(df['v3_gx'+str(list_num_users[0])+'_df'].index)
    drop_a = list(df['v3_gx'+str(list_num_users[0])+'_df'].index)[::dfactor[0]]
    drop_list = list((Counter(drop_b) - Counter(drop_a)).elements())
    df['v3_gx'+str(list_num_users[0])+'_df'] = df['v3_gx'+str(list_num_users[0])+'_df'].drop(drop_list)
    df['v3_gx'+str(list_num_users[0])+'_df'] = pd.melt(df['v3_gx'+str(list_num_users[0])+'_df'], id_vars='Iteration', var_name='Experiment', value_name='Sumrate')
    df['v3_gx'+str(list_num_users[0])+'_df']['method'] = method_list[-1]
    for i, d in enumerate(dir_list):
        df[d+str(list_num_users[0])+'_df'] = pd.read_pickle(os.path.join(dir_path,d+str(list_num_users[0])+'/'+d+file_name))
        for j in range(num_exp):
            df[d+str(list_num_users[0])+'_df']['exp_'+str(j+1)] = movmean(df[d+str(list_num_users[0])+'_df']['exp_'+str(j+1)].to_numpy(), window)
    df[d+str(list_num_users[0])+'_df'] = df[d+str(list_num_users[0])+'_df'].drop(drop_list)
    df[d+str(list_num_users[0])+'_df'] = pd.melt(df[d+str(list_num_users[0])+'_df'], id_vars='Iteration', var_name='Experiment', value_name='Sumrate')
    df[d+str(list_num_users[0])+'_df']['method'] = method_list[i+1]
    df_final_10 = pd.concat(list(df.values()))
    #25 users
    df = {}
    df['v3_gx'+str(list_num_users[1])+'_df'] = pd.read_pickle(os.path.join(dir_path,'v3'+str(list_num_users[1])+'/v3_g_x_df.pkl'))
    df['v2_gf'+str(list_num_users[1])+'_df'] = pd.read_pickle(os.path.join(dir_path,'v2'+str(list_num_users[1])+'/v2_g_vec_f_df.pkl'))
    drop_b = list(df['v3_gx'+str(list_num_users[1])+'_df'].index)
    drop_a = list(df['v3_gx'+str(list_num_users[1])+'_df'].index)[::dfactor[1]]
    drop_list = list((Counter(drop_b) - Counter(drop_a)).elements())
    df['v3_gx'+str(list_num_users[1])+'_df'] = df['v3_gx'+str(list_num_users[1])+'_df'].drop(drop_list)
    df['v2_gf'+str(list_num_users[1])+'_df'] = df['v2_gf'+str(list_num_users[1])+'_df'].drop(drop_list)
    df['v3_gx'+str(list_num_users[1])+'_df'] = pd.melt(df['v3_gx'+str(list_num_users[1])+'_df'], id_vars='Iteration', var_name='Experiment', value_name='Sumrate')
    df['v2_gf'+str(list_num_users[1])+'_df'] = pd.melt(df['v2_gf'+str(list_num_users[1])+'_df'], id_vars='Iteration', var_name='Experiment', value_name='Sumrate')
    df['v3_gx'+str(list_num_users[1])+'_df']['method'] = method_list[-1]
    df['v2_gf'+str(list_num_users[1])+'_df']['method'] = method_list[0]
    for i, d in enumerate(dir_list):
        df[d+str(list_num_users[1])+'_df'] = pd.read_pickle(os.path.join(dir_path,d+str(list_num_users[1])+'/'+d+file_name))
        for j in range(num_exp):
            df[d+str(list_num_users[1])+'_df']['exp_'+str(j+1)] = movmean(df[d+str(list_num_users[1])+'_df']['exp_'+str(j+1)].to_numpy(), window)
        df[d+str(list_num_users[1])+'_df'] = df[d+str(list_num_users[1])+'_df'].drop(drop_list)
        df[d+str(list_num_users[1])+'_df'] = pd.melt(df[d+str(list_num_users[1])+'_df'], id_vars='Iteration', var_name='Experiment', value_name='Sumrate')
        df[d+str(list_num_users[1])+'_df']['method'] = method_list[i+1]
    df_final_25 = pd.concat(list(df.values()))
    #50 users
    df = {}
    df['v3_gx'+str(list_num_users[2])+'_df'] = pd.read_pickle(os.path.join(dir_path,'v3'+str(list_num_users[2])+'/v3_g_x_df.pkl'))
    df['v2_gf'+str(list_num_users[2])+'_df'] = pd.read_pickle(os.path.join(dir_path,'v2'+str(list_num_users[2])+'/v2_g_vec_f_df.pkl'))
    drop_b = list(df['v3_gx'+str(list_num_users[2])+'_df'].index)
    drop_a = list(df['v3_gx'+str(list_num_users[2])+'_df'].index)[::dfactor[2]]
    drop_list = list((Counter(drop_b) - Counter(drop_a)).elements())
    df['v3_gx'+str(list_num_users[2])+'_df'] = df['v3_gx'+str(list_num_users[2])+'_df'].drop(drop_list)
    df['v2_gf'+str(list_num_users[2])+'_df'] = df['v2_gf'+str(list_num_users[2])+'_df'].drop(drop_list)
    df['v3_gx'+str(list_num_users[2])+'_df'] = pd.melt(df['v3_gx'+str(list_num_users[2])+'_df'], id_vars='Iteration', var_name='Experiment', value_name='Sumrate')
    df['v2_gf'+str(list_num_users[2])+'_df'] = pd.melt(df['v2_gf'+str(list_num_users[2])+'_df'], id_vars='Iteration', var_name='Experiment', value_name='Sumrate')
    df['v3_gx'+str(list_num_users[2])+'_df']['method'] = method_list[-1]
    df['v2_gf'+str(list_num_users[2])+'_df']['method'] = method_list[0]
    for i, d in enumerate(dir_list):
        df[d+str(list_num_users[2])+'_df'] = pd.read_pickle(os.path.join(dir_path,d+str(list_num_users[2])+'/'+d+file_name))
        for j in range(num_exp):
            df[d+str(list_num_users[2])+'_df']['exp_'+str(j+1)] = movmean(df[d+str(list_num_users[2])+'_df']['exp_'+str(j+1)].to_numpy(), window)
        df[d+str(list_num_users[2])+'_df'] = df[d+str(list_num_users[2])+'_df'].drop(drop_list)
        df[d+str(list_num_users[2])+'_df'] = pd.melt(df[d+str(list_num_users[2])+'_df'], id_vars='Iteration', var_name='Experiment', value_name='Sumrate')
        df[d+str(list_num_users[2])+'_df']['method'] = method_list[i+1]
    df_final_50 = pd.concat(list(df.values()))

    #plot
    fig, axes = plt.subplots(3,1)
    sns.set_palette(['#dd8452', '#55a868', '#8172b3', '#937860'])
    axes[0].set_xlim([-1000,300000])
    axes[0].set_ylim([0,1])
    axes[0].set_title(r"(a) 10 users", pad=3)
    b1 = sns.lineplot(ax = axes[0], x=r'Iteration', y=r'Sumrate', hue=r'method', data=df_final_10, hue_order=method_list[1:])
    axes[0].legend(loc='lower right')
    axes[0].set_ylabel(None)
    axes[0].set_xlabel(None)
    sns.set_palette(['#762e40', '#dd8452', '#55a868', '#8172b3', '#937860'])
    axes[1].set_xlim([-2000,600000])
    axes[1].set_ylim([0,0.3])
    axes[1].set_title(r"(b) 25 users", pad=3)
    b2 = sns.lineplot(ax = axes[1], x=r'Iteration', y=r'Sumrate', hue=r'method', data=df_final_25, hue_order=method_list, legend=False)
    plt.setp(b2.lines[0], alpha=0.1)
    axes[1].set_ylabel(r'Sumrate', labelpad=7)
    axes[1].set_xlabel(None)
    sns.set_palette(['#762e40', '#dd8452', '#55a868', '#8172b3', '#937860'])
    axes[2].set_xlim([-2500,800000])
    axes[2].set_ylim([0,0.15])
    axes[2].set_title(r"(c) 50 users", pad=3)
    b3 = sns.lineplot(ax = axes[2], x=r'Iteration', y=r'Sumrate', hue=r'method', data=df_final_50, hue_order=method_list, legend=False)
    plt.setp(b3.lines[0], alpha=0.1)
    axes[2].set_ylabel(None)
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(top=0.95, hspace=0.4)
    plt.savefig('mai_multi_plot.pdf')