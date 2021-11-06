import os
import numpy as np
import copy
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plotter(dir_path):
    df = {}
    df_v2 = pd.read_pickle(dir+"2v2_iter_time_df.pkl")
    df_v2['method'] = 'PD-ZDPG [2]'
    df_v3 = pd.read_pickle(dir+"2v3_iter_time_df.pkl")
    df_v3['method'] = 'PD-ZDPG+ (Proposed)'
    df_final = pd.concat([df_v2, df_v3])
    df_final

    fig, ax = plt.subplots()
    plot = sns.lineplot(ax=ax, x="Neural Network sizes", y="Best execution time per iteration (ms)", palette=['#dd8452', '#55a868'], hue='method', data=df_final, marker='o')
    plot.set_xticklabels(df_final["Neural Network sizes"],rotation=30)
    plt.legend(loc='upper left')
    # label points on the plot
    for x, y_v2, y_v3 in zip(df_v2["Neural Network sizes"], df_v2["Best execution time per iteration (ms)"], df_v3["Best execution time per iteration (ms)"]):
        i = df_v2[df_v2["Neural Network sizes"]== x].index.values
        print(i)
        ax.text(x = i-0.4, y = y_v2+10, s = '{:.2f}'.format(y_v2), fontsize=5.5)
        ax.text(x = x, y = y_v3-15, s = '{:.2f}'.format(y_v3), fontsize=5.5)
    plt.ylabel(r'Execution time (ms)')
    plt.xlabel(r'Neural Network size')
    plt.tight_layout(pad=0.5)
    plt.savefig('scale.pdf')