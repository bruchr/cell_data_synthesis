import os

from matplotlib import pyplot as plt
import numpy as np


class CTC_Det_Score():
    def __init__(self, res, n_fp, n_fn, n_s_ops, name, color) -> None:
        self.res = res
        self.n_fp = n_fp
        self.n_fn = n_fn
        self.n_s_ops = n_s_ops
        self.name = name
        self.color = color


def plot_res(score_list:list, save_path:str=None):
    fig, ax = plt.subplots(ncols=2, figsize=(5.5,  3), gridspec_kw={'width_ratios': [1, 3]})
    x = [1, 2, 3]
    get_x_loc = lambda x, ind_s: np.add(x, np.subtract(ind_s, (len(score_list)-1)/2) /16)
    for ind_s, score in enumerate(score_list):
        ax[0].plot(get_x_loc(0,ind_s), score.res, '.', color=score.color, label=score.name, ms=8)

        y = [score.n_fp, score.n_fn, score.n_s_ops]
        ax[1].plot(get_x_loc(x,ind_s), y, '.', color=score.color, label=score.name, ms=8)

    ax[0].set_xticks([0])
    ax[0].set_xticklabels(['DET Score'])
    ax[0].set_ylabel('%')
    ax[0].set_ylim([0.7, 1])
    ax[0].set_xlim([-0.25, 0.25])
    

    ax[1].set_xticks(x)
    ax[1].set_xticklabels(['FP', 'FN', 'SplitOps'])
    ax[1].set_xlim([0,4])
    ax[1].set_ylabel('Counts')
    ax[1].legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(os.path.join(save_path, 'fig_res.svg'))
        fig.savefig(os.path.join(save_path, 'fig_res.pdf'))
        fig.savefig(os.path.join(save_path, 'fig_res.png'))
    plt.show()


if __name__ == '__main__':
    
    ########
    save_path = './Evaluation/Results/segmentation'
    ########

    os.makedirs(save_path, exist_ok=True)

    score_list = []
    score_list.append(CTC_Det_Score(res=0.81592, n_s_ops=30, n_fn=0, n_fp=220,  name='Naive', color='tab:orange'))
    score_list.append(CTC_Det_Score(res=0.924876, n_s_ops=15, n_fn=1, n_fp=66,  name='Optimized', color='tab:blue'))
    score_list.append(CTC_Det_Score(res=0.926368, n_s_ops=17, n_fn=0, n_fp=63,  name='Naive+Opti.', color='tab:green'))
    score_list.append(CTC_Det_Score(res=0.720896, n_s_ops=12, n_fn=43, n_fp=71,  name='TWANG', color='black'))

    plot_res(score_list, save_path)