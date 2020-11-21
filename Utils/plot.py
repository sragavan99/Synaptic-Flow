import matplotlib.pyplot as plt
import pandas as pd

def get_plots(df, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    # plots of losses
    ax[0].plot(df['train_loss_trainmode'], label='train (train mode)')
    ax[0].plot(df['train_loss'], label='train (eval mode)')
    ax[0].plot(df['test_loss'], label='test')
    ax[0].set_title("Losses")
    ax[0].legend()

    # plots of accuracies
    ax[1].plot(df['train_top1'], label='train')
    ax[1].plot(df['test_top1'], label='test')
    ax[1].set_title("Accuracies")
    ax[1].legend()

    if save_path:
        fig.savefig(save_path)

    fig.show()