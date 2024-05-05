# -*- coding: utf-8 -*-
"""File for shared utilities between modules

Example:
    To use this module, import it into your python environment.

        >>> from src.utils.shared_utils import *

Sources:
    * shared_utilities.py from https://lightning.ai/lightning-ai/studios/dl-fundamentals-6-dl-tips-tricks?path=cloudspaces%2F01hp75nkp82mg3efpcdy3c03a9%2Fcode-units%2F6.1-checkpointing&tab=files&layout=column&y=1&x=2
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from experiments.n2c2.twenty18.task2.RE.config import MAX_EPOCHS

def plot_json_logger(
    json_path: str, out_path: str, 
    loss_names: lst[str,str] = ["training_loss_per_epoch", "val_loss"], 
    eval_names: lst[str,str] =["train_F1_macro", "val_F1_macro"],
    ) -> None:
    """Plot validation & training loss & mF1 against one another & save figures.

    Args:
        json_path: path to json file
        loss_names: column headings for loss
        eval_names: column headings for mF1 against eachother
    
    TODO:
        * Get this to work in trainer.py right before testing
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    metrics = pd.read_json(json_path)
    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)
    df_metrics = pd.DataFrame(aggreg_metrics)

    # Plot and save evaluation metrics
    plt.figure(figsize=(8, 6))
    df_metrics[eval_names].plot(grid=True, legend=True, xlabel="Epoch", ylabel="F1 Score", xlim=[0, MAX_EPOCHS], linestyle='-', marker='o')
    plt.title("Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "metrics.pdf"), format='pdf')
    plt.close()

    # Plot and save loss
    plt.figure(figsize=(8, 6))
    df_metrics[loss_names].plot(grid=True, legend=True, xlabel="Epoch", ylabel="Loss", xlim=[0, MAX_EPOCHS], linestyle='-', marker='o')
    plt.title("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "loss.pdf"), format='pdf')
    plt.close()

