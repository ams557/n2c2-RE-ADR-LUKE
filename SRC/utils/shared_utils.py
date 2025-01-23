import os
import pandas as pd
import matplotlib.pyplot as plt
from experiments.n2c2.twenty18.task2.RE.config import MAX_EPOCHS

def plot_json_logger(
    json_path, out_path, 
    loss_names=["training_loss_per_epoch", "val_loss"], 
    eval_names=["train_F1_macro", "val_F1_macro"],
    ):
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

