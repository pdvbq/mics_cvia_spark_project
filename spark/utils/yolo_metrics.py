import logging
import os
from numpy import save
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import matplotlib.pyplot as plt

logger = logger = logging.getLogger(__name__)

def generate_yolo_metrics(results_path: str) -> None:
    if not os.path.isfile(results_path):
        logger.error(f"Make sure {results_path} is a .csv file")

    dirname = os.path.dirname(results_path)
    results_data = pd.read_csv(results_path)

    plot_results_group(results_data, os.path.join(dirname, "results.png"))


def plot_results_group(data: pd.DataFrame, save_path: str): 
    # Apply smoothing to data
    smoothed_data = data.copy()
    for col in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 
                'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
                'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 
                'metrics/mAP50-95(B)']:
        smoothed_data[col] = gaussian_filter1d(data[col], sigma=2)

    # Set up a grid of subplots
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()

    # Titles for the subplots
    titles = [
        'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
        'metrics/precision(B)', 'metrics/recall(B)',
        'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
        'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
    ]

    # Columns to plot
    columns = [
        'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
        'metrics/precision(B)', 'metrics/recall(B)',
        'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
        'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
    ]

    # Plot each metric in a subplot
    for i, ax in enumerate(axes):
        ax.plot(data['epoch'], data[columns[i]], label='results', marker='o')
        ax.plot(data['epoch'], smoothed_data[columns[i]], label='smooth', linestyle='dotted')
        ax.set_title(titles[i])
        ax.legend()
        ax.grid()

    # Adjust layout and display
    plt.tight_layout()
    plt.savefig(save_path)
