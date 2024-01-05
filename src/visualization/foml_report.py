import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import src.visualization.classification as vis
import src.datasets.custom_coco_dataset as coco
import src.utils.constants as const

IMAGES_DIR = os.path.join(const.ROOT_COCO_DIR, "..", "images")
if not os.path.exists(IMAGES_DIR):
    os.mkdir(IMAGES_DIR)


def plot_confusion_matrix(
        classes_mask: np.ndarray,
        predictions: np.ndarray,
        save_fig_path: str | os.PathLike,
):
    fig = vis.confusion_matrix(
        y_true=classes_mask,
        y_pred=predictions,
        figsize=(12, 12)
    )
    fig.savefig(save_fig_path)


# Sample data (replace this with your actual training and validation errors)
# epochs = [1, 2, 3, 4, 5]  # X-axis values (e.g., epochs)
# training_errors = [0.5, 0.4, 0.3, 0.2, 0.1]  # Training errors for each epoch
# validation_errors = [0.6, 0.5, 0.4, 0.3, 0.2]  # Validation errors for each epoch


def train_validation_scores_lineplot(
        training_scores: np.ndarray,
        validation_scores: np.ndarray,
        save_fig_path: str | os.PathLike,
        training_line_title = "Training",
        validation_line_title = "Validation",
        y_axis_title = "Loss",
):
    assert len(training_scores) == len(validation_scores), "Train and Val scores must have same length"
    epochs = list(range(len(training_scores)))
    combined_errors = {
        'Epochs': epochs * 2,
        'Error': training_scores + validation_scores,
        'Title': [training_line_title] * len(epochs) + [validation_line_title] * len(epochs)}
    df = pd.DataFrame(combined_errors)

    # Set Seaborn style
    sns.set(style='whitegrid')

    # Plotting both lines on the same plot
    sns_plot = sns.lineplot(x='Epochs', y='Error', hue='Title', data=df, marker='o')

    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel(y_axis_title)
    plt.title('Training and Validation Scores Over Epochs')

    # Adding a legend
    plt.legend()

    # Display the plot
    plt.show()

    fig = sns_plot.get_figure()
    fig.savefig(save_fig_path)


def plot_images(
        ids,
        dataset: coco.CocoDataset,
        save_fig_dir: str | os.PathLike,
):
    """
    Useful to plot correct and wrong instances
    """

    if not os.path.exists(save_fig_dir):
        os.mkdir(save_fig_dir)

    for img_id in ids:
        batch: coco.ItemType = dataset[img_id]
        _, img, label, weight = batch
        # I = io.imread(img.permute(1, 2, 0))
        # print(type(I))

        plt.axis('off')
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"Class: {label}")
        plt.show()
        plt.savefig(os.path.join(save_fig_dir, f"img_{img_id}.png"))
