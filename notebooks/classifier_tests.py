# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
# %load_ext autoreload

import torch
import torch.utils.data as data
import src.datasets.custom_coco_dataset as coco
import src.ensemble_training.ada_boost as ada
from src.classifiers import alex_net as ax
from src.utils.constants import RND_SEED

# %%
torch.cuda.is_available()

# %%
torch.version.cuda

# %%
torch.manual_seed(RND_SEED)

# %%
dataset = coco.COCO_TEST_DATASET
dataset.load()

# %% [markdown]
# ## AdaBoost Tests

# %%
# ada_boost = ada.AdaBoost(dataset=dataset, n_eras=1, n_classes=2)

# %% [raw]
# strong_learner = ada_boost.start(verbose=4)

# %% [raw]
# subset = data.Subset(dataset=dataset, indices=[x for x in range(48)])
# for batch in data.DataLoader(subset, 4, shuffle=True):
#     batch: coco.BatchType
#     ids, imgs, labels, weights = batch
#     preds = strong_learner.predict_image(imgs)
#     
#     print(f"Predictions:\n{preds}")
#     print(f"Actual:\n{labels}")

# %% [markdown]
# ## AlexNet Tests

# %%
# y_true = torch.tensor([[0, 1.0], [1.0, 0]])
# y_pred = torch.tensor([[0.2, -0.3], [0.1, -0.4]])
# 
# cross_entropy: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss(reduction='none')
# x = cross_entropy(y_pred, y_true)
# x = x.mean()
# print(x)

# %%
# alex_net = ax.AlexNet(
#     act_fun=torch.nn.SiLU
# )
# alex_net.fit(
#     dataset=dataset,
#     verbose=5,
#     learning_rate=0.0005,
#     momentum=0.9,
#     batch_size=128,
#     epochs=100
# )

# %% [markdown]
# ## TODO Presentazione
#
# - dire che abbiamo scelto AlexNet come ispriazione, non l'abbiamp proprio seguita perche dataset diverso, image transformation diversi (loro fanno anche flip, noi invece semplicemente normalizziamo perche abbiamo un numero sufficiente di immagini) e non usiamo normalizzazione tra layer
# - dire che per adaboost abbiamo preso squeezenet come ispirazione ma l'abbiamo castrata, tenendo solo i layer meno profondi di ogni "blocco" (perche intuizione e che si puo andare piu in profodnita, ovvero aumentare numero di filtri per layer, mano a mano che la il numero di layer aument
# - paralre del dataset utilizzato e come e stato rielaborato in brevissimo
# - dire che per adaboost + squeezenet volevamo comparare i due modelli idealmente con un numero di parametri simile, quindi numero weak learner scelto in modo tale che numero di aprameteri simile ad AlexNet
# - mostrare i grafici
#

# %% [markdown]
# ## TODO grafici
#
# ### AlexNet
#
# - training (+ validation) loss di SGD con i parameteri di AlexNet
# - training (+ validation) loss di Adam con i nostri parametri (lr=0.0005, weight_decay=5e-3)
#
# Previsioni:
# - training accuracy
# - test accuracy
# - training confusion matrix
# - testing confusion matrix
#
# ### Adaboost
#
# Come AlexNet, in aggiunta:
# - vedere comportamento loss in base a numero di weak learner
#

# %% [markdown]
# ### Altri TODO
#
# - passare in qualche modo un parametro a cocodataset che mi permetta di sceglierne solo un sottoinsieme??? oppure passarlo al metodo train -> questo mi serve sia per velocizzare training e testing che per fare gli split per la validation
#     - per scegliere indici randomici posso usare train_test_split di sklearn passando un array di indici (setto RND SEED)
#     - poi uso subset passandoci la lista di indici per definire dataset di train e test sul dataloader
# - metodo train esterno al modello???
#     - questo metodo ritorna una struct di dati utili tipo loss per epoca e altri parametri 
# - passare optimizer e criterion direttamente come parametri di train
# - eseguo metodo esterno train() su dataset di train e validation
# - usiamo un decimo del dataset (12k, auspicabilmente bilanciate) per train + validation che altrimenti 100epoche ci si mette un giorno

# %% [markdown]
# ## Validation

# %%
import src.visualization.classification as vis

# for batch in dataset:
#     ids, images, labels, weights = batch
#     y_pred = model.predict()
#     vis.confusion_matrix(
#         y_true=labels,
#         y_pred=y_pred,
#         figsize=(12,12)
#     )

# %%
from copy import deepcopy
import torch.utils.data as data
from sklearn.model_selection import train_test_split

alex_net_train_dataset = coco.COCO_TRAIN_DATASET
ada_boost_train_dataset = deepcopy(alex_net_train_dataset)
val_dataset = deepcopy(alex_net_train_dataset)

alex_net_train_dataset.load()
ada_boost_train_dataset.load()
val_dataset.load()

# Don't use data augmentation for adaboost: samples are transformed on the fly,
#   meaning that the whole reasoning behind boosting, i.e. that of being able to
#   focus on the specific wrong samples, falls apart
ada_boost_train_dataset.transform = coco.COCO_TEST_DATASET.transform
ada_boost_train_dataset.transforms = coco.COCO_TEST_DATASET.transforms

# Don't use data augmentation for validation
val_dataset.transform = coco.COCO_TEST_DATASET.transform
val_dataset.transforms = coco.COCO_TEST_DATASET.transforms

labels_mask = alex_net_train_dataset.get_labels().cpu().detach().numpy()

# %%
train_split_idxs, val_split_idxs = train_test_split(
    np.arange(0, len(ada_boost_train_dataset)), 
    train_size=5000, 
    test_size=1000, 
    stratify=labels_mask, 
    random_state=RND_SEED
)
alex_net_train_subset = data.Subset(alex_net_train_dataset, indices=train_split_idxs)
ada_boost_train_subset = data.Subset(ada_boost_train_dataset, indices=train_split_idxs)
val_subset = data.Subset(val_dataset, indices=val_split_idxs)

alex_net_train_data_loader = data.DataLoader(alex_net_train_subset, batch_size=128, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=8)
ada_boost_train_data_loader = data.DataLoader(ada_boost_train_subset, batch_size=128, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=8)
val_data_loader = data.DataLoader(val_subset, batch_size=128, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=8)

classes_mask = ada_boost_train_dataset.get_labels()
_, class_cardinalities = torch.unique(classes_mask[train_split_idxs], sorted=True, return_counts=True)

# %% is_executing=true
ada_boost = ada.AdaBoost(n_classes=2) #device=torch.device("cpu"))
strong_learner, train_val_results = ada_boost.fit_and_validate(
    eras=15,
    train_data_loader=ada_boost_train_data_loader,
    validation_data_loader=val_data_loader,
    classes_mask=classes_mask,
    actual_train_dataset_length=len(ada_boost_train_dataset),
    weak_learner_optimizer_builder=lambda params: torch.optim.Adam(params, lr=1e-3, weight_decay=1e-6),
    weak_learner_epochs=10,
    verbose=2,
)

# %% [raw]
# alex_net = ax.AlexNet()
# train_val_results = alex_net.fit_and_validate(
#     train_data_loader=alex_net_train_data_loader, 
#     validation_data_loader=val_data_loader,
#     optimizer=torch.optim.Adam(alex_net.parameters(), lr=8e-5, weight_decay=1e-6),
#     epochs=200,
#     verbose=2
# )

# %%
import random

import skimage.io as io
from typing import Callable, Tuple
from datasets.custom_coco_dataset import BatchType
from tqdm import tqdm


def test_results(
        data_loader: torch.utils.data.DataLoader, 
        model: Callable[[torch.Tensor], torch.Tensor]
) -> Tuple[float, np.ndarray, np.ndarray]:
    accuracy = .0

    correct_ids = torch.tensor([])
    wrong_ids = torch.tensor([])
    for batch in tqdm(data_loader):
        batch: BatchType
        ids, x_batch, y_batch, _ = batch
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        predictions: torch.Tensor = model(x_batch)

        # noinspection PyUnresolvedReferences
        correctness_mask = (predictions == y_batch).cpu()
        ids = ids.cpu()
        correct_ids = torch.cat(correct_ids, ids[correctness_mask])
        wrong_ids = torch.cat(wrong_ids, ids[~correctness_mask])
        
        accuracy += correctness_mask.sum().item() / len(data_loader.dataset)
    return accuracy, correct_ids, wrong_ids



# %%
test_results(data_loader=val_data_loader, model=strong_learner[0])

# %%

# %% [markdown]
# ## Plotting Stuff TODO

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data (replace this with your actual training and validation errors)
# epochs = [1, 2, 3, 4, 5]  # X-axis values (e.g., epochs)
# training_errors = [0.5, 0.4, 0.3, 0.2, 0.1]  # Training errors for each epoch
# validation_errors = [0.6, 0.5, 0.4, 0.3, 0.2]  # Validation errors for each epoch

def train_validation_scores_lineplot(
        training_scores: np.ndarray,
        validation_scores: np.ndarray,
        training_line_title = "Training",
        validation_line_title = "Validation",
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
    sns.lineplot(x='Epochs', y='Error', hue='Title', data=df, marker='o')

    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Training and Validation Scores Over Epochs')

    # Adding a legend
    plt.legend()

    # Display the plot
    plt.show()


def plot_images(ids, dataset: coco.CocoDataset):
    """
    Useful to plot correct and wrong instances
    """
    for img_id in ids:
        batch: coco.ItemType = dataset[img_id]
        _, img, label, weight = batch
        I = io.imread(img.permute(1, 2, 0))
        print(type(I))

        plt.axis('off')
        plt.imshow(I)
        plt.show()

