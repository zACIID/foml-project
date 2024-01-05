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
import os.path

import numpy as np
# %load_ext autoreload

import torch
import torch.utils.data as data
import src.datasets.custom_coco_dataset as coco
import src.ensemble_training.ada_boost as ada
from src.classifiers import alex_net as ax
from src.utils.constants import RND_SEED, ROOT_COCO_DIR

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
    train_size=110000, 
    test_size=7500, 
    stratify=labels_mask, 
    random_state=RND_SEED
)
alex_net_train_subset = data.Subset(alex_net_train_dataset, indices=train_split_idxs)
ada_boost_train_subset = data.Subset(ada_boost_train_dataset, indices=train_split_idxs)
val_subset = data.Subset(val_dataset, indices=val_split_idxs)

alex_net_train_data_loader = data.DataLoader(alex_net_train_subset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=8)
ada_boost_train_data_loader = data.DataLoader(ada_boost_train_subset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4)
val_data_loader = data.DataLoader(val_subset, batch_size=128, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=8)

classes_mask = ada_boost_train_dataset.get_labels()
_, class_cardinalities = torch.unique(classes_mask[train_split_idxs], sorted=True, return_counts=True)

# %% [raw]
# ada_boost = ada.AdaBoost(n_classes=2) #device=torch.device("cpu"))
# strong_learner, train_val_results = ada_boost.fit_and_validate(
#     eras=50,
#     train_data_loader=ada_boost_train_data_loader,
#     validation_data_loader=val_data_loader,
#     classes_mask=classes_mask,
#     class_cardinalities=class_cardinalities,
#     actual_train_dataset_length=len(ada_boost_train_dataset),
#     #weak_learner_optimizer_builder=lambda params: torch.optim.SGD(params, lr=10000), # weight_decay=1e-6),
#     weak_learner_optimizer_builder=lambda params: torch.optim.Adam(params, lr=9e-3), # weight_decay=1e-6),
#     weak_learner_epochs=1,
#     verbose=2,
# )

# %%
alex_net = ax.AlexNet()
train_val_results = alex_net.fit_and_validate(
    train_data_loader=alex_net_train_data_loader, 
    validation_data_loader=val_data_loader,
    optimizer=torch.optim.Adam(alex_net.parameters(), lr=8e-5, weight_decay=1e-6),
    epochs=200,
    verbose=2
)

# %%

# %%
import skimage.io as io
from typing import Callable, Tuple
from datasets.custom_coco_dataset import BatchType
from tqdm import tqdm


def test_results(
        data_loader: torch.utils.data.DataLoader, 
        model: Callable[[torch.Tensor], torch.Tensor]
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    accuracy = .0

    correct_ids = torch.tensor([])
    wrong_ids = torch.tensor([])
    predictions = torch.tensor([])
    for batch in tqdm(data_loader):
        batch: BatchType
        ids, x_batch, y_batch, _ = batch
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        y_preds: torch.Tensor = model.predict(x_batch)

        # print(y_batch)
        # print(predictions)
        
        # noinspection PyUnresolvedReferences
        correctness_mask = (y_preds == y_batch).cpu()
        ids = ids.cpu().to(torch.int)
        correct_ids = torch.cat((correct_ids, ids[correctness_mask]))
        wrong_ids = torch.cat((wrong_ids, ids[~correctness_mask]))
        predictions = torch.cat((predictions, y_preds.cpu()))
        
        accuracy += correctness_mask.sum().item() / len(data_loader.dataset)
    return accuracy, predictions, correct_ids, wrong_ids



# %%
test_dataset = coco.COCO_TEST_DATASET
test_dataset.load()

# %%
test_data_loader = data.DataLoader(dataset=test_dataset, batch_size=32, num_workers=4, pin_memory=True)

# %%
import os
from src.utils.constants import RND_SEED, ROOT_COCO_DIR
import src.utils.serialization as ser

MODELS_DIR = os.path.join(ROOT_COCO_DIR, "models")
ALEX_NET_PATH = os.path.join(MODELS_DIR, "alex_net.torch")
ALEX_NET_RESULTS_PATH = os.path.join(MODELS_DIR, "alex_net_results.ser")
ADA_BOOST_PATH = os.path.join(MODELS_DIR, "ada_boost.torch")
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
    
torch.save(alex_net.state_dict(), ALEX_NET_PATH)
ser.serialize(
    filepath=ALEX_NET_RESULTS_PATH,
    obj=train_val_results
)

# %%
alex_net_accuracy, alex_net_preds, alex_net_correct_ids, alex_net_wrong_ids = test_results(data_loader=test_data_loader, model=alex_net)
alex_net_accuracy, alex_net_preds, alex_net_correct_ids, alex_net_wrong_ids

# %%

# %% [markdown]
# ## Plotting Stuff TODO

# %%
alex_net_train_results = ser.deserialize_or_save_object(
    type_=ax.TrainingValidationResults,
    filepath=ALEX_NET_RESULTS_PATH,
    builder=lambda: train_val_results,
)

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import src.visualization.classification as vis


IMAGES_DIR = os.path.join(ROOT_COCO_DIR, "..", "images")
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
        figsize=(12,12)
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


# %%
plot_confusion_matrix(
    classes_mask=test_dataset.get_labels().numpy(), 
    predictions=alex_net_preds.numpy(),
    save_fig_path=os.path.join(IMAGES_DIR, "alex-net-confusion-matrix.png")
)

# %%
train_validation_scores_lineplot(
    training_scores=alex_net_train_results.avg_train_loss,
    validation_scores=alex_net_train_results.avg_validation_loss,
    save_fig_path=os.path.join(IMAGES_DIR, "alex-net-train-val-loss.png")
)

# %%
train_validation_scores_lineplot(
    training_scores=alex_net_train_results.train_accuracy,
    validation_scores=alex_net_train_results.validation_accuracy,
    y_axis_title="Accuracy",
    save_fig_path=os.path.join(IMAGES_DIR, "alex-net-train-val-accuracy.png")
)

# %%
# test_dataset_no_transforms =  

# %%
test_dataset_no_transforms = deepcopy(test_dataset)
test_dataset_no_transforms.transform = None
test_dataset_no_transforms.transforms = None

# %%
rnd_ids, _ = train_test_split(alex_net_correct_ids.to(torch.int).numpy(), train_size=10, test_size=5, stratify=test_dataset_no_transforms.get_labels()[alex_net_correct_ids.to(torch.int)])
plot_images(
    ids=rnd_ids, 
    dataset=test_dataset_no_transforms,
    save_fig_dir=os.path.join(IMAGES_DIR, "alex-net-correct")
)

# %%
rnd_ids, _ = train_test_split(alex_net_wrong_ids.to(torch.int).numpy(), train_size=10, test_size=5, stratify=test_dataset_no_transforms.get_labels()[alex_net_wrong_ids.to(torch.int)])
plot_images(
    ids=rnd_ids, 
    dataset=test_dataset_no_transforms,
    save_fig_dir=os.path.join(IMAGES_DIR, "alex-net-wrong")
)

# %%
