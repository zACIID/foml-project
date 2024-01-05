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
# # %load_ext autoreload

import os.path
from copy import deepcopy
from typing import Tuple

import torch
import torch.utils.data as data
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import src.datasets.custom_coco_dataset as coco
import src.utils.serialization as ser
import src.visualization.foml_report as foml
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

# %%
train_dataset = coco.COCO_TRAIN_DATASET
val_dataset = deepcopy(train_dataset)

train_dataset.load()
val_dataset.load()

# Don't use data augmentation for validation
val_dataset.transform = coco.COCO_TEST_DATASET.transform
val_dataset.transforms = coco.COCO_TEST_DATASET.transforms

labels_mask = train_dataset.get_labels().cpu().detach().numpy()

# %%
train_split_idxs, val_split_idxs = train_test_split(
    np.arange(0, len(train_dataset)),
    train_size=0.8,
    test_size=0.2,
    stratify=labels_mask,
    random_state=RND_SEED
)
train_subset = data.Subset(train_dataset, indices=train_split_idxs)
val_subset = data.Subset(val_dataset, indices=val_split_idxs)

train_data_loader = data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=8)
val_data_loader = data.DataLoader(val_subset, batch_size=128, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=8)

classes_mask = train_dataset.get_labels()
_, class_cardinalities = torch.unique(classes_mask[train_split_idxs], sorted=True, return_counts=True)

# %%
MODELS_DIR = os.path.join(ROOT_COCO_DIR, "models")
ALEX_NET_PATH = os.path.join(MODELS_DIR, "alex_net_state_dict.pth")
ALEX_NET_RESULTS_PATH = os.path.join(MODELS_DIR, "alex_net_train_val_results.ser")

if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

# %%
alex_net = None
train_val_results = None
if not os.path.exists(ALEX_NET_PATH):
    alex_net = ax.AlexNet()
    train_val_results = alex_net.fit_and_validate(
        train_data_loader=train_data_loader,
        validation_data_loader=val_data_loader,
        optimizer=torch.optim.Adam(alex_net.parameters(), lr=8e-5, weight_decay=1e-6),
        epochs=100,
        verbose=2
    )

    torch.save(alex_net.state_dict(), ALEX_NET_PATH)
    ser.serialize(
        filepath=ALEX_NET_RESULTS_PATH,
        obj=train_val_results
    )
else:
    alex_net = ax.AlexNet()
    alex_net = alex_net.load_state_dict(torch.load(ALEX_NET_PATH))
    train_val_results = ser.deserialize(
        type_=ax.TrainingValidationResults,
        filepath=ALEX_NET_RESULTS_PATH,
    )


# %% [markdown]
# ## Test Dataset Results

# %%
def test_results(
        data_loader: torch.utils.data.DataLoader,
        model: ax.AlexNet
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    accuracy = .0

    correct_ids = torch.tensor([], dtype=torch.int)
    wrong_ids = torch.tensor([], dtype=torch.int)
    predictions = torch.tensor([])
    for batch in tqdm(data_loader):
        batch: coco.BatchType
        ids, x_batch, y_batch, _ = batch
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        y_preds: torch.Tensor = model.predict(x_batch)


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
alex_net_accuracy, alex_net_preds, alex_net_correct_ids, alex_net_wrong_ids = test_results(data_loader=test_data_loader, model=alex_net)
alex_net_accuracy, alex_net_preds, alex_net_correct_ids, alex_net_wrong_ids

# %% [markdown]
# ## Plots

# %%
foml.plot_confusion_matrix(
    classes_mask=test_dataset.get_labels().numpy(), 
    predictions=alex_net_preds.numpy(),
    save_fig_path=os.path.join(foml.IMAGES_DIR, "alex-net-confusion-matrix.png")
)

# %%
foml.train_validation_scores_lineplot(
    training_scores=train_val_results.avg_train_loss,
    validation_scores=train_val_results.avg_validation_loss,
    save_fig_path=os.path.join(foml.IMAGES_DIR, "alex-net-train-val-loss.png")
)

# %%
foml.train_validation_scores_lineplot(
    training_scores=train_val_results.train_accuracy,
    validation_scores=train_val_results.validation_accuracy,
    y_axis_title="Accuracy",
    save_fig_path=os.path.join(foml.IMAGES_DIR, "alex-net-train-val-accuracy.png")
)

# %%
test_dataset_no_transforms = deepcopy(test_dataset)
test_dataset_no_transforms.transform = None
test_dataset_no_transforms.transforms = None

# %%
rnd_ids, _ = train_test_split(alex_net_correct_ids.to(torch.int).numpy(), train_size=10, test_size=5, stratify=test_dataset_no_transforms.get_labels()[alex_net_correct_ids.to(torch.int)])
foml.plot_images(
    ids=rnd_ids, 
    dataset=test_dataset_no_transforms,
    save_fig_dir=os.path.join(foml.IMAGES_DIR, "alex-net-correct")
)

# %%
rnd_ids, _ = train_test_split(alex_net_wrong_ids.to(torch.int).numpy(), train_size=10, test_size=5, stratify=test_dataset_no_transforms.get_labels()[alex_net_wrong_ids.to(torch.int)])
foml.plot_images(
    ids=rnd_ids, 
    dataset=test_dataset_no_transforms,
    save_fig_dir=os.path.join(foml.IMAGES_DIR, "alex-net-wrong")
)

# %%
