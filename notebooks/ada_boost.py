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
import src.ensemble_training.ada_boost as ada
import src.classifiers.strong_learner as strong
from classifiers.simple_learner import WeakLearnerValidationResults
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
train_dataset = deepcopy(coco.COCO_TRAIN_DATASET)
val_dataset = deepcopy(train_dataset)

train_dataset.load()
val_dataset.load()

# Don't use data augmentation for adaboost: samples are transformed on the fly,
#   meaning that the whole reasoning behind boosting, i.e. that of being able to
#   focus on the specific wrong samples, falls apart
train_dataset.transform = coco.COCO_TEST_DATASET.transform
train_dataset.transforms = coco.COCO_TEST_DATASET.transforms

# Don't use data augmentation for validation
val_dataset.transform = coco.COCO_TEST_DATASET.transform
val_dataset.transforms = coco.COCO_TEST_DATASET.transforms

labels_mask = train_dataset.get_labels().cpu().detach().numpy()

# %%
train_split_idxs, val_split_idxs = train_test_split(
    np.arange(0, len(train_dataset)),
    train_size=9000,#0.8,
    test_size=500,
    stratify=labels_mask,
    random_state=RND_SEED
)
train_subset = data.Subset(train_dataset, indices=train_split_idxs)
val_subset = data.Subset(val_dataset, indices=val_split_idxs)

train_data_loader = data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4)
val_data_loader = data.DataLoader(val_subset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=1)

classes_mask = train_dataset.get_labels()
_, train_split_class_cardinalities = torch.unique(classes_mask[train_split_idxs], sorted=True, return_counts=True)

# %%
MODELS_DIR = os.path.join(ROOT_COCO_DIR, "models")
ADA_BOOST_TRAIN_VAL_PATH = os.path.join(MODELS_DIR, "ada_boost-train_val.pth")
ADA_BOOST_TRAIN_VAL_RESULTS_PATH = os.path.join(MODELS_DIR, "ada_boost-train_val_results.ser")
ADA_BOOST_FULL_PATH = os.path.join(MODELS_DIR, "ada_boost-full.pth")

if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

# %% is_executing=true
strong_learner_val = None
train_val_results = None
if not os.path.exists(ADA_BOOST_TRAIN_VAL_PATH):
    ada_boost = ada.AdaBoost(n_classes=2) #device=torch.device("cpu"))
    strong_learner_val, train_val_results = ada_boost.fit_and_validate(
        eras=50,
        train_data_loader=train_data_loader,
        validation_data_loader=val_data_loader,
        classes_mask=classes_mask,
        actual_train_ids=train_split_idxs,
        total_train_dataset_length=len(train_dataset),
        #weak_learner_optimizer_builder=lambda params: torch.optim.SGD(params, lr=10000), # weight_decay=1e-6),
        weak_learner_optimizer_builder=lambda params: torch.optim.Adam(params, lr=9e-3),#, weight_decay=1e-6),
        weak_learner_epochs=12,
        verbose=2,
    )


    torch.save(strong_learner_val, ADA_BOOST_TRAIN_VAL_PATH)
    ser.serialize(
        filepath=ADA_BOOST_TRAIN_VAL_RESULTS_PATH,
        obj=train_val_results
    )
else:
    strong_learner_val = torch.load(ADA_BOOST_TRAIN_VAL_PATH)
    train_val_results = ser.deserialize(
        type_=WeakLearnerValidationResults,
        filepath=ADA_BOOST_TRAIN_VAL_RESULTS_PATH,
    )


# %% [markdown]
# ## Training Full Model

# %% [raw]
# train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=8)

# %% [raw]
# strong_learner_full = None
# if not os.path.exists(ADA_BOOST_FULL_PATH):
#     ada_boost = ada.AdaBoost(n_classes=2) #device=torch.device("cpu"))
#     strong_learner_full, train_val_results = ada_boost.fit_and_validate(
#         eras=50,
#         train_data_loader=train_data_loader,
#         validation_data_loader=val_data_loader,
#         classes_mask=classes_mask,
#         class_cardinalities=class_cardinalities,
#         actual_train_dataset_length=len(train_dataset),
#         #weak_learner_optimizer_builder=lambda params: torch.optim.SGD(params, lr=10000), # weight_decay=1e-6),
#         weak_learner_optimizer_builder=lambda params: torch.optim.Adam(params, lr=9e-3), # weight_decay=1e-6),
#         weak_learner_epochs=1,
#         verbose=2,
#     )
#
#     torch.save(strong_learner, ADA_BOOST_FULL_PATH)
# else:
#     ada_boost = torch.load(ADA_BOOST_FULL_PATH)

# %% [markdown]
# ## Test Dataset Results

# %%
def test_results(
        data_loader: torch.utils.data.DataLoader,
        model: strong.StrongLearner
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
accuracy, preds, correct_img_ids, wrong_image_ids = test_results(data_loader=test_data_loader, model=strong_learner_val)
accuracy, preds, correct_img_ids, wrong_image_ids

# %% [markdown]
# ## Plots

# %%
foml.plot_confusion_matrix(
    classes_mask=test_dataset.get_labels().numpy(), 
    predictions=preds.numpy(),
    save_fig_path=os.path.join(foml.IMAGES_DIR, "ada-boost-confusion-matrix.png")
)

# %%
foml.train_validation_scores_lineplot(
    training_scores=train_val_results.avg_train_loss,
    validation_scores=train_val_results.avg_validation_loss,
    save_fig_path=os.path.join(foml.IMAGES_DIR, "ada-boost-train-val-loss.png")
)

# %%
foml.train_validation_scores_lineplot(
    training_scores=train_val_results.train_accuracy,
    validation_scores=train_val_results.validation_accuracy,
    y_axis_title="Accuracy",
    save_fig_path=os.path.join(foml.IMAGES_DIR, "ada-boost-train-val-accuracy.png")
)

# %%
test_dataset_no_transforms = deepcopy(test_dataset)
test_dataset_no_transforms.transform = None
test_dataset_no_transforms.transforms = None

# %%
rnd_ids, _ = train_test_split(correct_img_ids.to(torch.int).numpy(), train_size=10, test_size=5, stratify=test_dataset_no_transforms.get_labels()[correct_img_ids.to(torch.int)])
foml.plot_images(
    ids=rnd_ids, 
    dataset=test_dataset_no_transforms,
    save_fig_dir=os.path.join(foml.IMAGES_DIR, "ada-boost-correct")
)

# %%
rnd_ids, _ = train_test_split(wrong_image_ids.to(torch.int).numpy(), train_size=10, test_size=5, stratify=test_dataset_no_transforms.get_labels()[wrong_image_ids.to(torch.int)])
foml.plot_images(
    ids=rnd_ids, 
    dataset=test_dataset_no_transforms,
    save_fig_dir=os.path.join(foml.IMAGES_DIR, "ada-boost-wrong")
)

# %%
