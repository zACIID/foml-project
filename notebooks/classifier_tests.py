# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload

import torch
import torch.utils.data as data
import src.datasets.custom_coco_dataset as coco
import src.ensemble_training.ada_boost as ada
from src.classifiers import alex_net as ax

# %%
torch.cuda.is_available()

# %%
torch.version.cuda

# %%
dataset = coco.COCO_TRAIN_DATASET
dataset.load()

# %% [markdown]
# ## AdaBoost Tests

# %%
ada_boost = ada.AdaBoost(dataset=dataset, n_eras=1, n_classes=2)

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
alex_net = ax.AlexNet(
    act_fun=torch.nn.SiLU
)
alex_net.fit(
    dataset=dataset,
    verbose=5,
    learning_rate=0.1,
    momentum=1
)

# %%
