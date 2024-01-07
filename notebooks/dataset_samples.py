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

import src.datasets.custom_coco_dataset as coco
import src.visualization.foml_report as foml

# %%
train_dataset = coco.COCO_TRAIN_DATASET
train_dataset.load()

# %%
foml.plot_images(ids=range(20), dataset=train_dataset, save_fig_dir=os.path.join(foml.IMAGES_DIR, "dataset-samples"))

# %%
