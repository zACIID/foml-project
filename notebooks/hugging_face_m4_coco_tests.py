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
from datasets import load_dataset
import os

DATA_DIR = os.path.join(os.getcwd(), "..", "data")
print(DATA_DIR)

# dataset = load_dataset("ydshieh/coco_dataset_script", "2017", data_dir="./dummy_data/")
dataset = load_dataset("HuggingFaceM4/COCO")

# %%
import datasets.splits as split

train = dataset[split.Split.TRAIN]
test = dataset[split.Split.TEST]
val = dataset[split.Split.VALIDATION]

# %%
train.column_names

# %%
train.num_rows

# %%
train[0]["image"]

# %%
train[250000]["filepath"]

# %%
train.info

# %%
val[0]["filepath"]

# %%
test[25000]["filepath"]

# %%
