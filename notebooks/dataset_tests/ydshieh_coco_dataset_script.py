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

dataset = load_dataset("ydshieh/coco_dataset_script", "2017", data_dir="./dummy_data/")

# %%
dataset

# %%
print(list(dataset.keys()))

# %%
dataset["train"].__dict__

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
train[0]["image_id"]

# %%
train[0]["image_path"]

# %%
train.info

# %%
test[25000]["filepath"]

# %%
