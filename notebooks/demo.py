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

# %% [markdown]
# # Constants declaration

# %%
import os

import src.visualization.foml_report as foml
import src.utils.constants as const

# %% [markdown]
# # Models fetcher

# %%
MODELS_DIR = os.path.join(const.ROOT_COCO_DIR, "models")
ADA_BOOST_TRAIN_VAL_PATH = os.path.join(MODELS_DIR, "ada_boost-train_val.pth")
ALEX_NET_PATH = os.path.join(MODELS_DIR, "alex_net-full.pth")

# %%
from typing import TypeVar

T = TypeVar('T')


class ModelHandler:
    @staticmethod
    def store_model(model: T, path: str):
        torch.dump(model, path)

    @staticmethod
    def get_model(path: str) -> T:
        return torch.load(path)



# %% [markdown]
# # Take a photo

# %%
import cv2

def get_a_pic(out_dir: str) -> str:
    # Open a connection to the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    # Capture a single frame
    ret, frame = cap.read()

    # Release the camera connection
    cap.release()

    # Specify the filename (you can customize this)
    filename: str = "demo_image.jpg"

    # Create the full path
    full_path: str = f"{out_dir}/{filename}"

    # Save the captured frame to the specified directory
    cv2.imwrite(full_path, frame)

    return full_path


# %%
import os

os.getcwd()

# %% [markdown]
# # Demo 

# %% [markdown]
# ### Get model and print the output 

# %%
from torch import Tensor

def demo_runner(model_path: str, img: Tensor) -> None:
    model = ModelHandler.get_model(model_path)

    model_name: str = os.path.basename(model_path)
    pred: int = model(img).item()

    if not pred:
        print(f"\033[32m{model_name}: Human detected\033[0m", end='\n\n')
    else:
        print(f"\033[91m{model_name}: No human detected\033[0m", end='\n\n')


# %% [markdown]
# ### Preprocess the image 

# %%
import torchvision.transforms.v2 as t2
import torch

def preprocess(img_path: str) -> Tensor:
    img: Tensor = tv.io.read_image(path=img_path)
    resizer = t2.Resize(size=const.INPUT_IMAGE_SIZE, antialias=True)
    dtyper = t2.ToDtype(torch.float32)
    normalizer = t2.Normalize(mean=const.IMAGE_NET_IMAGE_MEANS, std=const.IMAGE_NET_IMAGE_STDS)
    
    img = resizer(img)
    img = dtyper(img)
    img = normalizer(img)
    
    return img


# %% [markdown]
# ### Demo runner

# %%
import torchvision as tv

def demo_handler(adaboost_path: str, alex_net_path: str, img_dir_path: str) -> None:
    img_path: str = get_a_pic(img_dir_path)
    img: Tensor = preprocess(img_path)
    
    demo_runner(adaboost_path, img)
    demo_runner(alex_net_path, img)


# %%
demo_handler(alex_net_path=ALEX_NET_PATH, adaboost_path=ADA_BOOST_TRAIN_VAL_PATH, img_dir_path=foml.IMAGES_DIR)

# %% [markdown]
# ### A Test

# %% [raw]
# class FakeModel: 
#     def __init__(self, model_a: bool = True): 
#         self.model_a: bool = model_a
#     
#     def __call__(self, img: Tensor) -> Tensor:
#         return torch.tensor([[0]]) if self.model_a else torch.tensor([[1]])

# %% [raw]
# model1 = FakeModel()
# model2 = FakeModel(False)
#
# ModelHandler.store_model(model1, "models/model1.plk")
# ModelHandler.store_model(model2, "models/model2.plk")

# %%
demo_handler("models/model1.plk", "models/model2.plk", "img")
