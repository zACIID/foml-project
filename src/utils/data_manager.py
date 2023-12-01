import os
from PIL import Image
from numpy import array, ndarray
from torch import tensor, Tensor
from typing import Generator
import type_utility as tu


"""
    Class which given a directory offer an ensemble of methods to 
    turn images into tensor
"""


class DataManager:

    def __init__(self, dir_path: str):
        self._dir_path: str = dir_path

    def _from_img_to_tensor(self, img_path: str) -> Tensor:
        total_path: str = os.path.join(self._dir_path, img_path)
        with Image.open(total_path) as img:
            data: tu.image = array(img)
            return tensor(data)

    def get_img(self, img_path: str) -> Tensor:
        return self._from_img_to_tensor(img_path)

    def get_dataset(self) -> Generator[Tensor, None, None]:
        images_ids: list[str] = os.listdir(self._dir_path)
        for image_id in images_ids:
            yield self.get_img(image_id)

    def get_list_dataset(self) -> list[Tensor]:
        return list(self.get_dataset())

    """
        Worth to mention that each tensor was initialized with the flag requires_grad=False
        this means that when we will pass this tensors to our models we need to set 
        requires_grad=True otherwise the nets would not be able to learn. Thus for each tensor
        representing an image (img) we just need to call the method img.requires_grad_()
    """

