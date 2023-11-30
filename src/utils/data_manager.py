import os
from PIL import Image
from numpy import array, ndarray
from torch import tensor, Tensor
from tensor_utility import tensorAppend


"""
    Class which given a directory offer an ensemble of methods to 
    turn images into tensor
"""


class DataManager:

    def __init__(self, dir_path: str):
        self._dir_path: str = dir_path

    def _from_img_to_tensor(self, img_path: str) -> Tensor:
        total_path: str = os.path.join(self._dir_path, img_path)
        with Image.open(total_path) as image:
            data: ndarray[float] = array(image)
            return tensor(data)

    def get_img(self, img_path: str) -> Tensor:
        return self._from_img_to_tensor(img_path)

    def get_dataset_RAM_in(self) -> Tensor:
        images_ids: list[str] = os.listdir(self._dir_path)
        images: Tensor = tensor([])
        for image_id in images_ids:
            image: Tensor = self.get_img(image_id)
            images = tensorAppend(images, image, retain_dim=True)

        return images

    def get_dataset_RAM_out(self) -> list[Tensor]:
        images_ids: list[str] = os.listdir(self._dir_path)
        images: list[Tensor] = []
        for image_id in images_ids:
            image: Tensor = self.get_img(image_id)
            images.append(image)

        return images

