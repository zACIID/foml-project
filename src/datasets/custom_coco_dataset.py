import enum
import os.path
from typing import Any, Callable, List, Optional, Tuple

import torch
import torchvision as tv
import torchvision.transforms.v2 as t2
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset



class CustomCocoLabels(enum.IntEnum):
    PERSON = 1
    OTHER = 2


# Notes on performance:
# PyTorch recommends using transforms.v2 as well as Tensors (as opposed to PIL images)
# [reference](https://pytorch.org/vision/stable/transforms.html#v1-or-v2-which-one-should-i-use)

# TODO(pierluigi): stuff to do
# - I need to load images and cast the category id to CustomCocoLabels
# - I need to consequently deal with class imbalance. what to do?
#   - I was reading this https://github.com/muellerzr/fastai-Experiments-and-tips/blob/master/Class%20Imbalances/Class_Imbalance_Experiments.ipynb
#       which was about the use of oversampling versus stratified sampling
#   - do I just sample X images from those with different category ids and rebuild the dataset?
#   - note on stratified sampling: the idea that each batch has the balanced classes so that
#       stochastic gradient descent is more smooth maybe makes sense?
#       Check the study above for an answer. Additionally, if I were to do stratified sampling,
#       I might need to take a look at sklearn stratified sampling utilities to implement it myself,
#       because pytorch isn't really helpful in this sense. There is this though:
#           https://github.com/muellerzr/fastai-Experiments-and-tips/blob/master/Class%20Imbalances/Class_Imbalance_Experiments.ipynb
# - Decide which transformations to apply.
#   - Since transforms are applied on the fly when the dataloader requests an
#       image, then theres no need to augment the dataset at this level:
#       data augmentation is implicitly performed as the number of epochs increases,
#       because at each epoch a different transform will be applied to the same image.
#       By setting a seed, everything should still be deterministic while having to
#       do less preprocessing work. Check this discussion for some color on the matter:
#           https://forums.fast.ai/t/data-augmentation-vs-number-of-epochs/8108/3
#   - Another question becomes, how do I define to randomly apply a transform to the image?
#       Is it done automatically, do I have to set any param?

class Coco2017Dataset(VisionDataset):
    """
    Base reference: https://pytorch.org/vision/main/_modules/torchvision/datasets/coco.html
    """

    def __init__(
            self,
            image_dir: str,
            ann_file_path: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        """
        :param image_dir (string): Root directory where images are downloaded to.
        :param ann_file_path (string): Path to json annotation file.
        :param transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        :param target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        :param transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
        """
        super().__init__(image_dir, transforms, transform, target_transform)

        self.coco = COCO(annotation_file=ann_file_path)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, img_id: int) -> torch.Tensor:
        path = self.coco.loadImgs(img_id)[0]["file_name"]
        return tv.io.read_image(path=os.path.join(self.root, path))

    def _load_target(self, img_id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(img_id))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, CustomCocoLabels]:
        img_id = self.ids[index]
        image = self._load_image(img_id)
        target = self._load_target(img_id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)


class RandomResize:
    # TODO(pierluigi): idea to rescale image from 640 to 320 and then take a random 224 crop?
    #  maybe cropping 320 to 224 is too dangerous
    def __call__(self, *args, **kwargs):
        # TODO(pierluigi): maybe parameterize so that resize is done to at most 1.25x the size of the final crop
        #   in order to make it more improbable that the subject is cut out from the pic during the crop
        # resize = tv.transforms.Resize(size=(300, 300))
        # crop = tv.transforms.RandomCrop(224)
        # composed = tv.transforms.Compose([resize, crop])

        composed = t2.RandomResizedCrop(
            size=(224, 224),
            # interval from which crop proportion will be uniformly sampled
            # crop is done first, then rescaling is applied to resize the image
            #   to the target size
            scale=(0.7, 1.0),
        )

        # Performance tips for transforms:
        # - use dtype.uint8 whenever possible
        # - normalization at the end of the pipeline and it requires dtype.float32

        # transform_train = transforms.Compose([
        #     transforms.RandomRotation(degrees=30),
        #     transforms.RandomVerticalFlip(),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(means, stds)
        #     v2.RandomResizedCrop(size=(224, 224), antialias=True),  # Or Resize(antialias=True)
        #     ...
        #     v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        #     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        return composed

# Check out this article to see many different types of image augmentation
# https://webcache.googleusercontent.com/search?q=cache:https://towardsdatascience.com/image-data-augmentation-for-deep-learning-77a87fabd2bf&strip=0&vwsrc=1&referer=medium-parser
# TODO(pierluigi): decide how many transformations to apply to each image based on the number of basic images in the dataset
