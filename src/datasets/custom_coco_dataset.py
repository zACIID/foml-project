import enum
import os.path
from typing import Callable, List, Optional, Tuple, Dict, Iterable

import torch
import torchvision as tv
import torchvision.transforms.v2 as t2
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset


# TODO(pierluigi): ask biagio the following:
#  1. is it okay if requires_grad is called whenever needed and not inside the dataset
#  2. what's the next step? to wrap this with the dataset class of the AdaBoost project?
#  3. Discuss image transformations
#  4. Tell biagio that in the end I stuck with a 4-tuple for __get_item__ because it allows us to type each field,
#       while dict does not


class CocoDataTypes(enum.StrEnum):
    TRAIN_2017 = "train2017"
    VAL_2017 = "val2017"
    TEST_2017 = "test2017"


class Labels(enum.IntEnum):
    PERSON = 0
    OTHER = 1


ItemType = Tuple[int, torch.Tensor, Labels, float]
BatchType = Tuple[Iterable[int], Iterable[torch.Tensor], Iterable[Labels], Iterable[float]]


class CocoDataset(VisionDataset[ItemType]):
    """
    Base reference: https://pytorch.org/vision/main/_modules/torchvision/datasets/coco.html
    """

    def __init__(
            self,
            coco_dir: str,
            dataset_type: CocoDataTypes,
            img_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        :param coco_dir: root directory of the COCO dataset,
            i.e. the one where it is downloaded to
        :param dataset_type: used to retrieve the correct images/annotations
        :param img_transform: A function/transform that takes in a torch.Tensor image
            and returns a transformed version. E.g, ``transforms.Compose(...)``
        """
        super().__init__(
            root=os.path.join(coco_dir, str(dataset_type)),
            transforms=None,
            transform=img_transform,
            target_transform=None
        )

        ann_file_path = os.path.join(coco_dir, "annotations", f"instances_{dataset_type}.json")
        self._coco = COCO(annotation_file=ann_file_path)

        self._img_ids: List[int] = list(sorted(self._coco.imgs.keys()))

        self._labels: torch.Tensor = torch.zeros(len(self._img_ids), dtype=torch.int8)
        """Tensor that contains, at position X, the class of the X-th sample"""

        self._weights: torch.Tensor = torch.zeros(len(self._img_ids), dtype=torch.float32)
        """
        Tensor that contains, at position X, the weight of the X-th sample.
        Mainly used to handle class imbalance and assign more weight to the loss
            of minority samples.
        """

        img_ids_by_class = self._get_img_ids_by_class()

        self._cardinality_by_class: Dict[Labels, int] = {
            c: len(ids) for c, ids in img_ids_by_class.items()
        }

        majority_class_proportion = max(self._cardinality_by_class.values()) / len(self._img_ids)

        # Init classes and weight tensors
        for c in Labels:
            c_img_ids = set(img_ids_by_class[c])

            # Get the indexes of the ids of the images belonging to the current class
            img_id_indexes = map(
                lambda idx_id_pair: idx_id_pair[0],
                filter(
                    lambda idx_id_pair: idx_id_pair[1] in c_img_ids,
                    enumerate(self._img_ids)
                )
            )
            img_id_indexes = torch.tensor(img_id_indexes)

            self._labels[img_id_indexes] = int(c)

            class_proportion = torch.numel(img_id_indexes) / len(self._img_ids)
            adjusted_class_proportion = majority_class_proportion / class_proportion
            self._weights[img_id_indexes] = adjusted_class_proportion

    def _get_img_ids_by_class(self) -> Dict[Labels, List[int]]:
        person_cat_ids = self._coco.getCatIds(supNms=["person"])
        person_img_ids = self._coco.getImgIds(catIds=person_cat_ids)

        non_person_img_ids = set(self._img_ids) - set(person_img_ids)

        return {
            Labels.PERSON: person_img_ids,
            Labels.OTHER: non_person_img_ids
        }

    def __getitem__(self, index: int) -> ItemType:
        img_id = self._img_ids[index]

        path = self._coco.loadImgs(img_id)[0]["file_name"]
        image = tv.io.read_image(path=os.path.join(self.root, path))

        label = self._labels[index]

        sample_weight = self._weights[index]

        if self.transform is not None:
            image = self.transform(image)

        return img_id, image, label, sample_weight

    def __len__(self) -> int:
        """
        NOTE: Implementing this method, along with __get_item__ and __init__ is required
        by subclasses of Dataset
        """
        return len(self._img_ids)

    def get_class_cardinality(self, label: Labels) -> int:
        return self._cardinality_by_class[label]

    @property
    def get_labels(self) -> torch.Tensor:
        """
        Tensor that contains, at position X, the class of the X-th sample
        """

        return self._labels


# TODO(pierluigi): maybe move all of these declarations out of this module and wherever they are actually used
ROOT_COCO_DIR = os.path.join("..", "..", "data")


# Check out this article to see many different types of image augmentation
# https://webcache.googleusercontent.com/search?q=cache:https://towardsdatascience.com/image-data-augmentation-for-deep-learning-77a87fabd2bf&strip=0&vwsrc=1&referer=medium-parser
# TODO(pierluigi): decide how many transformations to apply to each image based on the number of basic images in the dataset

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

COCO_TRAIN_DATASET = CocoDataset(
    coco_dir=ROOT_COCO_DIR,
    dataset_type=CocoDataTypes.TRAIN_2017,
    img_transform=t2.Compose(
        transforms=[
            # TODO
        ]
    )
)
COCO_VAL_DATASET = CocoDataset(
    coco_dir=ROOT_COCO_DIR,
    dataset_type=CocoDataTypes.VAL_2017,
    img_transform=t2.Compose(
        transforms=[
            # TODO
        ]
    )
)

# TODO(pierluigi): but do I even need the test dataset in this format?
COCO_TEST_DATASET = CocoDataset(
    coco_dir=ROOT_COCO_DIR,
    dataset_type=CocoDataTypes.TEST_2017,
    img_transform=t2.Compose(
        transforms=[
            # TODO
        ]
    )
)
# TODO(pierluigi): I do not have the annotations for the test dataset though,
#   where do I download them from?? Need the instances_test2017.json annotation file
