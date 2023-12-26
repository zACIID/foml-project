import enum
import os.path
from typing import Callable, List, Optional, Tuple, Dict

import torch
import torchvision as tv
import torchvision.transforms.v2 as t2
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode

import src.utils.constants as const


class CocoDatasetTypes(enum.StrEnum):
    TRAIN_2017 = "train2017"

    TEST_2017 = "val2017"
    """
    Why test dataset refers to val instead? 
    Because actual test dataset is not labelled, and its intended use is to make 
    predictions that are to be submitted to COCO challenges 
    (hence no annotations published for test dataset)
    """


class Labels(enum.IntEnum):
    PERSON = 0
    OTHER = 1


ItemType = Tuple[int, torch.Tensor, Labels, float]

BatchType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
"""(img_id, img, label, sample_weight)"""


class CocoDataset(VisionDataset):
    """
    Base reference: https://pytorch.org/vision/main/_modules/torchvision/datasets/coco.html
    """

    def __init__(
            self,
            coco_dir: str,
            dataset_type: CocoDatasetTypes,
            img_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            device: torch.device = None
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

        if device is not None:
            self._device = device
        else:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._coco: COCO | None = None
        self.dataset_type = dataset_type

        self._img_ids: List[int] | None = None

        self._labels: torch.Tensor | None = None
        """Tensor that contains, at position X, the class of the X-th sample"""

        self._weights: torch.Tensor | None = None
        """
        Tensor that contains, at position X, the weight of the X-th sample.
        Mainly used to handle class imbalance and assign more weight to the loss
            of minority samples.
        """

        self._cardinality_by_class: Dict[Labels, int] = {}

    def load(self):
        """
        Initialize the dataset by fetching image metadata,
            so that it is ready to provide data samples
        :return:
        """
        ann_file_path = os.path.join(self.root, "..", "annotations", f"instances_{self.dataset_type}.json")
        self._coco = COCO(annotation_file=ann_file_path)

        self._img_ids: List[int] = list(sorted(self._coco.imgs.keys()))
        self._labels: torch.Tensor = torch.zeros(len(self._img_ids), dtype=torch.int8)
        self._weights: torch.Tensor = torch.zeros(len(self._img_ids), dtype=torch.float32)

        img_ids_by_class = self._get_img_ids_by_class()

        self._cardinality_by_class = {
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
            img_id_indexes = torch.tensor(list(img_id_indexes))

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
        assert len(self._img_ids) > 0, ("Either method `load` was not called or something went wrong "
                                        "when fetching the dataset from the specified folder; "
                                        "verify that the COCO dataset is present at the specified location "
                                        f"({self.root})")

        img_id = self._img_ids[index]

        path = self._coco.loadImgs(img_id)[0]["file_name"]
        image = tv.io.read_image(
            path=os.path.join(self.root, path),
            # ensure that everything has 3 channels, because there are some grayscale images
            mode=ImageReadMode.RGB
        )
        image.to(self._device)

        label = self._labels[index]

        sample_weight = self._weights[index]

        if self.transform is not None:
            image = self.transform(image)

        # TODO(pierluigi): specify that now it is the index being returned and not an actual id
        return index, image, label, sample_weight

    def __len__(self) -> int:
        """
        NOTE: Implementing this method, along with __get_item__ and __init__ is required
        by subclasses of Dataset
        """
        return len(self._img_ids)

    def get_class_cardinality(self, label: Labels) -> int:
        return self._cardinality_by_class[label]

    def get_labels(self) -> torch.Tensor:
        """
        Tensor that contains, at position X, the class of the X-th sample
        """

        return self._labels


# Check out this article to see many different types of image augmentation
# https://webcache.googleusercontent.com/search?q=cache:https://towardsdatascience.com/image-data-augmentation-for-deep-learning-77a87fabd2bf&strip=0&vwsrc=1&referer=medium-parser
#
# Performance tips for transforms:
# - use dtype.uint8 whenever possible
# - normalization at the end of the pipeline and it requires dtype.float32
#
# As of now, no data augmentation, just resizing images to make them usable by the models,
#    and then normalizing them

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COCO_TRAIN_DATASET = CocoDataset(
    coco_dir=const.ROOT_COCO_DIR,
    dataset_type=CocoDatasetTypes.TRAIN_2017,
    img_transform=t2.Compose(
        transforms=[
            t2.Resize(size=const.INPUT_IMAGE_SIZE, antialias=True),
            t2.ToDtype(torch.float32, scale=True),
            t2.Normalize(mean=const.COCO_IMAGE_MEANS, std=const.COCO_IMAGE_STDS),
        ]
    ),
    device=_device
)

COCO_TEST_DATASET = CocoDataset(
    coco_dir=const.ROOT_COCO_DIR,
    dataset_type=CocoDatasetTypes.TEST_2017,
    img_transform=t2.Compose(
        transforms=[
            t2.Resize(size=const.INPUT_IMAGE_SIZE, antialias=True),
            t2.ToDtype(torch.float32, scale=True),
            t2.Normalize(mean=const.COCO_IMAGE_MEANS, std=const.COCO_IMAGE_STDS),
        ]
    ),
    device=_device
)
