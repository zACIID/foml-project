import os
from typing import Tuple


RND_SEED = 777

# NOTE(pierluigi): 3 jumps back from file absolute path: file->utils, utils->src, src->root
ROOT_COCO_DIR = os.path.abspath(os.path.join(__file__, "..", "..", "..", "data"))
"""Directory where COCO dataset is stored/downloaded"""

ALEX_NET_RESIZE_SIZE: Tuple[int, int] = (256, 256)
"""AlexNet first resizes images to this dimension, and then takes a center crop of lower size (224x224)"""

INPUT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
"""Input size of models in this project"""

COCO_IMAGE_MEANS: Tuple[float, float, float] = (119.8632, 113.9443, 103.9443)
"""Dataset, channel-wise means for MS COCO2017 dataset (training) images"""

COCO_IMAGE_STDS: Tuple[float, float, float] = (59.4222, 58.1447, 59.0335)
"""Dataset, channel-wise standard deviations for MS COCO2017 dataset (training) images"""

IMAGE_NET_IMAGE_MEANS: Tuple[float, float, float] = (123.6750, 116.2800, 103.5300)
"""Dataset, channel-wise means for ImageNet dataset (training) images"""

IMAGE_NET_IMAGE_STDS: Tuple[float, float, float] = (58.3950, 57.200, 57.3750)
"""Dataset, channel-wise standard deviations for ImageNet dataset (training) images"""


