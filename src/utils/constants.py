import os
from typing import Tuple


RND_SEED = 777

# NOTE(pierluigi): 3 jumps back from file absolute path: file->utils, utils->src, src->root
ROOT_COCO_DIR = os.path.abspath(os.path.join(__file__, "..", "..", "..", "data"))
"""Directory where COCO dataset is stored/downloaded"""

INPUT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
"""Input size of models in this project"""

COCO_IMAGE_MEANS: Tuple[float, float, float] = (119.8632, 113.9443, 103.9443)
"""Dataset, channel-wise means for MS COCO2017 dataset (training) images"""

COCO_IMAGE_STDS: Tuple[float, float, float] = (59.4222, 58.1447, 59.0335)
"""Dataset, channel-wise standard deviations for MS COCO2017 dataset (training) images"""
