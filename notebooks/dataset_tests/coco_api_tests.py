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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Download COCO 2017 Dataset - Uncomment if Necessary

# %%
import torch
# %matplotlib inline

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# %% [markdown]
# ! cd ../../data && /bin/bash coco_download.sh

# %% [markdown]
# ## Data Exploration - Example

# %%
dataDir='../../data'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# %%
# initialize COCO api for instance annotations
coco=COCO(annFile)

# %%
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# %%
# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
imgIds = coco.getImgIds(catIds=catIds)
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

# %%
# load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
I = io.imread(img['coco_url'])
print(type(I))

plt.axis('off')
plt.imshow(I)
plt.show()

# %%
# load and display instance annotations
plt.imshow(I)
plt.axis('off')

annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns, draw_bbox=True)

# %%
# initialize COCO api for person keypoints annotations
annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
coco_kps=COCO(annFile)

# %%
# load and display keypoints annotations
plt.imshow(I); plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)
coco_kps.showAnns(anns)

# %%
# initialize COCO api for caption annotations
annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps=COCO(annFile)

# %%
# load and display caption annotations
annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
plt.imshow(I); plt.axis('off'); plt.show()

# %%
anns

# %% [markdown]
# ## Data Exploration - Custom
#
# The things that I need are the following:
# - extract images with category `person`
# - also keep all images with other categories as counter examples: do not want to have only e.g. animals as counter examples. **Do a random, (possibly stratified) sampling of those images so that the number of counter examples isn't so big to dwarf person images**
# - *I think* I need the path of the images that I want to include in the dataset loader (ImageDataset from pytorch) and then let it handle the stuff.
#     - I think i could save the image ids into a json or some kind of annotation file to speed things up
#         - In such a file I save the path of the image directories from the training, val and test set, as well as the ids that I want to keep
#         - define a method in the loader class that regenerates this file

# %%
## Download COCO 2017 Dataset - Uncomment if Necessary
# %matplotlib inline
import os

from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# %%
DATA_DIR = os.path.join("..", "..", "data")
DATA_TYPE = 'val2017'  # use val set for sketches because it is smaller and faster to load
ANNFILE = os.path.join(DATA_DIR, "annotations", f"instances_{DATA_TYPE}.json")
# ANNFILE = os.path.join(DATA_DIR, "annotations", "image_info_test2017.json")

# initialize COCO api for instance annotations
coco = COCO(ANNFILE)

# %%
person_cat_ids = coco.getCatIds(supNms=["person"])

# %%
person_cat_ids

# %%
person_img_ids = set(coco.getImgIds(catIds=person_cat_ids))
print(len(person_img_ids))

# %%
coco.cats

# %%
non_person_cat_ids = list(
    filter(
        lambda x: x not in person_cat_ids,
        map(lambda x: x[1]["id"], coco.cats.items())
    )
)
print(len(non_person_cat_ids))

# %%
non_person_img_ids = list(
    filter(
        lambda img_id: img_id not in person_img_ids,
        {
            img_id for ids in map(
                lambda c_ids: coco.getImgIds(catIds=c_ids),
                map(lambda c_id: [c_id], non_person_cat_ids)
            ) for img_id in ids
        },
    )
)

# %%
len(non_person_img_ids)

# %%
len(coco.imgs)

# %%
list(coco.imgs.values())[:10]

# %% [markdown]
# ### Plotting Some Image Samples

# %% [markdown]
# #### Non Persons

# %%
import random

for i in range(1):
    sample_ids = random.Random().sample(population=non_person_img_ids, k=5)
    imgs = coco.loadImgs(ids=sample_ids)
    for img in imgs:
        I = io.imread(img['coco_url'])
        print(type(I))

        plt.axis('off')
        plt.imshow(I)
        plt.show()


# %% [markdown]
# #### Persons

# %%
import random

person_img_ids_list = list(person_img_ids)
for i in range(1):
    sample_ids = random.Random().sample(population=person_img_ids_list, k=5)
    imgs = coco.loadImgs(ids=sample_ids)
    for img in imgs:
        I = io.imread(img['coco_url'])
        print(type(I))

        plt.axis('off')
        plt.imshow(I)
        plt.show()


# %%

# %% [markdown]
# ## Other Stuf

# %%
# %load_ext autoreload

import torch
import torch.utils.data as thd
import torchvision.transforms.v2 as t2
import tqdm

import src.datasets.custom_coco_dataset as cd
import src.utils.constants as const

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

dataset = cd.CocoDataset(
    coco_dir=const.ROOT_COCO_DIR,
    dataset_type=cd.CocoDatasetTypes.TRAIN_2017,
    img_transform=t2.Compose([
        t2.ToDtype(dtype=torch.float32),
        t2.Resize(size=const.INPUT_IMAGE_SIZE, antialias=True),
    ])
)

loader = thd.DataLoader(dataset=dataset, batch_size=64, shuffle=True)

# for batch in loader:
#     batch: cd.BatchType
#     ids, imgs, labels, weights = batch
#     print(type(imgs))
#     print(imgs.shape)

# Reference:
# https://stackoverflow.com/questions/60101240/finding-mean-and-standard-deviation-across-image-channels-pytorch

n_images = 0
mean = 0.
std = 0.
try:
    for batch in tqdm.tqdm(loader):
        batch: cd.BatchType
        ids, imgs, labels, weights = batch
        
        if imgs.size(0) == 1:
            print(imgs.shape)
        
        # Rearrange batch to be the shape of [BatchSize, Channels, W * H], 
        # from [BatchSize, Channels, W, H].
        # -1 is used to let pytorch infer from other dims. 
        # Here it is the equivalent of imgs.size(2)*imgs.size(3)
        imgs: torch.Tensor = imgs.view(imgs.size(0), imgs.size(1), -1)  
        
        # Update total number of images
        n_images += imgs.size(0)
        
        # Compute mean and std here, on the pixel dimension, 
        #   and then sum every image in the batch
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)
except Exception as ex:
    print(ex)

# Final step
mean /= n_images
std /= n_images

print(mean)
print(std)

# %%

# %%
