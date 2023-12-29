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
# %load_ext autoreload

import torch
import torch.utils.data as data
import src.datasets.custom_coco_dataset as coco
import src.ensemble_training.ada_boost as ada
from src.classifiers import alex_net as ax
from src.utils.constants import RND_SEED

# %%
torch.cuda.is_available()

# %%
torch.version.cuda

# %%
torch.manual_seed(RND_SEED)

# %%
dataset = coco.COCO_TEST_DATASET
dataset.load()

# %% [markdown]
# ## AdaBoost Tests

# %%
ada_boost = ada.AdaBoost(dataset=dataset, n_eras=1, n_classes=2)

# %% [raw]
# strong_learner = ada_boost.start(verbose=4)

# %% [raw]
# subset = data.Subset(dataset=dataset, indices=[x for x in range(48)])
# for batch in data.DataLoader(subset, 4, shuffle=True):
#     batch: coco.BatchType
#     ids, imgs, labels, weights = batch
#     preds = strong_learner.predict_image(imgs)
#     
#     print(f"Predictions:\n{preds}")
#     print(f"Actual:\n{labels}")

# %% [markdown]
# ## AlexNet Tests

# %%
y_true = torch.tensor([[0, 1.0], [1.0, 0]])
y_pred = torch.tensor([[0.2, -0.3], [0.1, -0.4]])

cross_entropy: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss(reduction='none')
x = cross_entropy(y_pred, y_true)
x = x.mean()
print(x)

# %% is_executing=true
alex_net = ax.AlexNet(
    act_fun=torch.nn.SiLU
)
alex_net.fit(
    dataset=dataset,
    verbose=5,
    learning_rate=0.0005,
    momentum=0.9,
    batch_size=128,
    epochs=100
)

# %% [markdown]
# ## TODO Presentazione
#
# - dire che abbiamo scelto AlexNet come ispriazione, non l'abbiamp proprio seguita perche dataset diverso, image transformation diversi (loro fanno anche flip, noi invece semplicemente normalizziamo perche abbiamo un numero sufficiente di immagini) e non usiamo normalizzazione tra layer
# - dire che per adaboost abbiamo preso squeezenet come ispirazione ma l'abbiamo castrata, tenendo solo i layer meno profondi di ogni "blocco" (perche intuizione e che si puo andare piu in profodnita, ovvero aumentare numero di filtri per layer, mano a mano che la il numero di layer aument
# - paralre del dataset utilizzato e come e stato rielaborato in brevissimo
# - dire che per adaboost + squeezenet volevamo comparare i due modelli idealmente con un numero di parametri simile, quindi numero weak learner scelto in modo tale che numero di aprameteri simile ad AlexNet
# - mostrare i grafici
#

# %% [markdown]
# ## TODO grafici
#
# ### AlexNet
#
# - training (+ validation) loss di SGD con i parameteri di AlexNet
# - training (+ validation) loss di Adam con i nostri parametri (lr=0.0005, weight_decay=5e-3)
#
# Previsioni:
# - training accuracy
# - test accuracy
# - training confusion matrix
# - testing confusion matrix
#
# ### Adaboost
#
# Come AlexNet, in aggiunta:
# - vedere comportamento loss in base a numero di weak learner
#

# %% [markdown]
# ### Altri TODO
#
# - passare in qualche modo un parametro a cocodataset che mi permetta di sceglierne solo un sottoinsieme??? oppure passarlo al metodo train -> questo mi serve sia per velocizzare training e testing che per fare gli split per la validation
#     - per scegliere indici randomici posso usare train_test_split di sklearn passando un array di indici (setto RND SEED)
#     - poi uso subset passandoci la lista di indici per definire dataset di train e test sul dataloader
# - metodo train esterno al modello???
#     - questo metodo ritorna una struct di dati utili tipo loss per epoca e altri parametri 
# - passare optimizer e criterion direttamente come parametri di train
# - eseguo metodo esterno train() su dataset di train e validation
# - usiamo un decimo del dataset (12k, auspicabilmente bilanciate) per train + validation che altrimenti 100epoche ci si mette un giorno

# %% [markdown]
# ## Validation

# %%
import src.visualization.classification as vis

for batch in dataset:
    ids, images, labels, weights = batch
    y_pred = model.predict()
    vis.confusion_matrix(
        y_true=labels,
        y_pred=y_pred,
        figsize=(12,12)
    )
