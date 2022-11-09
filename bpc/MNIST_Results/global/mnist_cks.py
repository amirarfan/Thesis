#%%
"""

File: mnist_cks.py

Performing Clustered  Kennard-Stone Selection of images on MNIST data, with different amount of images, calculating training set and CV accuracy

"""
import os
import sys

from sklearn.metrics import accuracy_score

sys.path.append(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\utils"
)
sys.path.append(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\LooCV"
)
import numpy as np
import scipy.io as sio
from cks import ks_variant
from convolve_filts import convolve_filts
from extract_filters import extract_filters_np
from julia.api import Julia

Julia()
from julia import Main as jl

cv_classification = jl.include(r"..\..\LooCV\ReturnCVClassified.jl")
#%%
# Load Data
data = sio.loadmat(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\data\mnistdata.mat"
)
x_train = data["training"]
y_train = np.concatenate(data["Gtrain"])
# %%

num_imgs = [10]

for num_imgs in num_imgs:
    print("Computing for " + str(num_imgs))
    images_to_extract = np.concatenate(
        ks_variant(
            x_train.reshape(x_train.shape[0], -1),
            y_train,
            5,
            num_imgs,
            within_group=True,
        )
    )
    #%%
    images_to_extract = images_to_extract.reshape(
        images_to_extract.shape[0], 28, 28
    )
    # %%
    filters = extract_filters_np(
        images_to_extract, kernel_size=(14, 14), center=True
    )
    # %%
    filters = filters.reshape(filters.shape[0] * filters.shape[1], 14, 14)
    print(filters.shape[0])
    # %%
    x_convolved = convolve_filts(x_train, None, filters, bsize=24)

    print("Convolved")
    # %%
    x_convolved_orig = np.hstack(
        (x_train.reshape(x_train.shape[0], -1), x_convolved)
    )

    # %%
    oc_y_train = np.identity(10)[y_train - 1]
    # %%
    Ghat, GhatCV, b, _, _ = cv_classification(
        x_convolved_orig, oc_y_train, None, 10, False
    )
    # %%
    Ghat = np.concatenate(Ghat)
    GhatCV = np.concatenate(GhatCV)

    #%%
    # Save accuracy on GhatCV to text file
    print("Saving accuracy to text file")
    with open(
        os.path.join(os.path.dirname(__file__), "cv_accuracy_cks.txt"), "a"
    ) as filename:
        filename.write(
            f"CV_{x_convolved.shape[1]}:"
            + str(accuracy_score(y_train, GhatCV))
        )
        filename.write(
            f"\nTrain_{x_convolved.shape[1]}:"
            + str(accuracy_score(y_train, Ghat))
        )
        filename.write("\n")
    # %%
