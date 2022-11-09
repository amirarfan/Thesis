#%%
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
from cks_hu import ks_variant
from convolve_filts import convolve_filts
from extract_filters import extract_filters_np
from hu_moments import computescale_humoments
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
x_test = data["testing"]
y_train = np.concatenate(data["Gtrain"])
y_test = np.concatenate(data["Gtest"])
# %%

x_train_moments = computescale_humoments(x_train)
num_imgs = [2, 4, 20, 40, 80, 100, 120]

for num_imgs in num_imgs:
    print("Computing for " + str(num_imgs))

    images_for_filters = ks_variant(
        x_train,
        x_train_moments,
        y_train,
        5,
        num_imgs,
    )
    #%%
    images_to_extract = np.concatenate(images_for_filters)
    images_to_extract = images_to_extract[:, :784].reshape(
        images_to_extract.shape[0], 28, 28
    )
    # %%
    filters = extract_filters_np(
        images_to_extract, kernel_size=(14, 14), center=True
    )
    #%%
    filters = filters.reshape(filters.shape[0] * filters.shape[1], 14, 14)
    print(filters.shape[0])
    # %%
    x_convolved = convolve_filts(x_train, None, filters, bsize=64)

    print("Convolved")
    # %%
    x_convolved_orig = np.hstack(
        (x_train.reshape(x_train.shape[0], -1), x_convolved)
    )

    # %%
    oc_y_train = np.identity(10)[y_train - 1]
    # %%
    Ghat, GhatCV, _, _, _ = cv_classification(
        x_convolved_orig, oc_y_train, None, 10, False
    )
    # %%
    Ghat = np.concatenate(Ghat)
    GhatCV = np.concatenate(GhatCV)
    # %%

    #%%
    # Save accuracy on GhatCV to text file
    print("Saving accuracy to text file")
    with open(
        os.path.join(os.path.dirname(__file__), "cv_accuracy_ckshu.txt"), "a"
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
