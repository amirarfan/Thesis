#%%
precompute = True
import sys
import os

sys.path.append(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\utils"
)
sys.path.append(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\LooCV"
)
import numpy as np
from crs import crs
from extract_filters import extract_filters_np
import scipy.io as sio
from convolve_filts import convolve_filts
from load_fmnist import load_fmnist
from sklearn.metrics import accuracy_score
from julia.api import Julia

Julia()
from julia import Main as jl

cv_classification = jl.include(r"..\LooCV\ReturnCVClassified.jl")

threshold = 84
#%%
# Load Data
datapath = r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\data\fmnist"
# Load Data
x_train, y_train = load_fmnist(datapath, kind="train")
y_train = y_train.copy()
y_train += 1
x_train = x_train.reshape(x_train.shape[0], 28, 28)
# %%
num_imgs = [117]


for num_img in num_imgs:
    images_for_filters = crs(x_train, y_train, 5, num_img)
    images_to_extract = np.zeros((len(images_for_filters), 28, 28))
    # %%
    for image in range(len(images_for_filters)):
        images_to_extract[image] = images_for_filters[image].reshape(28, 28)
    # %%
    filters = extract_filters_np(
        images_to_extract, kernel_size=(14, 14), threshold=threshold
    )
    #%%
    filters = filters.reshape(filters.shape[0] * filters.shape[1], 14, 14)
    print(filters.shape[0])
    # %%
    x_convolved = convolve_filts(x_train, None, filters, bsize=64)
    print("convolved")

    if precompute:
        np.save("x_convolved_crs_{}".format(filters.shape[0]), x_convolved)
        sys.exit()
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

    #%%
    # Save accuracy on GhatCV to text file
    with open(
        os.path.join(os.path.dirname(__file__), "cv_accuracy_crs.txt"), "a"
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
