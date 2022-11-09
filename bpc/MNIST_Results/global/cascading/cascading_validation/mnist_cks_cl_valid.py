#%%
"""
file: mnist_cks_cl_valid.py

Testing of negative transfer effects of original problem, and BPC extracted features on a training and validation subset. 

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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools

Julia()
from julia import Main as jl

cv_classification = jl.include(r"..\..\..\..\LooCV\ReturnCVClassified.jl")

cv_prediction = jl.include(r"..\..\..\..\LooCV\ReturnCVPredicted.jl")

from joblib import Parallel, delayed

#%%
# Load Data
data = sio.loadmat(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\data\mnistdata.mat"
)
x_train = data["training"]
y_train = np.concatenate(data["Gtrain"])


#%%
x_t, x_v, y_t, y_v = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)


#%%
L = np.arange(1, 11)
uniq_combinations = list(itertools.combinations(L, 5))

#%%
Xtr_combs = [x_t[np.isin(y_t, combs)] for combs in uniq_combinations]
Ytr_combs = [y_t[np.isin(y_t, combs)] for combs in uniq_combinations]

#%%
xv_combs = [x_v[np.isin(y_v, combs)] for combs in uniq_combinations]
yv_combs = [y_v[np.isin(y_v, combs)] for combs in uniq_combinations]
#%%
def train_individuals(x, y, combination):
    map_dict = {x: y + 1 for y, x in enumerate(combination)}
    reverse_map_dict = {y + 1: x for y, x in enumerate(combination)}

    y = np.vectorize(map_dict.get)(y)

    oc_y = np.identity(5)[y - 1]
    _, _, b_ind, _, _ = cv_classification(
        x.reshape(x.shape[0], -1), oc_y, None, 5, False
    )

    return np.asarray(b_ind)


#%%
bs_individuals = [
    train_individuals(x, y, combination)
    for x, y, combination in zip(Xtr_combs, Ytr_combs, uniq_combinations)
]

#%%
def predict_individuals(x, b_lamb, combination):
    reverse_map_dict = {y + 1: x for y, x in enumerate(combination)}

    preds, _ = cv_prediction(x.reshape(x.shape[0], -1), b_lamb, 5)

    return np.vectorize(reverse_map_dict.get)(np.concatenate(preds))


#%%
predictions_individuals = [
    predict_individuals(x, b, combination)
    for x, b, combination in zip(xv_combs, bs_individuals, uniq_combinations)
]
#%%
accs_individuals = [
    accuracy_score(y_v_comb, y_hat)
    for y_v_comb, y_hat in zip(yv_combs, predictions_individuals)
]

#%%
oc_y = np.identity(10)[y_t - 1]
_, _, b_lambda_full, _, _ = cv_classification(
    x_t.reshape(x_t.shape[0], -1), oc_y, None, 10, False
)


#%%
b_lambda_full = np.asarray(b_lambda_full)
#%%
def predict_full(x, b_lambd):

    pred, _ = cv_prediction(x.reshape(x.shape[0], -1), b_lambd, 10)

    return np.concatenate(pred)


#%%
Preds_fulls = [predict_full(x, b_lambda_full) for x in xv_combs]


#%%
accuracy_fulls = [
    accuracy_score(y_v_comb, y_hat)
    for y_v_comb, y_hat in zip(yv_combs, Preds_fulls)
]
#%%
from scipy.signal import savgol_filter


def smooth_data_savgol_1(arr, span=1):
    return savgol_filter(arr, span * 2 + 1, 1)


#%%
np.save("accs_individuals", accs_individuals)
np.save("accuracy_fulls", accuracy_fulls)

#%%
import seaborn as sns

sns.set_style("whitegrid")
figure = plt.figure(figsize=(8, 4))
plt.plot(accuracy_fulls, label="Global Model")
plt.plot((accs_individuals), label="Local Models")
plt.xlabel("Combinations")
plt.ylabel("Validation Accuracy")
plt.title("Global vs Local Models - MNIST")
plt.legend()
plt.savefig("global_vs_local_mnist.pdf", dpi=800, bbox_inches="tight")
plt.show()


#%%
num_imgs = 20

images_to_extract = np.concatenate(
    ks_variant(
        x_t.reshape(x_t.shape[0], -1),
        y_t,
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
#%%
filters = filters.reshape(filters.shape[0] * filters.shape[1], 14, 14)

#%%
np.save("filters_full.npy", filters)
# %%
x_convolved = convolve_filts(x_t, None, filters, bsize=64)

# %%
x_convolved_orig = np.hstack((x_t.reshape(x_t.shape[0], -1), x_convolved))

# %%
oc_y_train = np.identity(10)[y_t - 1]
# %%
_, _, b_bpc_full, _, _ = cv_classification(
    x_convolved_orig, oc_y_train, None, 10, False
)
#%%
b_bpc_full = np.asarray(b_bpc_full)

#%%
np.save("b_bpc_first.npy", b_bpc_full)

#%%
preds_full = []

for x in xv_combs:
    xv_convolved = convolve_filts(x, None, filters, bsize=64)
    xv_stacked = np.hstack((x.reshape(x.shape[0], -1), xv_convolved))

    preds, _ = cv_prediction(xv_stacked, b_bpc_full, 10)

    preds_full.append(np.concatenate(preds))

#%%
acc_bpc_full = [
    accuracy_score(y_v_comb, y_hat)
    for y_v_comb, y_hat in zip(yv_combs, preds_full)
]

#%%
preds_bpc_inds = []

for idin, (xt, yt, vt, combination) in enumerate(
    zip(Xtr_combs, Ytr_combs, xv_combs, uniq_combinations)
):
    print(idin)
    map_dict = {x: y + 1 for y, x in enumerate(combination)}
    reverse_map_dict = {y + 1: x for y, x in enumerate(combination)}
    curr_y_train = np.vectorize(map_dict.get)(yt)

    images_to_extract = np.concatenate(
        ks_variant(
            xt.reshape(xt.shape[0], -1), curr_y_train, 5, 40, within_group=True
        )
    )
    images_to_extract = images_to_extract.reshape(
        images_to_extract.shape[0], 28, 28
    )

    filters_curr = extract_filters_np(
        images_to_extract, kernel_size=(14, 14), center=True
    )

    filters_curr = filters_curr.reshape(
        filters_curr.shape[0] * filters_curr.shape[1], 14, 14
    )

    xt_convolved = convolve_filts(xt, None, filters_curr, bsize=64)

    xt_stacked = np.hstack((xt.reshape(xt.shape[0], -1), xt_convolved))

    oc_y_train_curr = np.identity(5)[curr_y_train - 1]

    _, _, b_bpc_curr, _, _ = cv_classification(
        xt_stacked, oc_y_train_curr, None, 5, False
    )

    b_bpc_curr = np.asarray(b_bpc_curr)

    xv_convolved = convolve_filts(vt, None, filters_curr, bsize=64)
    xv_stacked = np.hstack((vt.reshape(vt.shape[0], -1), xv_convolved))

    preds, _ = cv_prediction(xv_stacked, b_bpc_curr, 5)

    preds_bpc_inds.append(
        np.vectorize(reverse_map_dict.get)(np.concatenate(preds))
    )
#%%
acc_bpc_inds = [
    accuracy_score(y_v_comb, y_hat)
    for y_v_comb, y_hat in zip(yv_combs, preds_bpc_inds)
]
# %%
import seaborn as sns

sns.set_style("whitegrid")
figure = plt.figure(figsize=(8, 4))
plt.plot(acc_bpc_full, label="Global Model")
plt.plot(acc_bpc_inds, label="Local Models")
plt.xlabel("Combinations")
plt.ylabel("Validation Accuracy")
plt.title("BPC-5000 - Global vs Local Models - MNIST")
plt.legend()
plt.savefig("global_vs_local_mnist_BPC.pdf", dpi=800, bbox_inches="tight")
plt.show()
# %%

np.save("acc_bpc_full.npy", acc_bpc_full)
np.save("acc_bpc_inds.npy", acc_bpc_inds)
# %%
