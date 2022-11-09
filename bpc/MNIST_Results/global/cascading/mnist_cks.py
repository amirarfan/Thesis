# %%
"""
file: mnist_cks.py

Testing out negative transfer effects of BPC extracted features problem, on the test-data, using Top-k minimization and majority voting
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

cv_classification = jl.include(r"..\..\..\LooCV\ReturnCVClassified.jl")

cv_prediction = jl.include(r"..\..\..\LooCV\ReturnCVPredicted.jl")
#%%
# Load Data
data = sio.loadmat(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\data\mnistdata.mat"
)
x_train = data["training"]
y_train = np.concatenate(data["Gtrain"])

#%%
x_test = data["testing"]
y_test = np.concatenate(data["Gtest"])
#%%

num_imgs = 20

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
#%%
filters = filters.reshape(filters.shape[0] * filters.shape[1], 14, 14)

#%%
np.save("filters_first.npy", filters)
# %%
x_convolved = convolve_filts(x_train, None, filters, bsize=64)

#%%
np.save("x_convolved_first.npy", x_convolved)


# %%
x_convolved_orig = np.hstack(
    (x_train.reshape(x_train.shape[0], -1), x_convolved)
)

# %%
oc_y_train = np.identity(10)[y_train - 1]
# %%
Ghat, GhatCV, b, Yhat, YhatCV = cv_classification(
    x_convolved_orig, oc_y_train, None, 10, False
)

#%%
np.save("Ghat_first.npy", Ghat)
np.save("GhatCV_first.npy", GhatCV)
np.save("b_first.npy", b)
np.save("Yhat_first.npy", Yhat)
np.save("YhatCV_first.npy", YhatCV)

#%%
top_5_predictions_cv = (np.argsort(YhatCV, axis=1) + 1)[:, ::-1][:, :5]

# %%
# Unique Top_5 predictions
unique_top_5 = np.unique(np.sort(top_5_predictions_cv, axis=1), axis=0)
#%%
np.save("unique_top_5_first.npy", unique_top_5)
#%%

combination_problems = {}
for i in range(len(top_5_predictions_cv)):
    for unique_predictions in unique_top_5:
        if np.array_equal(
            np.sort(top_5_predictions_cv[i, :]), unique_predictions
        ):
            combination_problems[i] = unique_predictions


# Find y_train values where it is equal to each combination in unique_top_5_misclassified_problems
# %%
x_train_uniques = {}
y_train_uniques = {}
for idx, combination in enumerate(unique_top_5):
    indexes = np.where(np.in1d(y_train, combination))[0]
    x_train_uniques[idx] = x_train[indexes]
    y_train_uniques[idx] = y_train[indexes]

# %%
# Find idxes, of samples where the combinations are mutual
common_combinations = {}
for idx, combination in enumerate(unique_top_5):
    common_combination = []
    for key, values in combination_problems.items():
        if np.array_equal(combination, values):
            common_combination.append(key)
    common_combinations[idx] = common_combination

# %%
b_lambdas = []
filters_per_combination = []
for idx, combination in enumerate(unique_top_5):
    print(f"{idx} / {len(unique_top_5)}")
    map_dict = {x: y + 1 for y, x in enumerate(combination)}
    curr_x_train = x_train_uniques[idx]
    curr_y_train = y_train_uniques[idx]
    curr_y_train = np.vectorize(map_dict.get)(curr_y_train)

    images_to_extract = np.concatenate(
        ks_variant(
            curr_x_train.reshape(curr_x_train.shape[0], -1),
            curr_y_train,
            5,
            40,
            within_group=True,
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
    filters_per_combination.append(filters_curr)
    x_convolved = convolve_filts(curr_x_train, None, filters_curr, bsize=24)
    x_convolved_orig_curr = np.hstack(
        (curr_x_train.reshape(curr_x_train.shape[0], -1), x_convolved)
    )
    oc_y_train = np.identity(5)[curr_y_train - 1]
    _, _, b_lambda, _, _ = cv_classification(
        x_convolved_orig_curr, oc_y_train, None, 5, False
    )

    b_lambdas.append(np.asarray(b_lambda))

# %%
Y_hat = np.zeros(60000)
for idx, combination in enumerate(unique_top_5):
    current_idxes = common_combinations[idx]
    x_predict_current = x_train[current_idxes]
    curr_filter = filters_per_combination[idx]
    reverse_map_dict = {y + 1: x for y, x in enumerate(combination)}

    convolved = convolve_filts(x_predict_current, None, curr_filter, bsize=64)
    x_convolved_predict = np.hstack(
        (x_predict_current.reshape(-1, 784), convolved)
    )

    predictions, _ = cv_prediction(x_convolved_predict, b_lambdas[idx], 5)

    predictions_converted = np.vectorize(reverse_map_dict.get)(
        np.concatenate(predictions)
    )

    for pred, indic in zip(predictions_converted, current_idxes):
        Y_hat[indic] = pred

# %%
np.save("Y_hat_second.npy", Y_hat)
# %%
np.save("b_lambdas_per_combination.npy", b_lambdas)
# %%
np.save("filters_per_combination.npy", filters_per_combination)

#%%
x_test_convolved = convolve_filts(x_test, None, filters, bsize=128)
# %%
x_test_orig = np.hstack(
    (x_test.reshape(x_test.shape[0], -1), x_test_convolved)
)
# %%
x_test_predictions, Y_test_preds = cv_prediction(
    x_test_orig, np.asarray(b), 10
)

# %%
Y_test_preds_top_5 = np.argsort(Y_test_preds, axis=1)[:, ::-1][:, :5] + 1
# %%
combination_problems_test = {}
for i in range(len(Y_test_preds_top_5)):
    for unique_predictions in unique_top_5:
        if np.array_equal(
            np.sort(Y_test_preds_top_5[i, :]), unique_predictions
        ):
            combination_problems_test[i] = unique_predictions
# %%
# Find idxes, of samples where the combinations are mutual
common_combinations_test = {}
for idx, combination in enumerate(unique_top_5):
    common_combination = []
    for key, values in combination_problems_test.items():
        if np.array_equal(combination, values):
            common_combination.append(key)
    common_combinations_test[idx] = common_combination
# %%
Y_test = np.zeros(10000)
for idx, combination in enumerate(unique_top_5):
    current_idxes = common_combinations_test[idx]
    x_predict_current = x_test[current_idxes]
    curr_filter = filters_per_combination[idx]
    reverse_map_dict = {y + 1: x for y, x in enumerate(combination)}

    convolved = convolve_filts(x_predict_current, None, curr_filter, bsize=64)
    x_convolved_predict = np.hstack(
        (x_predict_current.reshape(-1, 784), convolved)
    )

    predictions, _ = cv_prediction(x_convolved_predict, b_lambdas[idx], 5)

    predictions_converted = np.vectorize(reverse_map_dict.get)(
        np.concatenate(predictions)
    )

    for pred, indic in zip(predictions_converted, current_idxes):
        Y_test[indic] = pred

# %%
Y_test_majority_vote = np.zeros((252, 10000))
for idx, combination in enumerate(unique_top_5):
    print(f"{idx} / {len(unique_top_5)}")
    curr_filter = filters_per_combination[idx]
    reverse_map_dict = {y + 1: x for y, x in enumerate(combination)}
    convolved = convolve_filts(x_test, None, curr_filter, bsize=64)

    x_convolved_predict = np.hstack((x_test.reshape(-1, 784), convolved))

    predictions, _ = cv_prediction(x_convolved_predict, b_lambdas[idx], 5)

    predictions_converted = np.vectorize(reverse_map_dict.get)(
        np.concatenate(predictions)
    )
    Y_test_majority_vote[idx, :] = predictions_converted

# %%
def majority_vote(Y_test_majority_vote):
    Y_test_majority_vote = Y_test_majority_vote.astype(int)
    Y_test_majority_vote = np.transpose(Y_test_majority_vote)
    Y_test_majority_vote = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), 1, Y_test_majority_vote
    )
    return Y_test_majority_vote


Y_test_majority_argmaxed = majority_vote(Y_test_majority_vote)
# %%
