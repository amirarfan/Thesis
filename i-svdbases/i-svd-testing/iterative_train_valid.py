# %%
import sys

sys.path.append(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\i-svdbases"
)
import numpy as np
import scipy.io as io

n_components = 25
from isvbase import SVDDiff

# %%

X_train = np.load("../selectoptimalbasesfiles/Xtrain.npy")
X_test = np.load("../selectoptimalbasesfiles/Xtest.npy")
y_train = np.load("../selectoptimalbasesfiles/y_train.npy")
y_test = np.load("../selectoptimalbasesfiles/y_test.npy")

# %%
residuals = np.load(
    "../selectoptimalbasesfiles/train_diffs_per_component.npy"
)[5]
# %%
my_svddiff = SVDDiff(X_train, y_train, residuals)
# %%
my_svddiff.compute_accuracy()
# %%
my_svddiff.find_mc_samples()
# %%
my_svddiff.update_labels_with_mc()
my_svddiff.compute_U_per_digit(my_svddiff.updated_labels[-1])
# %%
my_svddiff.compute_res(25)
# %%
my_svddiff.compute_accuracy()
# %%
my_svddiff.find_mc_samples()
#%%
my_svddiff.update_labels_with_mc()

# %%
my_svddiff.compute_U_per_digit(my_svddiff.updated_labels[-1])

# %%
my_svddiff.compute_res(n_components)

#%%
my_svddiff.compute_accuracy()

# %%
import joblib

# joblib.dump(my_svddiff, "my_svddiff.pkl")

#%%
my_svddiff = joblib.load("my_svddiff.pkl")

# %%
x_test_res_3, x_test_pred_3 = my_svddiff.predict(
    X_test.astype(np.float32), 25, 2
)
# %%
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, x_test_pred_3))
# %%
x_test_res_2, x_test_ped_2 = my_svddiff.predict(
    X_test.astype(np.float32), 25, 1
)
# %%
my_svddiff.find_mc_samples()
# %%
my_svddiff.update_labels_with_mc()
# %%
my_svddiff.compute_U_per_digit(my_svddiff.updated_labels[-1])

# %%
my_svddiff.compute_res(25)
my_svddiff.compute_accuracy()

# %%
x_test_res_4, x_test_pred_4 = my_svddiff.predict(
    X_test.astype(np.float32), 25, 3
)
# %%
my_svddiff.find_mc_samples()
# %%
my_svddiff.update_labels_with_mc()
# %%
my_svddiff.compute_U_per_digit(my_svddiff.updated_labels[-1])
my_svddiff.compute_res(25)
my_svddiff.compute_accuracy()
# %%
train_accuracies_per_iteration = [acc for acc in my_svddiff.accuracy.values()]
# %%
x_test_res_1, x_test_pred_1 = my_svddiff.predict(
    X_test.astype(np.float32), 25, 0
)
# %%
test_preds = [x_test_pred_1, x_test_ped_2, x_test_pred_3, x_test_pred_4]

#%%
from sklearn.metrics import accuracy_score

test_accuracies_per_iteration = [
    accuracy_score(y_test, pred) for pred in test_preds
]
# %%
import seaborn as sns

sns.set_style("whitegrid")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_accuracies_per_iteration, label="Train", marker="o")
ax.plot(test_accuracies_per_iteration, label="Valid", marker="o")
ax.legend()
ax.set_xlabel("Number of iterations")
ax.set_ylabel("Accuracy")
plt.savefig("iterative_train_valid.pdf", bbox_inches="tight", dpi=800)
plt.show()
# %%
