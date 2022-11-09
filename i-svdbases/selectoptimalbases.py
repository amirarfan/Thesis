#%%
import numpy as np
import scipy.io as sio
from juliacall import Main as jl
from get_sing_vals import get_sing_vals
from sklearn.model_selection import train_test_split
from compute_res import compute_diff
from sklearn.metrics import accuracy_score

#%%
data = sio.loadmat(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\data\mnistdata.mat"
)

training = data["training"]
y = np.concatenate(data["Gtrain"])

# %%
training = training.reshape(training.shape[0], -1)

#%%
X_train, X_test, y_train, y_test = train_test_split(
    training, y, stratify=y, train_size=0.8
)
# %%

U_digit, S_digit, V_digit = get_sing_vals(X_train, y_train)
# %%
components = np.arange(0, 95, 5)
components[0] = 1

#%%

train_diffs_per_component = [
    compute_diff(X_train.astype(np.float32), comp, U_digit)
    for comp in components
]

#%%
test_diffs_per_component = [
    compute_diff(X_test.astype(np.float32), comp, U_digit)
    for comp in components
]

#%%

train_predictions = [np.argmin(train_diffs, axis=1)+1 for train_diffs in train_diffs_per_component]
# %%
accuracy_train = [accuracy_score(y_train, train_prediction) for train_prediction in train_predictions]

# %%
test_predictions = [np.argmin(test_diffs, axis=1)+1 for test_diffs in test_diffs_per_component]
# %%
accuracy_test = [accuracy_score(y_test, test_prediction) for test_prediction in test_predictions]
# %%

np.save("accuracy_train.npy", accuracy_train)
np.save("accuracy_test.npy", accuracy_test)
np.save("train_diffs_per_component.npy", train_diffs_per_component)
np.save("test_diffs_per_component.npy", test_diffs_per_component)
np.save("train_predictions.npy", train_predictions)
np.save("test_predictions.npy", test_predictions)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

# %%
np.save("Xtrain.npy", X_train)
np.save("Xtest.npy", X_test)




#%%
test_predictions = np.load(r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\i-svdbases\selectoptimalbasesfiles\test_predictions.npy")

train_predictions = np.load(r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\i-svdbases\selectoptimalbasesfiles\train_predictions.npy")

#%%
y_test = np.load(r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\i-svdbases\selectoptimalbasesfiles\y_test.npy")

#%%
y_train = np.load(r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\i-svdbases\selectoptimalbasesfiles\y_train.npy")

#%%
# Plot Train and Test Accuracies with respect to number of components
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as plticker
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(components, accuracy_train, label="Train", marker="o")
ax.plot(components, accuracy_test, label="Valid", marker="o")
ax.set_xlabel("Number of Bases")
ax.set_ylabel("Accuracy")
loc = plticker.MultipleLocator(base=5.0)
ax.xaxis.set_major_locator(loc)
ax.legend()
plt.savefig("accuracy_per_component.pdf", bbox_inches="tight", dpi=800)
plt.show()

# %%
