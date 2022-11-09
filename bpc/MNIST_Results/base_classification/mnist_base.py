#%%
import os
import sys

from sklearn.metrics import accuracy_score
import numpy as np
import scipy.io as sio
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

oc_y_train = np.identity(10)[y_train - 1]

Ghat, GhatCV, b, _, _ = cv_classification(
    x_train.reshape(x_train.shape[0], -1), oc_y_train, None, 10, False
)
# %%
Ghat = np.concatenate(Ghat)
GhatCV = np.concatenate(GhatCV)
#%%
# Save accuracy on GhatCV to text file
print("Saving accuracy to text file")
with open(
    os.path.join(os.path.dirname(__file__), "cv_accuracy_base.txt"), "a"
) as filename:
    filename.write(f"CV_base:" + str(accuracy_score(y_train, GhatCV)))
    filename.write(f"\nTrain_base:" + str(accuracy_score(y_train, Ghat)))
    filename.write("\n")
# %%
