#%%
import os
import sys

from sklearn.metrics import accuracy_score
import numpy as np
import scipy.io as sio
from julia.api import Julia

sys.path.append(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\utils"
)

from load_fmnist import load_fmnist

Julia()
from julia import Main as jl

cv_classification = jl.include(r"..\..\LooCV\ReturnCVClassified.jl")
#%%
datapath = r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\data\fmnist"
# Load Data
x_train, y_train = load_fmnist(datapath, kind="train")
y_train = y_train.copy()
y_train += 1

#%%
x_train = x_train.copy()

#%%

oc_y_train = np.identity(10)[y_train - 1]
#%%

Ghat, GhatCV, b, _, _ = cv_classification(x_train, oc_y_train, None, 10, False)
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
