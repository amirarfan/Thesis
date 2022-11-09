#%%
folder = "cks_61210"
fsize = 61210
import sys
import os

sys.path.append(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\utils"
)
sys.path.append(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\LooCV"
)
import numpy as np
from rs import rs
from extract_filters import extract_filters_np
import scipy.io as sio
from convolve_filts import convolve_filts
from load_fmnist import load_fmnist
from sklearn.metrics import accuracy_score

#%%
Ghat = np.concatenate(np.load(f"{folder}/Ghat_cks_{fsize}.npy"))
GhatCV = np.concatenate(np.load(f"{folder}/GhatCV_cks_{fsize}.npy"))

#%%
x_train, y_train = load_fmnist(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\data\fmnist",
    kind="train",
)

y_train = y_train.copy() + 1

#%%
with open(os.path.join("cv_accuracy_cks.txt"), "a") as filename:
    filename.write(f"CV_{fsize}:" + str(accuracy_score(y_train, GhatCV)))
    filename.write(f"\nTrain_{fsize}:" + str(accuracy_score(y_train, Ghat)))
    filename.write("\n")

# %%
