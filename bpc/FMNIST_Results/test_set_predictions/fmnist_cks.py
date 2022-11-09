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
from cks import ks_variant
from convolve_filts import convolve_filts
from extract_filters import extract_filters_np
from load_fmnist import load_fmnist


from juliacall import Main as jl
#%%
cv_prediction = jl.include(r"..\..\LooCV\ReturnCVPredicted.jl")

#%%
# Load Data
datapath = r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\data\fmnist"
#%%

x_test, y_test = load_fmnist(datapath, kind="t10k")
y_train = y_test.copy()
y_train += 1
x_test = x_test.reshape(x_test.shape[0], 28, 28)
# %%

filters = np.load(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\FMNIST_Results\filters_61210.npy"
)

#%%
x_test_convolved = convolve_filts(x_test, None, filters, bsize=32)
#%%
x_orig = np.hstack((x_test.reshape(x_test.shape[0], -1), x_test_convolved))
#%%
b_lambda = np.load(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\FMNIST_Results\cks_61210\b_cks_61210.npy"
)

preds, ys = cv_prediction(x_orig, b_lambda, 10)

# preds = np.concatenate(preds)
#%%
print(accuracy_score(y_train, preds))


# %%
