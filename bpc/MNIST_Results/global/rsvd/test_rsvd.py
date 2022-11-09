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
from julia.api import Julia

Julia()
from julia import Main as jl

cv_classification = jl.include(r"ReturnCVClassified.jl")
#%%
from scipy.linalg.interpolative import svd

#%%

from sklearn.utils.extmath import randomized_svd


# %%
data = sio.loadmat(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\data\mnistdata.mat"
)


y_train = np.concatenate(data["Gtrain"])
#
X = np.load(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\MNIST_Results\global\x_convolved_64475.npy"
)
#
#%%
#%%
# X = X - x_mean
#%%
y_train_oc = np.identity(10)[y_train - 1]
#%%
U, S, V = randomized_svd((X - np.mean(X, axis=0)), 15000)
# U = np.load("U.npy")
# S = np.load("S.npy")
# V = np.load("V_t.npy")
#%%
#%%
np.save("U_15k.npy", U)
np.save("S_15k.npy", S)
np.save("V_t_15k.npy", V)
# %%
Ghat, GhatCV, blamb, Yhat, YhatCV = cv_classification(
    X, y_train_oc, None, 10, False, U, S, V.T
)

# %%

Ghat = np.concatenate(Ghat)
GhatCV = np.concatenate(GhatCV)
# %%
from sklearn.metrics import accuracy_score

# %%
print(accuracy_score(Ghat, y_train))
print(accuracy_score(GhatCV, y_train))
# %%
