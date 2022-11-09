#%%
import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score

#%%
data = sio.loadmat(r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\data\mnistdata.mat")

y_train = data["Gtrain"]

#%%
Ghat = np.concatenate(np.load("Ghat_cks_100.npy"))
GhatCV = np.concatenate(np.load("GhatCV_cks_100.npy"))

#%%
print(accuracy_score(y_train, GhatCV))

print(accuracy_score(y_train, Ghat))
# %%
