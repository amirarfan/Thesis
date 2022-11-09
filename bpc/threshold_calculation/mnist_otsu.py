#%%
import sys
import os
import scipy.io as sio

sys.path.append(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\utils"
)
import numpy as np
from load_fmnist import load_fmnist
from extract_filters import extract_fillters_otsu

#%%
data = sio.loadmat(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\data\mnistdata.mat"
)
x_train = data["training"]


#%%
thresholds = extract_fillters_otsu(x_train)
# %%
np.round(np.mean(thresholds))
# %%
# Save mean of thresholds to text file
with open(
    os.path.join(os.path.dirname(__file__), "thresholds_mnist.txt"), "a"
) as f:
    f.write(str(np.round(np.mean(thresholds))) + "\n")
# %%
