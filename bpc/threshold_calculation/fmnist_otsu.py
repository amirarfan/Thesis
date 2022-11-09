#%%
import sys
import os

sys.path.append(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\utils"
)
import numpy as np
from load_fmnist import load_fmnist
from extract_filters import extract_fillters_otsu

#%%
datapath = r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\data\fmnist"
x_train, y_train = load_fmnist(datapath, kind="train")

#%%
x_train = x_train.reshape(x_train.shape[0], 28, 28)


#%%
thresholds = extract_fillters_otsu(x_train)
# %%
np.round(np.mean(thresholds))
# %%
# Save mean of thresholds to text file
with open(
    os.path.join(os.path.dirname(__file__), "thresholds_fmnist.txt"), "a"
) as f:
    f.write(str(np.round(np.mean(thresholds))) + "\n")
# %%
