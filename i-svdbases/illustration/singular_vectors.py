#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import scipy.io as sio

sys.path.append(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\i-svdbases"
)
from isvbase import SVDDiff

#%%
data = sio.loadmat(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\data\mnistdata.mat"
)

#%%
x_train = data["training"]
y_train = np.concatenate(data["Gtrain"])

#%%
my_svddiff = SVDDiff(x_train, y_train, None)

#%%
sing_vecs = my_svddiff.U_per_digit[0]
# %%
num_sing_vecs = [0, 10, 100, 500]
classes = [2, 8]
fig, axes = plt.subplots(2, 4, figsize=(12, 12))

for idx, axi in enumerate(axes):
    curr_sing_vecs = sing_vecs[classes[idx], :, :]
    for ax, sing_vec in zip(axi, num_sing_vecs):
        
        ax.imshow(curr_sing_vecs[:, sing_vec].reshape(28, 28), cmap="gray")
        ax.axis("off")
        ax.set_title(f"Left Singular Vector # {sing_vec}")
        ax.set_xticks([])
        ax.set_yticks([])
plt.subplots_adjust(top=1.5)
plt.gcf().tight_layout()
plt.savefig("singular_vectors_mnist.pdf", bbox_inches="tight", dpi=800)
plt.show()
# %%
