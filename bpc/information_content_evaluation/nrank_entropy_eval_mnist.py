#%%
import os
import sys
import numba

sys.path.append(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\utils"
)
sys.path.append(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\LooCV"
)
import numpy as np
import scipy.io as sio
from convolve_filts import convolve_filts_entropy, convolve_filts_full
from extract_filters import extract_filters_np
from cks import ks_variant
from skimage.measure import shannon_entropy

#%%
data = sio.loadmat(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\data\mnistdata.mat"
)
x_train = data["training"]
y_train = np.concatenate(data["Gtrain"])

#%%
entropy_per = {}
matrix_rank_per = {}
for p in range(1, 11):
    current_x = x_train[y_train == p]
    images_to_extract = np.concatenate(
        ks_variant(
            current_x.reshape(current_x.shape[0], -1),
            y_train[y_train == p],
            5,
            5,
        )
    )
    images_to_extract = images_to_extract.reshape(
        images_to_extract.shape[0], 28, 28
    )
    filters = extract_filters_np(images_to_extract)
    filters = filters.reshape(filters.shape[0] * filters.shape[1], 14, 14)
    entropy_combs = {}
    matrix_rank_combs = {}
    for k in range(1, 11):
        random_imgs = x_train[y_train == k][
            np.random.randint(0, x_train[y_train == k].shape[0], 100)
        ]
        entropies = convolve_filts_full(random_imgs, None, filters, bsize=5000)
        entropy_means_per_sample = []
        entropy_ranks_per_sample = []
        for i in range(entropies.shape[0]):
            matrix_ranks = []
            entropy_samples = []
            for j in range(entropies.shape[1]):
                entropy_samples.append(shannon_entropy(entropies[i, j, :, :]))
                matrix_ranks.append(
                    np.linalg.matrix_rank(entropies[i, j, :, :])
                )
            entropy_means_per_sample.append(np.mean(entropy_samples))
            entropy_ranks_per_sample.append(np.mean(matrix_ranks))
        entropy_combs[k] = np.mean(entropy_means_per_sample)
        matrix_rank_combs[k] = np.mean(entropy_ranks_per_sample)
    entropy_per[p] = entropy_combs
    matrix_rank_per[p] = matrix_rank_combs


# %%
import pandas as pd

# Create a dataframe from each dictionary in entropy_per as columns
#%%
my_df_entropies = pd.DataFrame.from_dict(entropy_per, orient="columns")
#%%
my_df_ranks = pd.DataFrame.from_dict(matrix_rank_per, orient="columns")

#%%
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# plot single heatmap in figure
row_max = my_df_entropies.idxmax(axis=1)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
sns.heatmap(
    my_df_entropies,
    ax=axes[0],
    annot=True,
    fmt=".2f",
    cmap="crest",
    vmin=8,
    vmax=10,
    annot_kws={"fontsize": 12.5},
)
axes[0].set_xlabel("Source class", fontsize=14)
axes[0].set_ylabel("Target class", fontsize=14)
axes[0].set_title("Mean Shannon Entropy of Feature Maps", fontsize=16)
# Increase distance between suptitle and plot
# plt.subplots_adjust(top=0.95)


for row, index in enumerate(my_df_entropies):
    position = my_df_entropies.columns.get_loc(row_max[index])
    axes[0].add_patch(
        Rectangle((position, row), 1, 1, fill=False, edgecolor="red", lw=1)
    )


row_max = my_df_ranks.idxmax(axis=1)
sns.heatmap(
    my_df_ranks,
    ax=axes[1],
    annot=True,
    fmt=".2f",
    cmap="crest",
    vmin=20,
    vmax=30,
    annot_kws={"fontsize": 12.5},
)
axes[1].set_xlabel("Source class", fontsize=14)
axes[1].set_ylabel("Target class", fontsize=14)
axes[1].set_title("Mean Matrix Rank of Feature Maps", fontsize=16)

for row, index in enumerate(my_df_ranks):
    position = my_df_ranks.columns.get_loc(row_max[index])
    axes[1].add_patch(
        Rectangle((position, row), 1, 1, fill=False, edgecolor="red", lw=1)
    )

plt.tight_layout()
plt.savefig("mnist_matrix_rank_entropy.pdf", dpi=800, bbox_inches="tight")
plt.show()

# %%
