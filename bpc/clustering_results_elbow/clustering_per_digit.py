#%%
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import scipy.io as sio
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
import numpy as np
import seaborn as sns

#%%
data = sio.loadmat(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\data\mnistdata.mat"
)
data = sio.loadmat(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\data\mnistdata.mat"
)
x_train = data["training"]
y_train = np.concatenate(data["Gtrain"])

#%%
divided_data = [x_train[y_train == i] for i in range(1, 11)]

#%%
distorsions_per_group = {}
for i, X in enumerate(divided_data):
    X = X.reshape(X.shape[0], -1)
    distorsions = []
    for k in range(2, 11):  # 2-10 clusters
        print("Group: " + str(i) + " k: " + str(k))
        cluster_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(n_clusters=k)),
            ]
        )

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distorsions.append(kmeans.inertia_)
    distorsions_per_group[i + 1] = distorsions

#%%
sns.set_style("whitegrid")
fig, axes = plt.subplots(5, 2, figsize=(24, 12))
for idx, ax in enumerate(axes.flatten()):
    ax.set_title("Group: " + str(idx + 1))
    sns.lineplot(x=range(2, 11), y=distorsions_per_group[idx + 1], ax=ax)

fig.suptitle("Elbow method for different digit groups", fontsize=22)

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()
# %%
