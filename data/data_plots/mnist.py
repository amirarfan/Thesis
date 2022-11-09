#%%
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

#%%
data = sio.loadmat("../mnistdata.mat")


# %%
x_train = data["training"]
y_train = np.concatenate(data["Gtrain"])
# %%
# Plot each random digit from each group using x_train and y_train, and save the plot
fig, ax = plt.subplots(2, 5, figsize=(10, 10))

for i in range(2):
    for j in range(5):
        ax[i, j].imshow(
            x_train[
                np.random.choice(np.where(y_train == 5 * i + j + 1)[0])
            ].reshape(28, 28),
            cmap="gray",
        )
        ax[i, j].set_title(f"Digit {5*i+j+1}")
        ax[i, j].axis("off")

plt.gcf().tight_layout()
plt.subplots_adjust(top=0.6)
plt.savefig("random_digits.pdf", dpi=800, bbox_inches="tight")
plt.show()

#%%
# Plot distribution of each digit in the training set, set x-axis to be 1-10
import seaborn as sns
sns.set_style("white")
sns.set_palette("dark")
plt.figure(figsize=(8, 4))
plt.hist(y_train, bins=np.arange(1, 12)-0.5, rwidth=0.8)
plt.xticks(range(1, 11))
plt.xlabel("Digit", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.title("Distribution of Digits in Training Set", fontsize=16)
plt.savefig("digit_distribution.pdf", dpi=800, bbox_inches="tight")
plt.show()
