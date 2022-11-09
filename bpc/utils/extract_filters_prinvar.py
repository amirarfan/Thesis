#%%
from julia.api import Julia
import sys
import numpy as np
import scipy.io as sio

Julia()
from julia import Main as jl
import operator

prinvar = jl.include(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\utils\prinvar.jl"
)


def extract_filters_np(images: np.ndarray):
    def reshape_split(
        image: np.ndarray, kernel_size: tuple, center: bool = True
    ):
        img_height, img_width = image.shape
        tile_height, tile_width = kernel_size

        tiled_array = image.reshape(
            img_height // tile_height,
            tile_height,
            img_width // tile_width,
            tile_width,
        )
        tiled_array = tiled_array.swapaxes(1, 2)
        tiled_array = tiled_array.reshape(-1, kernel_size[0], kernel_size[1])

        if center:
            bounding = kernel_size
            start = tuple(
                map(lambda a, da: a // 2 - da // 2, image.shape, bounding)
            )
            end = tuple(map(operator.add, start, bounding))
            slices = tuple(map(slice, start, end))
            tiled_array = np.vstack(
                (tiled_array, np.expand_dims(image[slices], axis=0))
            )
        return tiled_array

    filters = np.zeros((images.shape[0], 5, 14, 14))
    for image in range(images.shape[0]):
        filters[image] = reshape_split(images[image], (14, 14))
    return filters


def extract_filters_prinvar(X, y, num_per_patch):
    divided_images = [X[y == i] for i in range(1, 11)]
    filters = np.zeros((10, 5, num_per_patch, 14, 14))

    for i in range(10):
        patches = extract_filters_np(divided_images[i])
        for j in range(5):
            current_patches = patches[:, j, :, :]
            current_patches = current_patches.reshape(
                current_patches.shape[0], -1
            )
            _, _, ids, _, _, _, _, _ = prinvar(
                current_patches.T, num_per_patch
            )
            filters[i, j] = current_patches[ids, :].reshape(
                num_per_patch, 14, 14
            )

    return filters


def extract_images_prinvar(X, y, num_images):
    divided_images = [X[y == i] for i in range(1, 11)]
    images = []
    for i in range(10):
        current_images = divided_images[i]
        current_images = current_images.reshape(current_images.shape[0], -1)
        _, _, ids, _, _, _, _, _ = prinvar(current_images.T, num_images)
        images.append(current_images[ids, :].reshape(len(ids), 28, 28))

    return images


def extract_images_prinvar_hu(X, Xhu, y, num_images):
    divided_images = [X[y == i] for i in range(1, 11)]
    divided_images_hu = [Xhu[y == i] for i in range(1, 11)]
    images = []
    for i in range(10):
        current_images = divided_images[i]
        current_hu = divided_images_hu[i]
        _, _, ids, _, _, _, _, _ = prinvar(current_hu.T, num_images)
        images.append(current_images[ids, :784].reshape(len(ids), 28, 28))

    return images


#%%
if __name__ == "__main__":

    data = sio.loadmat(
        r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\imageclassification_proj\data\mnistdata.mat"
    )
    x_train = data["training"]
    x_test = data["testing"]
    y_train = np.concatenate(data["Gtrain"])
    y_test = np.concatenate(data["Gtest"])

    filters = extract_images_prinvar(x_train, y_train, 10)


# %%
