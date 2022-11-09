import operator
import cv2 as cv
import numpy as np


def extract_filters_np(
    images: np.ndarray,
    otsu: bool = False,
    threshold: int = 95,
    kernel_size: tuple = (14, 14),
    center: bool = True,
) -> np.ndarray:
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

    center_addition = 1 if center else 0
    num_filters = int(
        (images.shape[1] ** 2 / kernel_size[0] ** 2) + center_addition
    )
    filters = np.zeros(
        (images.shape[0], num_filters, kernel_size[0], kernel_size[1])
    )
    for image in range(images.shape[0]):
        filters[image] = reshape_split(
            images[image], kernel_size, center=center
        )

    if not otsu:
        filters = np.where(filters > threshold, 1, -1)
    else:
        filters = filters.reshape(filters.shape[0] * filters.shape[1], 14, 14)
        for filter in range(filters.shape[0]):
            thresh, filters[filter] = cv.threshold(
                filters[filter].astype("uint8"),
                0,
                1,
                cv.THRESH_BINARY + cv.THRESH_OTSU,
            )

        filters = filters.reshape(filters.shape[0] * filters.shape[1], -1)
        filters = np.where(filters == 1, 1, -1)
        filters = filters.reshape(filters.shape[0], 14, 14)

    return filters


def extract_fillters_otsu(
    images: np.ndarray,
    kernel_size: tuple = (14, 14),
    center: bool = True,
) -> np.ndarray:
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

    center_addition = 1 if center else 0
    num_filters = int(
        (images.shape[1] ** 2 / kernel_size[0] ** 2) + center_addition
    )
    filters = np.zeros(
        (images.shape[0], num_filters, kernel_size[0], kernel_size[1])
    )
    for image in range(images.shape[0]):
        filters[image] = reshape_split(
            images[image], kernel_size, center=center
        )

    thresholds = []
    filters = filters.reshape(filters.shape[0] * filters.shape[1], 14, 14)
    for filter in range(filters.shape[0]):
        thresh, filters[filter] = cv.threshold(
            filters[filter].astype("uint8"),
            0,
            1,
            cv.THRESH_BINARY + cv.THRESH_OTSU,
        )
        thresholds.append(thresh)

    return thresholds
