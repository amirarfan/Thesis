import numpy as np
from skimage.measure import moments_central, moments_normalized, moments_hu


def computescale_humoments(x):
    x_moments = np.array(
        list(
            map(
                lambda x: moments_hu(moments_normalized(moments_central(x))),
                list(x),
            )
        )
    )
    scale_moments = np.vectorize(lambda x: -np.sign(x) * np.log10(np.abs(x)))
    x_moments = scale_moments(x_moments)
    return x_moments
