import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
from sklearn.decomposition import PCA
from skimage.measure import (
    inertia_tensor,
    shannon_entropy,
    moments_central,
    moments_hu,
    moments_normalized,
)


def convolve_filts(training_data, testing_data, filters, bsize=5000):

    fsize = filters.shape[0]

    kernels = torch.tensor(filters).cuda()
    kernels.unsqueeze_(dim=1)
    training_data = torch.tensor(training_data)
    training_data.unsqueeze_(dim=1)
    if testing_data is not None:
        testing_data = torch.tensor(testing_data).cuda().unsqueeze_(1)
    conv_output_training = torch.empty((training_data.shape[0], fsize))
    prev_indices = 0
    for idx, batch in enumerate(torch.split(training_data, bsize)):
        batch = batch.cuda()
        next_indices = prev_indices + batch.shape[0]
        conv_output_training[prev_indices:next_indices] = torch.max(
            torch.reshape(
                F.conv2d(batch.float(), kernels.float(), padding="same"),
                (batch.shape[0], fsize, -1),
            ),
            2,
        )[0]
        prev_indices = next_indices

    if testing_data is not None:

        conv_output_testing = torch.empty((testing_data.shape[0], fsize))
        prev_indices = 0
        for idx, batch in enumerate(torch.split(testing_data, bsize)):
            batch = batch.cuda()
            next_indices = prev_indices + batch.shape[0]
            conv_output_testing[prev_indices:next_indices] = torch.max(
                torch.reshape(
                    F.conv2d(batch.float(), kernels.float(), padding="same"),
                    (batch.shape[0], fsize, -1),
                ),
                2,
            )[0]
            prev_indices = next_indices
        return conv_output_training.numpy(), conv_output_testing.numpy()

    return conv_output_training.numpy()


def convolve_filts_full(training_data, testing_data, filters, bsize=5000):
    fsize = filters.shape[0]
    kernels = torch.tensor(filters).cuda()
    kernels.unsqueeze_(dim=1)
    training_data = torch.tensor(training_data)
    training_data.unsqueeze_(dim=1)
    if testing_data is not None:
        testing_data = torch.tensor(testing_data).cuda().unsqueeze_(1)
    conv_output_training = torch.empty((training_data.shape[0], fsize, 28, 28))
    prev_indices = 0
    for idx, batch in enumerate(torch.split(training_data, bsize)):
        batch = batch.cuda()
        next_indices = prev_indices + batch.shape[0]
        conv_output_training[prev_indices:next_indices] = torch.reshape(
            F.conv2d(batch.float(), kernels.float(), padding="same"),
            (batch.shape[0], fsize, 28, 28),
        )
        prev_indices = next_indices

    if testing_data is not None:

        conv_output_testing = torch.empty(
            (testing_data.shape[0], fsize * 28 * 28)
        )
        prev_indices = 0
        for idx, batch in enumerate(torch.split(testing_data, bsize)):
            batch = batch.cuda()
            next_indices = prev_indices + batch.shape[0]
            conv_output_testing[prev_indices:next_indices] = torch.reshape(
                F.conv2d(batch.float(), kernels.float(), padding="same"),
                (batch.shape[0], -1),
            )
            prev_indices = next_indices
        return conv_output_training.numpy(), conv_output_testing.numpy()

    return conv_output_training.numpy()


def convolve_filts_full_second_order(
    training_data, testing_data, filters, bsize=5000
):
    fsize = filters.shape[0]
    kernels = torch.tensor(filters).cuda()
    kernels.unsqueeze_(dim=1)
    training_data = torch.tensor(training_data).cuda()
    training_data.unsqueeze_(dim=1)
    if testing_data is not None:
        testing_data = torch.tensor(testing_data).cuda().unsqueeze_(1)
    conv_output_training = torch.empty((training_data.shape[0], fsize))
    conv_output_training_second_order = torch.empty(
        (training_data.shape[0], fsize**2)
    )
    prev_indices = 0
    for idx, batch in enumerate(torch.split(training_data, bsize)):
        batch = batch.cuda()
        next_indices = prev_indices + batch.shape[0]
        convoluted_vals = F.conv2d(
            batch.float(), kernels.float(), padding="same"
        )
        conv_output_training[prev_indices:next_indices] = torch.max(
            torch.reshape(convoluted_vals, (batch.shape[0], fsize, -1)), 2
        )[0]

        temp_conv_second_order = torch.empty((batch.shape[0], fsize**2))
        for i in range(batch.shape[0]):
            temp_conv_second_order[i] = torch.max(
                torch.reshape(
                    F.conv2d(
                        convoluted_vals[i].unsqueeze(dim=1).float(),
                        kernels.float(),
                        padding="same",
                    ),
                    (fsize, fsize, -1),
                ),
                2,
            )[0].flatten()
        conv_output_training_second_order[
            prev_indices:next_indices
        ] = temp_conv_second_order

        prev_indices = next_indices

    if testing_data is not None:

        conv_output_testing = torch.empty(
            (testing_data.shape[0], fsize * 28 * 28)
        )
        prev_indices = 0
        for idx, batch in enumerate(torch.split(testing_data, bsize)):
            batch = batch.cuda()
            next_indices = prev_indices + batch.shape[0]
            conv_output_testing[prev_indices:next_indices] = torch.reshape(
                F.conv2d(batch.float(), kernels.float(), padding="same"),
                (batch.shape[0], -1),
            )
            prev_indices = next_indices
        return conv_output_training.numpy(), conv_output_testing.numpy()

    return (
        conv_output_training.numpy(),
        conv_output_training_second_order.numpy(),
    )


def convolve_filts_sv(training_data, testing_data, filters, bsize=5000):

    fsize = filters.shape[0]

    kernels = torch.tensor(filters).cuda()
    kernels.unsqueeze_(dim=1)
    training_data = torch.tensor(training_data).cuda()
    training_data.unsqueeze_(dim=1)
    if testing_data is not None:
        testing_data = torch.tensor(testing_data).cuda().unsqueeze_(1)
    conv_output_training = torch.empty((training_data.shape[0], fsize))
    conv_singular_vals_training = torch.empty(
        training_data.shape[0], 28 * fsize
    )
    prev_indices = 0
    for idx, batch in enumerate(torch.split(training_data, bsize)):
        batch = batch.cuda()
        next_indices = prev_indices + batch.shape[0]
        current_conv = F.conv2d(batch.float(), kernels.float(), padding="same")
        conv_singular_vals_training[prev_indices:next_indices] = torch.reshape(
            torch.linalg.svdvals(current_conv), (batch.shape[0], -1)
        )
        conv_output_training[prev_indices:next_indices] = torch.max(
            torch.reshape(
                current_conv,
                (batch.shape[0], fsize, -1),
            ),
            2,
        )[0]
        prev_indices = next_indices

    if testing_data is not None:

        conv_output_testing = torch.empty((testing_data.shape[0], fsize))
        conv_singular_vals_testing = torch.empty(
            testing_data.shape[0], fsize * 28
        )
        prev_indices = 0
        for idx, batch in enumerate(torch.split(testing_data, bsize)):
            batch = batch.cuda()
            next_indices = prev_indices + batch.shape[0]
            current_conv = F.conv2d(
                batch.float(), kernels.float(), padding="same"
            )
            conv_singular_vals_testing[
                prev_indices:next_indices
            ] = torch.reshape(
                torch.linalg.svdvals(current_conv), (batch.shape[0], -1)
            )
            conv_output_testing[prev_indices:next_indices] = torch.max(
                torch.reshape(
                    current_conv,
                    (batch.shape[0], fsize, -1),
                ),
                2,
            )[0]
            prev_indices = next_indices
        return (
            conv_output_training.numpy(),
            conv_output_testing.numpy(),
            conv_singular_vals_training.numpy(),
            conv_singular_vals_testing.numpy(),
        )

    return conv_output_training.numpy(), conv_singular_vals_training.numpy()


def shannon_entropy_pytorch(x):
    x = torch.unique(x, return_counts=True)[1].float()
    x = x / torch.sum(x)
    return -x.mul(x.log2()).sum()


def convolve_filts_entropy(training_data, testing_data, filters, bsize=5000):

    fsize = filters.shape[0]

    kernels = torch.tensor(filters).cuda()
    kernels.unsqueeze_(dim=1)
    training_data = torch.tensor(training_data)
    training_data.unsqueeze_(dim=1)
    if testing_data is not None:
        testing_data = torch.tensor(testing_data).cuda().unsqueeze_(1)
    conv_filts_entropy = torch.empty(training_data.shape[0], fsize)
    prev_indices = 0
    for idx, batch in enumerate(torch.split(training_data, bsize)):
        batch = batch.cuda()
        next_indices = prev_indices + batch.shape[0]
        current_conv = F.conv2d(batch.float(), kernels.float(), padding="same")
        temp_conv_filts_entropy = torch.empty((batch.shape[0], fsize))
        for i in range(batch.shape[0]):
            for j in range(fsize):
                temp_conv_filts_entropy[i, j] = shannon_entropy_pytorch(
                    current_conv[i, j, :, :]
                )
        conv_filts_entropy[prev_indices:next_indices] = temp_conv_filts_entropy
        prev_indices = next_indices

    if testing_data is not None:

        conv_output_testing = torch.empty((testing_data.shape[0], fsize))
        conv_singular_vals_testing = torch.empty(
            testing_data.shape[0], fsize * 28
        )
        prev_indices = 0
        for idx, batch in enumerate(torch.split(testing_data, bsize)):
            batch = batch.cuda()
            next_indices = prev_indices + batch.shape[0]
            current_conv = F.conv2d(
                batch.float(), kernels.float(), padding="same"
            )
            conv_singular_vals_testing[
                prev_indices:next_indices
            ] = torch.reshape(
                torch.linalg.svdvals(current_conv), (batch.shape[0], -1)
            )
            conv_output_testing[prev_indices:next_indices] = torch.max(
                torch.reshape(
                    current_conv,
                    (batch.shape[0], fsize, -1),
                ),
                2,
            )[0]
            prev_indices = next_indices
        return (
            conv_output_testing.numpy(),
            conv_singular_vals_testing.numpy(),
        )

    return conv_filts_entropy.numpy()
