# Using Intel Python Library for Speedup
import numpy as np
import numba
from numba import prange


@numba.jit(nopython=True, fastmath=True)
def compute_res(u, z, I):
    return np.linalg.norm((I - np.dot(u, u.T)) @ z)


@numba.jit(nopython=True, parallel=True, fastmath=True)
def compute_diff(x_targ, num_vecs, U_per_digit):
    residuals = np.zeros((x_targ.shape[0], U_per_digit.shape[0]), np.float32)
    I = np.eye(x_targ.shape[1], dtype=np.float32)
    for i in prange(x_targ.shape[0]):
        for digit in range(U_per_digit.shape[0]):

            residuals[i, digit] = np.linalg.norm(
                (
                    I
                    - np.dot(
                        U_per_digit[digit][:, :num_vecs],
                        U_per_digit[digit][:, :num_vecs].T,
                    )
                )
                @ x_targ[i]
            )

    return residuals


if __name__ == "__main__":
    # Run With %time to measure time of the script
    U = np.random.rand(10, 784, 700)
    Z = np.random.rand(100, 784)
    print(U.dtype)
    print(Z.dtype)
    print(compute_res(U[0], Z[0], np.eye(784, dtype=np.float32)))
# compute_diff(Z, 20, U)
