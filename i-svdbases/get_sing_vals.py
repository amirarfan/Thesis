import numpy as np
import scipy.linalg as la


def get_sing_vals(x_tr, y_train, reshape=False):

    U_per_digit = np.zeros(
        (np.max(y_train), x_tr.shape[1], x_tr.shape[1]), dtype=np.float32
    )

    S_per_digit = np.zeros(np.max(y_train), dtype=object)

    V_per_digit = np.zeros(np.max(y_train), dtype=object)

    if reshape:

        x_tr = x_tr.reshape(x_tr.shape[0], -1)

    for i in range(1, np.max(y_train) + 1):

        U_trm, S_trm, V_trm = la.svd(x_tr[y_train == i].T)

        U_per_digit[i - 1] = U_trm

        S_per_digit[i - 1] = S_trm

        V_per_digit[i - 1] = V_trm

    return U_per_digit, S_per_digit, V_per_digit
