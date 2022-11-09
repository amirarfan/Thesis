import numpy as np
from get_sing_vals import get_sing_vals
from sklearn.metrics import accuracy_score
from compute_res import compute_diff


class SVDDiff:
    def __init__(self, train_data, y_train, precomputed_residuals):

        self.pred_converter = np.vectorize(self.convert_predictions)
        self.mc_converter = np.vectorize(self.mcify)

        self.U_iteration = 0
        self.res_iteration = 0

        self.train_data = train_data.reshape(train_data.shape[0], -1).astype(
            np.float32
        )

        self.y_train = y_train

        self.U_per_digit = {}

        self.compute_U_per_digit(self.y_train)

        self.mc_indexes = {}
        self.predictions = {}
        self.misclassified_samples = {}
        self.accuracy = {}

        self.updated_labels = []

        if precomputed_residuals is not None:
            self.residuals = {0: precomputed_residuals}
        else:
            self.residuals = {}

    @staticmethod
    def convert_predictions(pred_iter, pred):
        if pred > pred_iter * 10:
            return pred - (pred_iter * 10)
        else:
            return SVDDiff.convert_predictions(pred_iter - 1, pred)

    @staticmethod
    def mcify(res_iter, label):
        return (res_iter * 10) + label

    def compute_U_per_digit(self, labels):
        U, _, _ = get_sing_vals(self.train_data, labels)
        if self.U_iteration not in self.U_per_digit:
            self.U_per_digit[self.U_iteration] = U
        else:
            self.U_iteration += 1
            self.U_per_digit[self.U_iteration] = U

    def compute_res(self, num_vecs):
        res = compute_diff(
            self.train_data, num_vecs, self.U_per_digit[self.U_iteration]
        )

        if self.res_iteration not in self.residuals:
            self.residuals[self.res_iteration] = res
        else:
            self.res_iteration += 1
            self.residuals[self.res_iteration] = res

    def compute_accuracy(self):
        predictions = np.argmin(self.residuals[self.res_iteration], axis=1) + 1
        self.predictions[self.res_iteration] = self.pred_converter(
            self.res_iteration, predictions
        )
        self.accuracy[self.res_iteration] = accuracy_score(
            self.predictions[self.res_iteration], self.y_train
        )
        print(self.accuracy[self.res_iteration])

    def find_mc_samples(self):
        self.mc_indexes[self.res_iteration] = np.where(
            self.predictions[self.res_iteration] != self.y_train
        )[0]
        print(
            f"Found {self.mc_indexes[self.res_iteration].shape[0]} misclassified samples"
        )

    def update_labels_with_mc(self):

        true_mc_labels = self.y_train[self.mc_indexes[self.res_iteration]]
        true_mc_labels_converted = self.mc_converter(
            self.res_iteration + 1, true_mc_labels
        )

        if len(self.updated_labels) > 0:
            cu = np.copy(self.updated_labels[-1])
            cu[self.mc_indexes[self.res_iteration]] = true_mc_labels_converted
            self.updated_labels.append(cu)
        else:
            cu = np.copy(self.y_train)
            cu[self.mc_indexes[self.res_iteration]] = true_mc_labels_converted
            self.updated_labels.append(cu)

    def predict(self, x_valid, num_vecs, u_iteration):
        res = compute_diff(x_valid, num_vecs, self.U_per_digit[u_iteration])
        predictions = self.pred_converter(
            u_iteration, np.argmin(res, axis=1) + 1
        )

        return res, predictions
