# %%

# Load Relevant Utilities and Functions
import sys

from sklearn.metrics import accuracy_score

sys.path.append(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\utils"
)
sys.path.append(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\LooCV"
)
import numpy as np
import scipy.io as sio
from convolve_filts import convolve_filts
from julia.api import Julia

Julia()
from julia import Main as jl

cv_prediction = jl.include(r"..\..\..\LooCV\ReturnCVPredicted.jl")

# %%

# Load Test - Data and Filters
data = sio.loadmat(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\data\mnistdata.mat"
)
x_test = data["testing"]
y_test = np.concatenate(data["Gtest"])

filters = np.load(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\MNIST_Results\global\prestored_models\filters\filters_cks_61700.npy"
)

# %%

# Convolve filters with Test-Data

x_te_convolved = convolve_filts(x_test, None, filters, bsize=32)

# Stack original and convolved filters together

x_te_convolved_orig = np.hstack(
    (x_test.reshape(x_test.shape[0], -1), x_te_convolved)
)

# Load Regularized Regression Model

b_lambda = np.load(
    r"C:\Users\amira\Documents\Python_Projects\MasterOppgave\Thesis\bpc\MNIST_Results\global\prestored_models\cks_61700\cks_61700.npy"
)
# %%

# Perform Prediction

preds, ys = cv_prediction(x_te_convolved_orig, b_lambda, 10)
preds = np.concatenate(preds)

# %%
# Print Accuracy on Test-set

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, preds))
