include("ReturnCVClassified.jl")
using NPZ, MAT;

XtR = npzread("conv_x_train.npy")
XtE = npzread("conv_x_test.npy")
Ytrain_mine = npzread("y_train.npy")
Ghat, GhatCV, Gpred = returnCVClassified(XtR, Ytrain_mine, XtE, 10, true)

vars = matread("C:\\Users\\amira\\Documents\\Python_Projects\\MasterOppgave\\imageclassification_proj\\lstsq_gpu\\data\\mnistdata.mat"); # Xtrain = vars["Xtrain"][:,1:end]; # Xtest  = vars["Xtest"][:,1:end];
training = vars["training"];
Gtrain = Int.(vars["Gtrain"]); # Training data (in 28x28 image format) and -labels.
testing = vars["testing"];
Gtest = Int.(vars["Gtest"]);  # Test data (in 28x28 image format) and -labels.
m1 = size(training, 1);
m2 = size(testing, 1);
ng = maximum(Gtrain); # The number samples (m1 & m2) and groups (ng).
Iz = Int.(eye(ng)); # Identity matrix for generating the one-hot (dummy) encoding of the training- and test labels.
Ytrain = Iz[Gtrain, :][:, 1, :];