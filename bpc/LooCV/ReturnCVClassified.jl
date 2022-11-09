using MKL;
using FLoops;
include("TregsYLooCV.jl")
function returnCVClassified(XtR, Ytrain, XtE, ng, compute_test_preds)
    low = -2
    high = 12
    nλ = 101    # Range of reg. parameter values.
    λs = 10 .^ (range(low, high, length=nλ))' # Regularization parameter values.
    press, minid, U, σ, V, H, bcoefs, λ, bλ, hλ = TregsYLooCV(XtR, Ytrain, λs)
    m1 = size(XtR, 1)
    # ---------------------------------------------------------
    # Compute fitted values (Yhat) and Cross-validated predictions (YhatCV) from the minimum CV-error models based on TregsYLooCV:
    Yhat = zeros(m1, ng)
    YhatCV = zeros(m1, ng)

    @floop for k = 1:ng
        Yhat[:, k] = bλ[k][1] .+ XtR * bλ[k][2:end] # The fitted values.
        YhatCV[:, k] = (Yhat[:, k] - hλ[k] .* Ytrain[:, k]) ./ (1 .- hλ[k]) # The CV predictions.
    end
    # The resulting predicted groups:
    Ghat = getindex.(argmax(Yhat, dims=2), 2)
    GhatCV = getindex.(argmax(YhatCV, dims=2), 2)


    if compute_test_preds
        m2 = size(XtE, 1)
        Ypred = zeros(m2, ng)
        @floop for k = 1:ng
            Ypred[:, k] = bλ[k][1] .+ XtE * bλ[k][2:end] # The testset predictions
        end
        # The resulting predicted groups:
        Gpred = getindex.(argmax(Ypred, dims=2), 2)
        return Ghat, GhatCV, Gpred, Yhat, YhatCV, Ypred
    end

    return Ghat, GhatCV, bλ, Yhat, YhatCV 
end