using MKL;
using LinearAlgebra, Statistics, VMLS, FLoops
include("svdr.jl");
function TregsYLooCV(X, Y, λs; smo=0) # THE INPUTs ARE: (X,Y)-data, λs: a vector of reg. param. values, smo: order of smoothing.
    m, n = size(X)
    q = size(Y)[2]
    λs = reshape(λs, 1, length(λs))  # λs is a row vector of many reg. parameter values.
    x¯ = mean(X, dims=1)
    y¯ = mean(Y, dims=1)    # - mean of (X,Y)-data.
    Y = Y .- y¯
    X = X .- x¯              # - centering of (X,Y)-data.
    if smo > 0 # smo. the order of smoothing. 
        L = [speye(n); zeros(smo, n)]
        for i = 1:smo
            L = diff(L, dims=1)
        end
        X = X / L # L is a discrete derivative smoothing matrix of order "smo".
    end
    U, σ, V = svdr(X) # SVD of X & (σ, λ)-factors required for calc. of bcoefs and H.
    σ_plus_λs_over_σ = (σ .+ (λs ./ σ))
    H = (U .^ 2) * (σ ./ σ_plus_λs_over_σ) .+ 1 / m               # Simultaneous calc. of the leverage vectors for all λs.

    bcoefs = Vector(undef, q)
    minid = Vector(undef, q)
    press = Vector(undef, q)
    bλ = Vector(undef, q)
    λ = Vector(undef, q)
    hλ = Vector(undef, q)
    @floop begin
        for k = 1:q
            bcoefs[k] = V * ((U' * Y[:, k]) ./ σ_plus_λs_over_σ)       # Simultaneous calc. of the regression coeffs for all λs.
            # press   = sum(((Y[:,k].-X*bcoefs)./(1 .-H)).^2, dims=1)'; # The PRESS-values corresponding to all λs.
            press[k] = sum(((Y[:, k] .- U * ((σ .* (U' * Y[:, k])) ./ σ_plus_λs_over_σ)) ./ (1 .- H)) .^ 2, dims=1)' # The PRESS-values corresponding to all λs.
            if smo > 0
                bcoefs[k] = L \ bcoefs[k]
            end              # The X-regression coeffs in cases of smoothing (smo > 0).
            minid[k] = findmin(vec(press[k]))[2] # Find index of minimum press-value & identify corresponding λ-value, ...
            λ[k] = λs[minid[k]]
            bλ[k] = [y¯[k] .- x¯ * bcoefs[k][:, minid[k]]; bcoefs[k][:, minid[k]]]
            hλ[k] = H[:, minid[k]] # ...regression coeffs (bλ) and leverages (hλ).
        end
    end
    return press, minid, U, σ, V, H, bcoefs, λ, bλ, hλ  # The OUTPUT-parameters of the function.
end