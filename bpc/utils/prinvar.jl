using LinearAlgebra, VMLS, Statistics
include("svdr.jl")
function prinvar(X, A; B=minimum(size(X)))
    # Q, R, ids, vperm, ssEX, ni, U, s = prinvar(X, A);
    """
    "Greedy" algorithm (https://en.wikipedia.org/wiki/Greedy_algorithm)
    for extracting the most dominant (principal)variables (X-coulumns)
    w.r.t. explained X-variance.
    INPUTS:
    X - the datamatrix.
    A - the number of selected variables.
    B - number of PC's included in the voting (optional).
    OUTPUTS:
    Q   - orthonormal scores (associated with the selected variables).
    R   - corresponding loadings. NOTE: R(:,vperm) is upper triangular.
    ids - indices arranged in the order of the A selected variables.
    vperm - indices arranged in the order of the A selected- and all non-selected variables. NOTE: R(:,vperm) is upper triangular.
    ssEX - the partial variances explained by the selected variables.
    ni  - the norms of the (residual) selected variables before the score-normalization (Q).
    U   - the normalized PCA-scores.
    s   - singular values of the mean centered X
    """
    # INITIALIZATIONS:
    x¯ = mean(X, dims=1)
    X = X .- x¯     # Centering of the data matrix.
    U, s, ~, rk = svdr(X)                   # Essentially PCA of the centered datamatrix X.
    A = min(A, rk)
    ntol = 1e-10                            # Threshold defining a vanishing norm.
    m, n = size(X)                          # Data dimensions.
    ids = 0 * Array{Int64}(undef, A)         # Indices of the selected variables.
    Q = zeros(m, A)
    R = zeros(A, n)       # See Output-description above.
    ssEX = zeros(A, 1)                       # See Output-description above.
    ni = zeros(A, 1)                       # See Output-description above.
    vAll = zeros(1, n)                       # Vector for holding the votes for all variables.
    B = max(min(B, rk), 2)                 # At least 2 PC's must be included in the voting function.
    T = U[:, 1:B] .* s[1:B]'                # Non-normalized PCA-scores for implementing the voting.
    ssTX = sum(s .^ 2)                        # The total sum-of-squares.
    a = 1
    idn = 1:n                    # Book-keeping: a-th variable to be selected & idn - indices of candidate variables available for voting/selection.
    while a <= A
        Xidn = X[:, idn]
        vAll[idn] = sum((T' * Xidn) .^ 2, dims=1) ./ sum(Xidn .^ 2, dims=1) # Update the vAll-votes for the X-variables subject to selection.
        mx, si = findmax(vAll)
        si = si[2]     # Identify the variable accounting for the maximum variance.
        ni[a] = norm(X[:, si])                 # norm of the selected candiate variable.
        if ni[a] > ntol                            # Consider only candidate variables with non-vanishing norms.
            ids[a] = si                          # Index of the a-th selected variable.
            qa = X[:, si] ./ ni[a]              # Unit vector in the direction of the selected variable.
            R[a, idn] = (qa' * Xidn)                  # Deflation coefficients in the chosen (v) direction.
            ssEX[a] = 100 * mx / ssTX                 # Store fraction of explained variance by the chosen variable.
            X[:, idn] = Xidn - qa * R[[a], idn]        # Deflate X wrt the chosen variable (modified Gram-Schmidt step).
            Q[:, a] = qa                          # Store qa into Q.
            a = a + 1                         # Update a to prepare for the subsequent selection.
        end
        idn = setdiff(idn, si)                  # Eliminate selected/vanishing variable from future voting assignments.
        vAll[si] = 0   #X[:,si] = 0;             # Set the future votes of the selected/vanishing variable to 0.
    end
    vperm = [ids; setdiff((1:n)', ids)]           # Indices of the selected- and unselected variables.
    return Q, R, ids .- 1, vperm, ssEX, ni, U, s # Ids - 1 because Python is zero indexed 
end

