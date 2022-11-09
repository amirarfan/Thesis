function returnCVClassified(XtE, bλ, ng)
    m2 = size(XtE, 1)
    Ypred = zeros(m2, ng)
    for k = 1:ng
        Ypred[:, k] = bλ[k, 1] .+ XtE * bλ[k, 2:end] # The testset predictions
    end
    # The resulting predicted groups:
    Gpred = getindex.(argmax(Ypred, dims=2), 2)
    return Gpred, Ypred
end