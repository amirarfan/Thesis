function makeFilters(images, Pos, fs, T)
    # Inputs:
    # images - the images to be binarized and divided into filters
    # Pos - 3d array defining sub-image regions (for filtermaking)
    # fs - filter size (fs x fs) in pixels
    # Output:
    # Filters - an array of convolution filters
    nR = size(images, 1)
    nFs = size(Pos, 3)
    Filters = zeros(nR * nFs, fs, fs)
    indeks = 0
    for ii = 1:nR
        im = images[ii, :, :]
        for jj = 1:nFs
            Filters[indeks+jj,:,:] = 2 .*(im[Pos[:,1,jj],Pos[:,2,jj]] .> T ) .- 1
        end
        indeks = indeks + nFs
    end
    return Filters
end
