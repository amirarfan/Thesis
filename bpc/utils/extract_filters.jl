using NPZ
include("makeFilters.jl")

function extractFilters(training, filter_size, num_filters_per_image=5, T=110)
    fs = filter_size   # Filtersize (fs x fs)
    np = size(training, 2)   # Number of pixels in each image is np*np.
    ids = 1:np # Indices for the vertical & horizontal pixel positions
    cmarg1 = ceil((np - fs) / 2)
    cmarg2 = ceil((np - fs) / 2) # Center-margins
    nFs = num_filters_per_image    # Number of filters derived from each image.
    Pos = Int.(zeros(fs, 2, nFs)) # Sub-image pixel positions.
    Pos[:, :, 1] = [ids[1:fs]' ids[1:fs]']
    Pos[:, :, 2] = [ids[1:fs]' ids[end-fs+1:end]']
    Pos[:, :, 3] = [ids[end-fs+1:end]' ids[1:fs]']
    Pos[:, :, 4] = [ids[end-fs+1:end]' ids[end-fs+1:end]']
    Pos[:, :, 5] = [ids[end-fs+1:end]' .- cmarg1 ids[end-fs+1:end]' .- cmarg2]

    # ---------------------------------------------------------
    # Extract ("binary") filters from randomly selected training-images of each group:
    filts = makeFilters(training, Pos, fs, T)
    return filts
end