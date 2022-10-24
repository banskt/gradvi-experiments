# This python script implements the "gradvi_direct" module.

from fit_gradvi import fit_ash_trendfiltering_gradvi

if yinit is not None:
    binit = np.dot(Xinv, yinit)
else:
    binit = None

model, mu, beta = fit_ash_trendfiltering_gradvi(X, y, objtype,
                    degree = degree, ncomp = ncomp, sparsity = sparsity, skbase = skbase,
                    binit = binit, s2init = s2init)
