# This python script implements the "gradvi_direct" module.

from fit_gradvi import fit_ash_gradvi

model, mu, beta, pip = fit_ash_gradvi(X, y, objtype, 
    ncomp = ncomp, sparsity = sparsity, skbase = skbase, return_pip = True)
