# This python script implements the "gradvi" module with ash prior

from fit_gradvi import fit_ash_gradvi

model, mu, beta = fit_ash_gradvi(X, y, objtype, 
    ncomp = ncomp, sparsity = sparsity, skbase = skbase, binit = init_beta, s2init = init_sigma2, winit = init_mixcoef, 
    run_initialize = run_init, return_pip = False)
