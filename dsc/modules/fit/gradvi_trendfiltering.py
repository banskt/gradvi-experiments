# This python script implements the "gradvi_direct" module.

from fit_gradvi import fit_ash_trendfiltering_gradvi

if scale_grid:
    skfactor = (degree + 1) * 20.0
else:
    skfactor = 1.0

model, mu, beta, ypred = fit_ash_trendfiltering_gradvi(y, objtype, degree = degree,
                    maxiter = maxiter, tol = 1e-8,
                    ncomp = ncomp, sparsity = sparsity, skbase = skbase, skfactor = skfactor,
                    yinit = yinit, s2init = s2init, run_initialize = False, # this is a separate initialization for theta
                    standardize_basis = standardize_basis, scale_basis = scale_basis, standardize = True,
                    return_mrash_elbo = get_mrash_elbo)
