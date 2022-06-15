# This python script implements the "gradvi_direct" module.

from fit_gradvi import fit_mrash_gradvi_direct

model, mu, beta = fit_mrash_gradvi_direct(X, y, 
                    ncomp = ncomp, sparsity = sparsity, skbase = skbase)
