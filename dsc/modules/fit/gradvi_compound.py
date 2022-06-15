# This python script implements the "gradvi_direct" module.

from fit_gradvi import fit_mrash_gradvi_compound

model, mu, beta = fit_mrash_gradvi_compound(X, y, 
                    ncomp = ncomp, sparsity = sparsity, skbase = skbase)
