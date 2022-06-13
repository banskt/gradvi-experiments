# This R script implements the "mr.ash" module.
out <- fit_mr_ash(X, as.vector(y), 
                  sa2 = grid, 
                  init_pi = init_pi, init_beta = init_beta, init_sigma2 = init_sigma2,
                  update_pi = update_pi, update_sigma2 = update_sigma2,
                  update_order = update_order)

