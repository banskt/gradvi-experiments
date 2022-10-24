# This R script implements the "mr.ash.lasso.init" module.

grid <- (((degree+1)*20)^((0:19)/20) - 1)^2
out <- fit_mr_ash(X, as.vector(y), sa2 = grid, 
                  intercept = FALSE,
                  init_pi = init_pi, init_beta = init_beta, init_sigma2 = init_sigma2)
