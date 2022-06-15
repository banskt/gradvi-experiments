# This R script implements the "mr.ash.lasso.init" module.

lasso_res   <- fit_lasso(X, as.vector(y), cvlambda = "1se")
init_beta   <- lasso_res$beta
init_sigma2 <- var(as.vector(y) - drop(lasso_res$mu + X %*% init_beta))
out <- fit_mr_ash(X, as.vector(y),
                  sa2 = grid,
                  init_pi = init_pi, init_beta = init_beta, init_sigma2 = init_sigma2)
