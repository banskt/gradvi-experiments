# This R script implements the "mr.ash.init" module for trendfiltering

grid <- (((degree+1)*20)^((0:19)/20) - 1)^2
init_beta <- drop(Xinv %*% as.vector(yinit))
init_sigma2 <- var(as.vector(y) - as.vector(yinit))
out <- fit_mr_ash(X, as.vector(y),
                  sa2 = grid, intercept = FALSE,
                  init_pi = init_pi, init_beta = init_beta, init_sigma2 = init_sigma2)
out$ypred <- drop(X %*% as.vector(out$beta))
