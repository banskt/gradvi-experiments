# This R script implements the "mr.ash" module for trendfiltering

if (scale_grid) {
    grid <- ((degree + 1) * 20)^2 * grid
}

out <- fit_mr_ash(X, as.vector(y), sa2 = grid,
                  intercept = FALSE,
                  init_pi = init_pi, init_beta = init_beta, init_sigma2 = init_sigma2)
out$ypred <- drop(X %*% as.vector(out$beta))
