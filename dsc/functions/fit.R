# This file contains simple implementations of various methods for
# fitting linear regression models; methods include ridge regression,
# the Lasso, the Elastic Net, and varbvs (variational inference for
# Bayesian variable selection).

# Fit a ridge regression model to the data, and estimate the penalty
# strength (i.e., the normal prior on the regression coefficients)
# using cross-validation. Input X should be an n x p numeric matrix,
# and input y should be a numeric vector of length n. Input "nfolds"
# is the number of folds used in the cross-validation. The return
# value is a list with four elements: (1) the fitted glmnet object,
# (2) the output of cv.glmnet, (3) the intercept at the penalty
# strength ("lambda") chosen by cross-validation, and (4) the
# regression coefficients at this value of lambda.
fit_ridge <- function (X, y, nfolds = 10)
  fit_lasso(X, y, nfolds, 0)

# Fit a Lasso model to the data, and estimate the penalty strength
# (lambda) using cross-validation. Input X should be an n x p numeric
# matrix, and input y should be a numeric vector of length n. Input
# "nfolds" is the number of folds used in the cross-validation. four
# elements: (1) the fitted glmnet object, (2) the output of cv.glmnet,
# (3) the intercept at the penalty strength ("lambda") chosen by
# cross-validation, and (4) the regression coefficients at this value
# of lambda.
fit_lasso <- function (X, y, nfolds = 10, alpha = 1, cvlambda = "min") {
  out.cv <- glmnet::cv.glmnet(X, y, alpha = alpha, nfolds = nfolds)
  fit    <- glmnet::glmnet(X, y, alpha = alpha, standardize = FALSE)
  cvs    <- if (cvlambda == "1se") out.cv$lambda.1se else out.cv$lambda.min
  b      <- as.vector(coef(fit, s = cvs))
  ypred  <- drop(b[1] + X %*% b[-1])
  sigma2 <- var(y - ypred)
  return(list(fit = fit, cv = out.cv, mu = b[1], beta = b[-1], sigma2 = sigma2))
}

# Fit an Elastic Net model to the data, and estimate the Elastic Net
# parameters (penalty strength, "lambda", and mixing parameter,
# "alpha") using cross-validation. Input X should be an n x p numeric
# matrix, and input y should be a numeric vector of length n. Input
# "nfolds" is the number of folds used in the cross-validation, and
# input "alpha" is the vector of candidate values of the Elastic Net
# mixing parameter. The return value is a list with five elements: (1)
# the fitted glmnet object, (2) the output of cv.glmnet, (3) the
# setting of alpha minimizing the mean cross-validation error, (4) the
# intercept at the penalty strength ("lambda") chosen by
# cross-validation, and (3) the regression coefficients at this value
# of lambda.
fit_elastic_net <- function (X, y, nfolds = 10, alpha = seq(0,1,0.05), cvlambda = "min") {
  n          <- nrow(X)
  foldid     <- rep_len(1:nfolds,n)
  out.cv     <- NULL
  cvm.min    <- Inf
  alpha.min  <- 1

  # Identify the setting of alpha that minimizes the mean
  # cross-validation error.
  for (i in alpha) {
    out <- glmnet::cv.glmnet(X, y, nfolds = nfolds, foldid = foldid, alpha = i)
    if (min(out$cvm) < cvm.min) {
      cvm.min    <- min(out$cvm)
      alpha.min  <- i
      out.cv     <- out
    }
  }

  # Fit the Elastic Net model using the chosen value of alpha.
  fit <- glmnet::glmnet(X, y, standardize = FALSE, alpha = alpha.min)
  cvs <- if (cvlambda == "1se") out.cv$lambda.1se else out.cv$lambda.min
  b   <- as.vector(coef(fit, s = cvs))
  return(list(fit = fit, cv = out.cv, alpha = alpha.min, mu = b[1],
              beta = b[-1]))
}

# Fit a "sum of single effects" (SuSiE) regression model to the
# provided data. The data are specified by inputs X and y; X should be
# an n x p numeric matrix, and y should be a numeric vector of length
# n. Note that we found that the prediction performance was more
# robust when setting estimate_prior_variance = FALSE.
fit_susie <- function (X, y, L = 20, scaled_prior_variance = 0.2) {
  fit <- susieR::susie(X, y, L = L, max_iter = 1000, standardize = FALSE,
                       scaled_prior_variance = scaled_prior_variance,
                       estimate_prior_variance = FALSE)
  b   <- as.vector(coef(fit))
  return(list(fit = fit, mu = b[1], beta = b[-1], pip = fit$pip))
}

# Compute a fully-factorized variational approximation for Bayesian
# variable selection in linear regression. Input X should be an n x p
# numeric matrix, and input y should be a numeric vector of length n.
# 
# In this implementation, candidate values of the prior inclusion
# probability (determined by "logodds") are provided, and the results
# are averaged over the settings, whereas the hyperparameters are
# automatically fitted separately for each logodds setting.
fit_varbvs <- function (X, y) {
  logodds <- seq(-log10(ncol(X)), 1, length.out = 40)
  fit     <- varbvs::varbvs(X, NULL, y, logodds = logodds, verbose = FALSE)
  b       <- as.vector(coef(fit)[,"averaged"])
  return(list(fit = fit, mu = b[1], beta = b[-1]))
}

# Compute a fully-factorized variational approximation for the
# Bayesian variable selection model with mixture-of-normals priors on
# the regression coefficients. The variances of the mixture components
# are chosen automatically based on the data. Input argument k
# controls the number of mixture components.
fit_varbvsmix <- function (X, y, k = 20) {
  fit <- varbvs::varbvsmix(X, NULL, y, k, verbose = FALSE)
  b   <- as.vector(coef(fit))
  return(list(fit = fit, mu = b[1], beta = b[-1]))
}

# Fit a Bayesian Lasso model to the data
fit_blasso <- function (X, y, niter = 1500, burn_in = 500) {
  out <- tempfile()
  fit <- suppressWarnings(BGLR::BGLR(y, ETA = list(list(X = X, model="BL", standardize = FALSE)),
                                     verbose = FALSE, nIter = niter, burnIn = burn_in,
                                     saveAt = out))
  b   <- c(fit$ETA[[1]]$b)
  return (list(fit = fit, mu = fit$mu, beta = b))
}

# Fit a BayesB model to the data
fit_bayesb <- function (X, y, niter = 1500, burn_in = 500) {
  out <- tempfile()
  fit <- suppressWarnings(BGLR::BGLR(y, ETA = list(list(X = X, model="BayesB", standardize = FALSE)),
                                     verbose = FALSE, nIter = niter, burnIn = burn_in,
                                     saveAt = out))
  b   <- c(fit$ETA[[1]]$b * fit$ETA[[1]]$d)
  return (list(fit = fit, mu = fit$mu, beta = b))
}

# Fit using coordinate descent algorithms for nonconvex penalized regression (ncvreg).
# Different penalty functions can be used, particularly:
#   1) smoothly clipped absolute deviation (SCAD) penalty
#   2) minimax concave penalty (MCP)
# The tuning parameter of MCP / SCAD penalty is gamma.
# By default, gamma = 3.7 for SCAD and gamma = 3 for MCP.
# If we do not want to select gamma, use gamma = switch(penalty, SCAD=3.7, 3)
#
fit_ncvreg <- function (X, y, penalty, nfolds = 10, gamma = seq(1, 5, length.out = 10)) {
  cve.min <- Inf
  # Iterate through all values of gamma to select the best fit
  for (g in gamma) {
    cvfit <- suppressWarnings(ncvreg::cv.ncvreg(X, y, penalty = penalty, gamma = g, nfolds = nfolds))
    if (cve.min > min(cvfit$cve)) {
      fit     <- cvfit
      cve.min <- min(fit$cve)
    }
  }
  b <- as.vector(coef(fit))
  return (list(fit = fit, mu = b[1], beta = b[-1]))
}


fit_scad <- function (X, y, nfolds = 10, gamma = seq(2.1, 5.3, length.out = 11)) {
  return (fit_ncvreg(X, y, "SCAD", nfolds = nfolds, gamma = gamma))
}


fit_mcp  <- function (X, y, nfolds = 10, gamma = seq(1.1, 4.9, length.out = 11)) {
  return (fit_ncvreg(X, y, "MCP", nfolds = nfolds, gamma = gamma))
}

# Perform nfolds cross-validation on a L0 regression model 
fit_l0learn <- function (X, y, nfolds = 10) {
  cvfit      <- L0Learn::L0Learn.cvfit(X, y, nFolds = nfolds)
  lambda.min <- cvfit$fit$lambda[[1]][which.min(cvfit$cvMeans[[1]])]
  b          <- as.vector(coef(cvfit, lambda = lambda.min))
  return (list(fit = cvfit$fit, mu = b[1], beta = b[-1]))
}

# Perform Genlasso trendfiltering
fit_genlasso_trendfilter <- function(y, order = 1, nfolds = 5, cvlambda = "1se") {
  out   <- genlasso::trendfilter(y, ord = order)
  cvout <- genlasso::cv.trendfilter(out, k = nfolds)
  cvlam <- if (cvlambda == "1se") cvout$lambda.1se else cvout$lambda.min
  cvidx <- if (cvlambda == "1se") cvout$i.1se else cvout$i.min
  b     <- coef(out, lambda = cvlam)
  ypred <- out$fit[, cvidx]
  return (list(fit = out, mu = 0, beta = as.vector(b$beta),
               cv = cvout, ypred = ypred, df = b$df))
}


# Fit "Mr.ASH" to the provided data, X and y.
# X should be an n x p numeric matrix, and y should be a vector of length p
# Mr.ASH is a variational empirical Bayes (VEB) method for multiple linear regression. 
# The fitting algorithms (locally) maximize the approximate marginal likelihood 
# (the "evidence lower bound", or ELBO) via coordinate-wise updates.
# Several flavors have been implemented in the DSC pipeline which
# requires different optional settings.
fit_mr_ash <- function (X, y,
                        max_iter = 2000, sa2 = NULL,
                        init_pi = NULL, init_beta = NULL, init_sigma2 = NULL,
                        update_pi = TRUE, update_sigma2 = TRUE, 
                        update_order = NULL,
                        intercept = TRUE,
                        tol = list(epstol = 1e-12, convtol = 1e-8)) {
  fit  <- suppressWarnings(mr.ash.alpha::mr.ash(X, y, 
                                                standardize = FALSE, intercept = intercept,
                                                max.iter = max_iter, sa2 = sa2,
                                                beta.init = init_beta, 
                                                update.pi = update_pi, pi = init_pi,
                                                update.sigma2 = update_sigma2, sigma2 = init_sigma2,
                                                update.order = update_order,
                                                tol = tol))
  return(list(fit = fit,mu = fit$intercept, beta = fit$beta))
}
