# This R script implements the "lasso_1se" module.
out <- fit_lasso(X, as.vector(y), cvlambda = "1se")

