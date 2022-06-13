# This R script implements the "elastic_net_1se" module.
out <- fit_elastic_net(X, as.vector(y), cvlambda = "1se")

