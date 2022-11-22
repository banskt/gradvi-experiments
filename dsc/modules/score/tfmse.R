# This R script implements the "tfmse" module in the trendfiltering DSC.
err <- mse(y,yest)
init_err <- mse(y,yinit)
