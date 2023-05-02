# A DSC for evaluating prediction accuracy of 
# multiple linear regression methods in different scenarios.

DSC:
  R_libs:         glmnet, 
                  mr.ash.alpha,
                  genlasso
  python_modules: numpy,
                  gradvi
  lib_path:       functions
  exec_path:      modules/simulate,
                  modules/initialize,
                  modules/fit,
                  modules/predict,
                  modules/score
  #output:         /gpfs/commons/groups/knowles_lab/sbanerjee/sparse-regression/gradvi-experiments/trendfiltering_runtime
  output:         /gpfs/commons/groups/knowles_lab/sbanerjee/sparse-regression/gradvi-experiments/trendfiltering_runtime_lowmem_trial
  replicate:      1
  define:
    simulate:     changepoint_lowmem
    initialize:   gvma
    fit:          gradvi_compound, genlasso
    score:        tfmse, tfmae
  run:
    linreg_corr:  simulate * initialize * fit * score


# simulate modules
# ===================

changepoint:      changepoint.py
# Module for simulation of changepoint data.
# Input parameters and output data for all simulation designs.
#
# sfix: Number of knots in the changepoint
# degree: Degree of trendfiltering (provided as input, also required in output for next steps)
#
# signal: distribution of the coefficients for the changepoint coefficients
#    - "normal": sample from Gaussian(mean = 0, sd = 1)
#    - "gamma" : sample from Gamma distribution (k = 40, theta = 0.1) and multiply with random signs
#    - "fixed":  Use pre-defined value(s) of beta
#                bfix: sequence / float of predefined beta
#                (if sequence, length must be equal to number of non-zero coefficients).
#
# X: basis matrix for regression solution
# 
  bfix:    None
  signal:  "normal"
  n:       100, 1000, 10000
  strue:   0.6
  sfix:    10
  dtrue:   0
  $X:      H
  $Xinv:   Hinv
  $Xscale: Hscale
  $Xinvscale: Hinvscale
  $y:      y
  $ytest:  ytest
  $ytrue:  ytrue
  $beta:   beta
  $snr:    snr
  $degree: dtrue


changepoint_lowmem (changepoint): changepoint_lowmem.py
  n: 100000, 1000000 


# initialize with gvma
# ========================
gvma: moving_average_initialize.py
  y: $y
  $y_ma: y_smooth


# fit modules
# ===================
# All fit modules must have these inputs and outputs
# Extra inputs and outputs can be specified in 
# respective submodules.
fitR:
  y:          $y
  degree:     $degree
  $ypred:     out$ypred
  $model:     out

fitpy:
  y:          $y
  degree:     $degree
  $ypred:     ypred
  $model:     model


# Fit genlasso crossvalidation
genlasso (fitR):  genlasso_trendfiltering.R


# Fit Mr.ASH
# This is an abstract base class, which contains all default values.
# Several variations of Mr.ASH use this abstract base class (see below).
mr_ash_base (fitR): mr_ash_trendfiltering.R
  init_pi:       NULL
  init_beta:     NULL
  init_sigma2:   NULL
  update_pi:     TRUE
  update_sigma2: TRUE
  update_order:  NULL
  grid:          (2^((0:19)/20) - 1)^2
  scale_grid:    TRUE

# This is the default variant of Mr.ASH
# For runtime, we are not looking at Lasso initialization.
mr_ash (mr_ash_base): mr_ash_genlasso_trendfiltering.R
  X: $X
  Xinv: $Xinv
  yinit: $y_ma 

mr_ash_scaled (mr_ash_base): mr_ash_genlasso_trendfiltering.R
  X: $Xscale
  Xinv: $Xinvscale
  yinit: $y_ma
  
# GradVI methods
# Mr.Ash (unscaled) prior
#
gradvi_trendfiltering(fitpy): gradvi_trendfiltering.py
  Xinv: $Xinv
  s2init: None
  ncomp: 20
  sparsity: 0.99
  skbase: 2.0
  scale_grid: True
  scale_basis: False
  standardize_basis: False
  yinit: $y_ma


gradvi_direct(gradvi_trendfiltering):
  objtype: "direct"
  standardize_basis: True


gradvi_compound(gradvi_trendfiltering):
  objtype: "reparametrize"


# score modules
# =============
# A "score" module takes as input a vector of predicted outcomes and a
# vector of true outcomes, and outputs a summary statistic that can be
# used to evaluate accuracy of the predictions.

# Compute the mean squared error summarizing the differences between
# the predicted outcomes and the true outcomes.
tfmse: tfmse.R
  y:     $ytrue
  yest:  $ypred
  yinit: $y_ma
  $err:  err
  $init_err: init_err

# Compute the mean absolute error summarizing the differences between
# the predicted outcomes and the true outcomes.
tfmae: tfmae.R
  y:     $ytrue
  yest:  $ypred
  yinit: $y_ma
  $err:  err
  $init_err: init_err
