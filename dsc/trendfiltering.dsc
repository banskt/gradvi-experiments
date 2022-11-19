# A DSC for evaluating prediction accuracy of 
# multiple linear regression methods in different scenarios.

DSC:
  R_libs:         MASS, 
                  glmnet, 
                  susieR,
                  varbvs >= 2.6-3,
                  mr.ash.alpha,
                  L0Learn,
                  BGLR,
                  ncvreg,
                  genlasso
  python_modules: numpy,
                  gradvi
  lib_path:       functions
  exec_path:      modules/simulate,
                  modules/fit,
                  modules/predict,
                  modules/score
  output:         /home/saikatbanerjee/scratch/work/gradvi-experiments/trendfiltering
  #output:         /home/saikatbanerjee/scratch/work/gradvi-experiments/trendfiltering_trial
  replicate:      10
  define:
    simulate:     changepoint
    initialize:   genlasso
    fit:          mr_ash, mr_ash_init,
                  mr_ash_scaled, mr_ash_scaled_init,
                  gradvi_direct, gradvi_direct_init,
                  gradvi_compound, gradvi_compound_init,
                  gradvi_compound_scaled, gradvi_compound_scaled_init
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
  n:       1024
  strue:   0.1
  sfix:    4, 8, 16
  dtrue:   0, 1, 2
  #sfix:    4
  #dtrue:  1
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


# initialize with genlasso
# ========================
genlasso:  genlasso_trendfiltering.R
  y: $y
  degree: $degree
  $tf_y: out$ypred
  $tf_model: out


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
mr_ash (mr_ash_base): mr_ash_trendfiltering.R
  X: $X

mr_ash_scaled (mr_ash_base): mr_ash_trendfiltering.R
  X: $Xscale
  
# This is Mr.Ash with Lasso initialization
mr_ash_init (mr_ash_base):  mr_ash_genlasso_trendfiltering.R
  X:     $X
  Xinv:  $Xinv
  yinit: $tf_y


mr_ash_scaled_init (mr_ash_base): mr_ash_genlasso_trendfiltering.R
  X:     $Xscale
  Xinv:  $Xinvscale
  yinit: $tf_y


# GradVI methods
# Mr.Ash (unscaled) prior
#
gradvi_trendfiltering(fitpy): gradvi_trendfiltering.py
  Xinv: $Xinv
  yinit: None
  s2init: None
  ncomp: 20
  sparsity: 0.9
  skbase: 2.0
  scale_grid: True
  scale_basis: False
  standardize_basis: False


gradvi_direct(gradvi_trendfiltering):
  objtype: "direct"
  standardize_basis: True


gradvi_direct_init(gradvi_trendfiltering):
  objtype: "direct"
  yinit: $tf_y
  standardize_basis: True


gradvi_compound(gradvi_trendfiltering):
  objtype: "reparametrize"


gradvi_compound_scaled(gradvi_trendfiltering):
  objtype: "reparametrize"
  standardize_basis: True


gradvi_compound_init(gradvi_trendfiltering):
  objtype: "reparametrize"
  yinit: $tf_y


gradvi_compound_scaled_init(gradvi_trendfiltering):
  objtype: "reparametrize"
  yinit: $tf_y
  standardize_basis: True


# score modules
# =============
# A "score" module takes as input a vector of predicted outcomes and a
# vector of true outcomes, and outputs a summary statistic that can be
# used to evaluate accuracy of the predictions.

# Compute the mean squared error summarizing the differences between
# the predicted outcomes and the true outcomes.
tfmse: mse.R
  y:    $ytrue
  yest: $ypred
  $err: err 

# Compute the mean absolute error summarizing the differences between
# the predicted outcomes and the true outcomes.
tfmae: mae.R
  y:    $ytrue
  yest: $ypred
  $err: err 
