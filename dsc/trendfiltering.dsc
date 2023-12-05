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
  output:         /gpfs/commons/groups/knowles_lab/sbanerjee/sparse-regression/gradvi-experiments/trendfiltering
  replicate:      50
  define:
    simulate:     changepoint
    initialize:   gvma
    fit:          mr_ash_init,
                  mr_ash_scaled_init,
                  gradvi_direct_scaled_init,
                  gradvi_compound_init,
                  gradvi_compound_scaled_init
    score:        tfmse, tfmae
  run: 
    linreg_corr:  simulate * initialize * fit * score


# simulate modules
# ===================

changepoint:      changepoint_lowmem.py
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
# X: basis matrix for regression solution. Return only if lowmem is False. 
# If lowmem is true, then X is not created, instead a low memory 2 x 2 dummy matrix is returned.
# For large P, we cannot create the X matrix due to memory constraints.
# 
  bfix:    None
  signal:  "normal"
  n:       4096
  snr:     5.0
  sfix:    10
  dtrue:   0
  lowmem:  False
  $X:      H
  $Xinv:   Hinv
  $Xscale: Hscale
  $Xinvscale: Hinvscale
  $y:      y
  $ytest:  ytest
  $ytrue:  ytrue
  $beta:   beta
  $strue:  std
  $degree: dtrue


# initialize with genlasso
# ========================
genlasso:  genlasso_trendfiltering.R
  y: $y
  degree: $degree
  $y_init: out$ypred
  $tf_model: out


# initialize with moving average
# ==============================
gvma: moving_average_initialize.py
  y: $y
  $y_init: y_smooth


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
mr_ash (mr_ash_base):
  X: $X

mr_ash_scaled (mr_ash_base):
  X: $Xscale

mr_ash_init (mr_ash_base): mr_ash_trendfiltering_init.R
  X: $X
  Xinv: $Xinv
  yinit: $y_init

mr_ash_scaled_init (mr_ash_base): mr_ash_trendfiltering_init.R
  X: $Xscale
  Xinv: $Xinvscale
  yinit: $y_init


# GradVI methods
#
gradvi_trendfiltering(fitpy): gradvi_trendfiltering.py
  yinit: None
  s2init: None
  ncomp: 20
  sparsity: 0.9
  skbase: 2.0
  scale_grid: True
  scale_basis: False
  standardize_basis: False
  maxiter: 2000
  get_mrash_elbo: True


# Do not run this in pipeline, 
# numerical inversion is costly for unscaled X
gradvi_direct(gradvi_trendfiltering):
  objtype: "direct"


gradvi_direct_init(gradvi_trendfiltering):
  objtype: "direct"
  yinit: $y_init


gradvi_direct_scaled(gradvi_trendfiltering):
  objtype: "direct"
  standardize_basis: True


gradvi_direct_scaled_init(gradvi_trendfiltering):
  objtype: "direct"
  yinit: $y_init
  standardize_basis: True


gradvi_compound(gradvi_trendfiltering):
  objtype: "reparametrize"


gradvi_compound_init(gradvi_trendfiltering):
  objtype: "reparametrize"
  yinit: $y_init


gradvi_compound_scaled(gradvi_trendfiltering):
  objtype: "reparametrize"
  standardize_basis: True


gradvi_compound_scaled_init(gradvi_trendfiltering):
  objtype: "reparametrize"
  yinit: $y_init
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
