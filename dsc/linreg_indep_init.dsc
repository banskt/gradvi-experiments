# A DSC for evaluating prediction accuracy of 
# multiple linear regression methods in different scenarios.

# A DSC for evaluating prediction accuracy of multiple linear regression
# methods in different scenarios.
# This is designed to reproduce the results of the manuscript of Mr. Ash by Kim, Wang, Carbonetto and Stephens
DSC:
  R_libs:         MASS, 
                  glmnet, 
                  susieR, 
                  varbvs >= 2.6-3,
                  mr.ash.alpha,
                  L0Learn,
                  BGLR,
                  ncvreg
  python_modules: numpy,
                  gradvi
  lib_path:       functions
  exec_path:      modules/simulate,
                  modules/fit,
                  modules/predict,
                  modules/score
  output:         /home/saikatbanerjee/scratch/work/gradvi-experiments/linreg_indep_init
  replicate:      1
  define:
    simulate:     equicorrgauss
    initialize:   lasso
    fit:          mr_ash, mr_ash_lasso_init,
                  gradvi_direct, gradvi_compound,
                  gradvi_direct_lasso_init, gradvi_compound_lasso_init
    predict:      predict_linear
    score:        mse, coef_mse
  run: 
    linreg:       simulate * initialize * fit * predict * score


# simulate modules
# ===================

simparams:
# This is an abstract module for simulation.
# Input parameters and output data for all simulation designs.
#
# sfix:  Number of predictors with non-zero coefficients.
#        If sfix is not set / None, then the number is calculated dynamically from sfrac.
# sfrac: Fraction of predictors which have non-zero coefficients
# signal: distribution of the coefficients for the non-zero predictors
#    - "normal": sample from Gaussian(mean = 0, sd = 1)
#    - "gamma" : sample from Gamma distribution (k = 40, theta = 0.1) and multiply with random signs
#    - "fixed":  Use pre-defined value(s) of beta
#                bfix: sequence / float of predefined beta
#                (if sequence, length must be equal to number of non-zero coefficients).
# pve: proportion of variance explained (required for equicorrgauss.py)
  #dims:    R{list(c(n=500, p=10000))}
  #sfix:    2, 5, 10, 20
  dims:    R{list(c(n=500, p=1000))}
  sfix:    5
  bfix:    None
  sfrac:   None
  signal:  "normal"
  ntest:   1000
  $X:      X
  $y:      y
  $Xtest:  Xtest
  $ytest:  ytest
  $n:      n
  $p:      p
  $s:      s
  $beta:   beta
  $se:     sigma

equicorrgauss(simparams): equicorrgauss.py
  #pve:     0.4, 0.6, 0.8
  pve:     0.6
  rho:     0.0

# initialize with lasso
# =====================
# Fit a Lasso model using glmnet. The penalty strength ("lambda") is
# estimated via cross-validation.
lasso: lasso.R
  X:               $X
  y:               $y
  $init_intercept: out$mu
  $init_beta:      out$beta
  $init_sigma2:    out$sigma2
  $model_init:     out

# fit modules
# ===================
# All fit modules must have these inputs and outputs
# Extra inputs and outputs can be specified in 
# respective submodules.
fitR:
  X:          $X
  y:          $y
  $intercept: out$mu
  $beta_est:  out$beta
  $model:     out

fitpy:
  X:          $X
  y:          $y
  $intercept: mu
  $beta_est:  beta
  $model:     model

# Fit Mr.ASH
# This is an abstract base class, which contains all default values.
# Several variations of Mr.ASH use this abstract base class (see below).
mr_ash_base (fitR):     mr_ash.R
  grid:          NULL
  init_pi:       NULL
  init_beta:     NULL
  init_sigma2:   NULL
  update_pi:     TRUE
  update_sigma2: TRUE
  update_order:  NULL

# This is the default variant of Mr.ASH
mr_ash (mr_ash_base):
  grid:          (2^((0:19)/20) - 1)^2

# This is Mr.Ash with Lasso initialization
mr_ash_lasso_init (mr_ash_base):
  grid:          (2^((0:19)/20) - 1)^2
  init_beta:     $init_beta
  init_sigma2:   $init_sigma2


# GradVI methods
# Mr.Ash prior
gradvi_base(fitpy): gradvi_ash.py
  ncomp: 20
  sparsity: None
  skbase: 2.0
  init_beta: None
  init_sigma2: None
  init_mixcoef: None
  run_init: False

gradvi_direct(gradvi_base):
  objtype: "direct"

gradvi_compound(gradvi_base):
  objtype: "reparametrize"

gradvi_direct_lasso_init(gradvi_base):
  objtype:      "direct"
  init_beta:    $init_beta
  init_sigma2:  $init_sigma2
  run_init:     True

gradvi_compound_lasso_init(gradvi_base):
  objtype:      "reparametrize"
  init_beta:    $init_beta
  init_sigma2:  $init_sigma2
  run_init:     True


# predict modules
# ===============
# A "predict" module takes as input a fitted model (or the parameters
# of this fitted model) and an n x p matrix of observations, X, and
# returns a vector of length n containing the outcomes predicted by
# the fitted model.

# Predict outcomes from a fitted linear regression model.
predict_linear: predict_linear.R
  X:         $Xtest
  intercept: $intercept
  beta_est:  $beta_est
  $yest:     y   


# score modules
# =============
# A "score" module takes as input a vector of predicted outcomes and a
# vector of true outcomes, and outputs a summary statistic that can be
# used to evaluate accuracy of the predictions.

# Compute the mean squared error summarizing the differences between
# the predicted outcomes and the true outcomes.
mse: mse.R
  y:    $ytest
  yest: $yest
  $err: err 

# Compute the mean absolute error summarizing the differences between
# the predicted outcomes and the true outcomes.
mae: mae.R
  y:    $ytest
  yest: $yest
  $err: err 

coef_mse: coef_mse.R
  beta:     $beta
  beta_est: $beta_est
  $err: err

coef_mae: coef_mae.R
  beta:     $beta
  beta_est: $beta_est
  $err: err
