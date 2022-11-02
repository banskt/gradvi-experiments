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
  replicate:      10
  define:
    simulate:     changepoint
    initialize:   genlasso
    fit:          mr_ash, mr_ash_lasso_init,
                  gradvi_direct, gradvi_compound,
                  gradvi_direct_init, gradvi_compound_init
    predict:      predict_linear
    score:        mse, coef_mse
  run: 
    linreg_corr:  simulate * initialize * fit * predict * score


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
  order:   0, 1, 2
  $X:      H
  $Xinv:   Hinv
  $y:      y
  $ytest:  ytest
  $ytrue:  ytrue
  $beta:   beta
  $snr:    snr
  $degree: order


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
  X:          $X
  y:          $y
  degree:     $degree
  $intercept: out$mu
  $beta_est:  out$beta
  $model:     out

fitpy:
  X:          $X
  y:          $y
  degree:     $degree
  $intercept: mu
  $beta_est:  beta
  $model:     model

# Fit a ridge regression model using glmnet. The penalty strength
# (i.e., the normal prior on the coefficients) is estimated using
# cross-validation.
ridge (fitR):           ridge.R
  
# Fit a Lasso model using glmnet. The penalty strength ("lambda") is
# estimated via cross-validation.
lasso (fitR):           lasso.R
lasso_1se (fitR):       lasso_1se.R

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

# This is the default variant of Mr.ASH
mr_ash (mr_ash_base): mr_ash_trendfiltering.R

# This is Mr.Ash with Lasso initialization
mr_ash_lasso_init (mr_ash_base):  mr_ash_genlasso_trendfiltering.R
  Xinv:          $Xinv
  yinit:         $tf_y


# GradVI methods
# Mr.Ash prior
#
gradvi_trendfiltering(fitpy): gradvi_trendfiltering.py
  Xinv: $Xinv
  yinit: None
  s2init: None
  ncomp: 20
  sparsity: 0.9
  skbase: 20.0


gradvi_direct(gradvi_trendfiltering):
  objtype: "direct"


gradvi_compound(gradvi_trendfiltering):
  objtype: "reparametrize"


gradvi_direct_init(gradvi_trendfiltering):
  objtype: "direct"
  yinit: $tf_y


gradvi_compound_init(gradvi_trendfiltering):
  objtype: "reparametrize"
  yinit: $tf_y


# predict modules
# ===============
# A "predict" module takes as input a fitted model (or the parameters
# of this fitted model) and an n x p matrix of observations, X, and
# returns a vector of length n containing the outcomes predicted by
# the fitted model.

# Predict outcomes from a fitted linear regression model.
predict_linear: predict_linear.R
  X:         $X
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
