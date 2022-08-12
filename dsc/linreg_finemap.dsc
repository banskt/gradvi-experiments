# A DSC for evaluating prediction accuracy of 
# multiple linear regression methods in different scenarios.

DSC:
  R_libs:         MASS, 
                  glmnet, 
                  susieR 
  python_modules: numpy,
                  gradvi
  lib_path:       functions
  exec_path:      modules/simulate,
                  modules/fit,
                  modules/predict,
                  modules/score
  output:         /home/saikatbanerjee/scratch/work/gradvi-experiments/linreg_finemap
  replicate:      5
  define:
    simulate:     blockdiag
    fit:          susie,
                  mr_ash, mr_ash_lasso_init,
                  gradvi_direct, gradvi_compound
  run: 
    linreg_corr:  simulate * fit


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
  dims:    R{list(c(n=500, p=10000))}
  sfix:    20
  bfix:    None
  sfrac:   None
  signal:  "normal"
  ntest:   0
  $X:      X
  $y:      y
  $Xtest:  Xtest
  $ytest:  ytest
  $n:      n
  $p:      p
  $s:      s
  $beta:   beta
  $se:     sigma

blockdiag(simparams): blockdiag.py
  pve:     0.6
  rholist: [0.95, 0.95]
  min_block_size: 200

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
  $pip:       out$pip

fitpy:
  X:          $X
  y:          $y
  $intercept: mu
  $beta_est:  beta
  $model:     model
  $pip:       pip

# Fit a "sum of single effects" (SuSiE) regression model.
susie (fitR):           susie.R

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
mr_ash_lasso_init (mr_ash_base):  mr_ash_lasso_init.R
  grid:          (2^((0:19)/20) - 1)^2

# GradVI methods
# Mr.Ash prior
gradvi_direct(fitpy): gradvi_ash_pip.py
  ncomp: 20
  sparsity: None
  skbase: 2.0
  objtype: "direct"


gradvi_compound(fitpy): gradvi_ash_pip.py
  ncomp: 20
  sparsity: None
  skbase: 2.0
  objtype: "reparametrize"
