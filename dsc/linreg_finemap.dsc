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

blockdiag(simparams): blockdiag.py
  pve:     0.6
  rholist: [0.9, 0.9, 0.9]
  min_block_size: 1000

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
