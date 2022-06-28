import dscrutils2py as dscrutils
import os
import pandas as pd
import pickle

outdir  = "/home/saikatbanerjee/scratch/work/gradvi-experiments/linreg_indep"
outfile = "/home/saikatbanerjee/work/sparse-regression/gradvi-experiments/dsc/results/linreg_indep_dscout.pkl"

targets = ["simulate", "simulate.dims", "simulate.se", "simulate.rho", "simulate.sfix", "simulate.pve", 
           "fit", "fit.DSC_TIME", "mse.err", "coef_mse.err"]

dscout = dscrutils.dscquery(os.path.realpath(outdir), targets)
dscout.to_pickle(outfile)
