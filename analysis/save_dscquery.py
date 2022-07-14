import dscrutils2py as dscrutils
import os
import pandas as pd
import pickle
import argparse
import sys

def parse_args():

    parser = argparse.ArgumentParser(description='Save dscout to pickle format.')

    parser.add_argument('--out',
                        type=str,
                        dest='outfile',
                        metavar='FILE',
                        required=True,
                        help='Name of output file')

    parser.add_argument('--dsc',
                        type=str,
                        dest='dscdir',
                        metavar='FILE',
                        required=True,
                        help='Name of DSC output directory')

    try:
        options = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    return options

args = parse_args()

dscdir  = os.path.normpath(args.dscdir)
#dscdir = "/home/saikatbanerjee/scratch/work/gradvi-experiments/linreg_indep"
outfile = os.path.normpath(args.outfile)
#outfile = "/home/saikatbanerjee/work/sparse-regression/gradvi-experiments/dsc/results/linreg_indep_dscout.pkl"

targets = ["simulate", "simulate.dims", "simulate.se", "simulate.rholist", "simulate.sfix", "simulate.pve", 
           "fit", "fit.DSC_TIME", "mse.err", "coef_mse.err"]

if os.path.isdir(os.path.dirname(outfile)):
    dscout = dscrutils.dscquery(os.path.realpath(dscdir), targets)
    dscout.to_pickle(outfile)
else:
    print ("No such file or directory: {:s}".format(os.path.dirname(outfile)))

## one lines for copy-paste
## targets = ["simulate", "simulate.dims", "simulate.se", "simulate.rho", "simulate.sfix", "simulate.pve", "fit", "fit.DSC_TIME", "mse.err", "coef_mse.err"]
## dscrutils.dscquery(dscdir, targets).to_pickle(outfile)
