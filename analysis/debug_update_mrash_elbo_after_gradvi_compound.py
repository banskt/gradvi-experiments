#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pickle
import sys
import os
import dsc
from dsc.query_engine import Query_Processor as dscQP
from dsc import dsc_io

import matplotlib.pyplot as plt
from pymir import mpl_stylesheet
from pymir import mpl_utils
mpl_stylesheet.banskt_presentation(splinecolor = 'black', dpi = 300)

from mrashpen.inference.mrash_wrapR import MrASHR

import utils


def stratify_dfcol(df, colname, value):
    #return pd_utils.select_dfrows(df, [f"$({colname}) == {value}"])
    return df.loc[df[colname] == value]

def stratify_dfcols(df, condition_list):
    for (colname, value) in condition_list:
        df = stratify_dfcol(df, colname, value)
    return df

def stratify_dfcols_in_list(df, colname, values):
    return df.loc[df[colname].isin(values)]

def read_result(dfrow):
        
    def read_dsc_pickle(fprefix):
        fname = os.path.join(dsc_output, f"{fprefix}.pkl")
        return dsc_io.load_dsc(fname)
            
    # Input data
    data_fprefix = dfrow['simulate.output.file']
    gv_fprefix   = dfrow['fit.output.file']
    data = read_dsc_pickle(data_fprefix)
    gv   = read_dsc_pickle(gv_fprefix)
    return data, gv

def get_mse_from_dscout(df, sfix, pve, dsc, method):
    dfrows = stratify_dfcols(df, 
                             [("simulate.sfix", sfix), 
                              ("simulate.pve", pve), 
                              ("DSC", dsc), 
                              ("fit", method)])
    mse = dfrows[~dfrows['mse.err'].isnull()]['mse.err'].values
    if len(mse) == 1:
        return mse[0]
    else:
        print ("Error fetching value")
        return mse[0]

def get_msedf(elbodf, dscdf):
    methods = [x[:-7] for x in list(elbodf.columns) if x.endswith("_status")]
    msedict = elbodf.to_dict()
    ikeys = list(msedict['simulate.sfix'].keys())
    for method in methods:
        for i in ikeys:
            mse = get_mse_from_dscout(dscout, 
                          msedict['simulate.sfix'][i], 
                          msedict['simulate.pve'][i],
                          msedict['DSC'][i], 
                          method)
            msedict[method][i] = mse 
    return pd.DataFrame.from_dict(msedict)

def get_dict_index(dict1, dfrow):
    sfixidx = [i for i, val in dict1['simulate.sfix'].items() if val == dfrow['simulate.sfix']]
    pveidx  = [i for i, val in dict1['simulate.pve'].items() if val == dfrow['simulate.pve']]
    dscidx  = [i for i, val in dict1['DSC'].items() if val == dfrow['DSC']]
    tgtidx  = list(set(dscidx).intersection(set(sfixidx).intersection(set(pveidx))))[0]
    return tgtidx

def get_elbo_mse(dfrow):
    data, gv = read_result(dfrow)
    cavi  = MrASHR(option = "rds", debug = False)
    sk    = gv['model']['prior']['sk']
    winit = gv['model']['prior']['w']
    cavi.fit(data['X'], data['y'], sk, winit = winit, binit = gv['beta_est'], maxiter = 500)
    elbo  = cavi.elbo_path[-1]
    mse   = np.mean(np.square(data['ytest'] - (np.dot(data['Xtest'], cavi.coef[:, 0]) + cavi.fitobj['intercept'])))
    return elbo, mse



dsc_output = "/home/saikatbanerjee/scratch/work/gradvi-experiments/linreg_corr_init"
dsc_fname  = os.path.basename(os.path.normpath(dsc_output))
db = os.path.join(dsc_output, dsc_fname + ".db")
elbopkl   = os.path.join("../dsc/results", dsc_fname + "_elbo.pkl")
msepkl    = os.path.join("../dsc/results", dsc_fname + "_mse.pkl")
dscoutpkl = os.path.join("../dsc/results", dsc_fname + "_dscout.pkl")
dscout    = pd.read_pickle(dscoutpkl)

refresh_pickle = False

print(db)
plotprefix = "mrash_elbo_update_after_gradvi_compound"


target = ["simulate", "simulate.sfix", "simulate.pve", "simulate.se", "simulate.dims", "fit"]
#condition = ["simulate.sfix == 2", "simulate.signal == 'normal'", "simulate.dims == '(50, 200)'"]
#groups = ["fit_cpt:"]
condition = [""]


qp = dscQP(db, target, condition)
df = qp.output_table

if refresh_pickle:
    elbodf = get_elbodf(df)
    elbodf.to_pickle(elbopkl)
else:
    elbodf = pd.read_pickle(elbopkl)


msedf = get_msedf(elbodf, dscout)


msedict = msedf.to_dict()
elbodict = elbodf.to_dict()
msedict['gradvi_compound_mrash'] = dict()
elbodict['gradvi_compound_mrash'] = dict()

gvdf = stratify_dfcol(df, 'fit', 'gradvi_compound')

for idx in range(gvdf.shape[0]):
#for idx in range(2):
    dfrow = gvdf.iloc[idx]
    elbo, mse = get_elbo_mse(dfrow)
    elboidx = get_dict_index(elbodict, dfrow)
    print("Running row", idx + 1, " out of ", gvdf.shape[0])
    print("ELBO index", elboidx)
    elbodict['gradvi_compound_mrash'][elboidx] = elbo
    mseidx  = get_dict_index(msedict, dfrow)
    print("MSE index", mseidx)
    msedict['gradvi_compound_mrash'][mseidx] = mse


new_msedf = pd.DataFrame.from_dict(msedict)
new_elbodf = pd.DataFrame.from_dict(elbodict)
new_elbodf.to_pickle(elbopkl)
new_msedf.to_pickle(msepkl)
