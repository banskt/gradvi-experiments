import os
import subprocess
import tempfile
import pandas as pd
from dsc import dsc_io

import rpy2.robjects as robj
import rpy2.robjects.vectors as rvec
from rpy2.robjects.packages import importr 
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri
numpy2ri.activate()
from rpy2.robjects import pandas2ri
pandas2ri.activate()


'''
A Python wrapper for the dscquery in R
Brute force method which saves a temporary RDS file 
and loads it in Python.
See below for a rpy2 implementation,
which does not work with pkl files.
'''
def dscquery(dsc_outdir, targets,
             conditions = None,
             groups = None,
             verbose = True,
             sep = "::",
             dolr = "##"
            ):

    os_handle, \
        rds_file = tempfile.mkstemp(suffix = ".rds")
    rscript_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dscquery.R")

    def list_to_string(xlist):
        x = sep.join(xlist)
        x = x.replace("$", dolr)
        return x

    cmd  = ["Rscript",     rscript_file]
    cmd += ["--outdir",    dsc_outdir]
    cmd += ["--rdsfile",   rds_file]
    cmd += ["--separator", sep]
    cmd += ["--cmarker",   dolr]
    cmd += ["--targets",   list_to_string(targets)]
    if conditions is not None:
        cmd += ["--conditions", list_to_string(conditions)]
    if groups is not None:
        cmd += ["--groups",     list_to_string(groups)]

    process = subprocess.Popen(cmd,
                               stdout = subprocess.PIPE,
                               stderr = subprocess.PIPE
                              )
    res     = process.communicate()
    #print("OUTPUT ==>")
    print(res[0].decode('utf-8'))

    if len(res[1].decode('utf-8')) > 0:
        print("")
        print("ERROR ==>")
        print(res[1].decode('utf-8'))

    retcode = process.returncode
    dscout  = pd.DataFrame(dsc_io.load_rds(rds_file)) if retcode == 0 else None
    if os.path.exists(rds_file): os.remove(rds_file)
    return dscout


'''
A Python wrapper for the dscquery in R using rpy2.
DEPRECATED. 
** reticulate does not work in the R subprocess called from Python. **
Hence, it cannot load pkl files and throws error.
'''
def _dscquery(dsc_output, targets,
              conditions = None,
              verbose = True
             ):
    dscrutils    = importr('dscrutils')
    r_targets    = rvec.StrVector(targets)        if targets    is not None else robj.NULL
    r_conditions = rvec.StrVector(conditions)     if conditions is not None else robj.NULL

    dscoutr = dscrutils.dscquery(dsc_output, r_targets,
                                 conditions = r_conditions, 
                                 verbose = verbose)

    with localconverter(robj.default_converter + pandas2ri.converter):
        dscout = robj.conversion.rpy2py(dscoutr)

    return dscout
