## python script UUID: 33870585

import sys, os, tempfile, timeit, pickle, inspect
DSC_07FAB3F9 = dict()
from dsc.dsc_io import load_dsc as __load_dsc__, source_dirs as __source_dirs__
DSC_07FAB3F9 = __load_dsc__(['/home/saikatbanerjee/scratch/work/gradvi-experiments/trendfiltering/changepoint/changepoint_28.pkl','/home/saikatbanerjee/scratch/work/gradvi-experiments/trendfiltering/genlasso/changepoint_28_genlasso_1.rds'])
DSC_REPLICATE = DSC_07FAB3F9["DSC_DEBUG"]["replicate"]
#DSC_SEED = DSC_07FAB3F9["DSC_DEBUG"]["seed"] + 27
DSC_SEED = 100
X = DSC_07FAB3F9['X']
Xinv = DSC_07FAB3F9['Xinv']
degree = DSC_07FAB3F9['degree']
y = DSC_07FAB3F9['y']
yinit = DSC_07FAB3F9['tf_y']
for name, func in __source_dirs__(['functions']):
    globals()[name] = func
s2init = None
ncomp = 20
sparsity = 0.9
skbase = 20
objtype = "direct"
TIC_07FAB3F9 = timeit.default_timer()
DSC_SEED += DSC_REPLICATE
import random
#random.seed(DSC_SEED)
try:
	import numpy
	#numpy.random.seed(DSC_SEED)
except Exception:
	pass

## BEGIN DSC CORE
from fit_gradvi import fit_ash_trendfiltering_gradvi
if yinit is not None:
    binit = np.dot(Xinv, yinit)
    run_init = True
else:
    binit = None
    run_init = False
model, mu, beta = fit_ash_trendfiltering_gradvi(X, y, objtype,
                    degree = degree, ncomp = ncomp, sparsity = sparsity, skbase = skbase,
                    binit = binit, s2init = s2init, run_initialize = run_init)
## END DSC CORE

pickle.dump({"intercept": mu, "beta_est": beta, "model": model, 'DSC_DEBUG': dict([('time', timeit.default_timer() - TIC_07FAB3F9), ('script', inspect.getsource(inspect.getmodule(inspect.currentframe()))), ('replicate', DSC_REPLICATE), ('seed', DSC_SEED)])}, open('/home/saikatbanerjee/scratch/work/gradvi-experiments/trendfiltering/gradvi_direct_init/changepoint_28_genlasso_1_gradvi_direct_init_1.pkl', "wb"))


