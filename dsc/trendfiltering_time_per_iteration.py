import numpy as np
import os
import pickle
import gc

from gradvi.tests import toy_data
from gradvi.tests import toy_priors
from gradvi.inference import LinearRegression
from gradvi.inference import Trendfiltering
from mrashpen.inference.mrash_wrapR import MrASHR

niter   = 10
degrees = [0, 1, 2]
dims    = np.power(2, 7 + np.arange(9))
nknots  = 10
strue   = 0.5
knots   = np.linspace(0, 1, nknots+2)[1:-1]
methods = ['cavi', 'gradvi', 'gradvi_tf', 'gradvi_tf_scaled', 'gradvi_tf_direct', 'gradvi_tf_direct_scaled']

itertime_pklfile = "results/trendfiltering_time_per_iteration.pkl"
verbose = True


def load_itertime():
    if os.path.isfile(itertime_pklfile):
        with open(itertime_pklfile, 'rb') as fhandle:
            itertime = pickle.load(fhandle)
    else:
        itertime = dict()
    return itertime


def save_itertime(xdict):
    with open(itertime_pklfile, 'wb') as fhandle:
        pickle.dump(xdict, fhandle, protocol = pickle.HIGHEST_PROTOCOL)


def append_itertime(xdict, degree, method, dim, itr, res):
    if res is not None:
        if degree not in xdict.keys():
            xdict[degree] = dict()
        if method not in xdict[degree].keys():
            xdict[degree][method] = dict()
        if dim not in xdict[degree][method].keys():
            xdict[degree][method][dim] = dict()
        xdict[degree][method][dim][itr] = res
    return xdict


def log_iteration(xdict, degree, method, dim, itr):
    message = f"Degree {degree}. N = {dim}. Iteration {itr + 1}."
    if check_run_exists(xdict, degree, method, dim, itr):
        mtime = xdict[degree][method][dim][itr]
        message += f" Iteration time: {mtime:.4f} for {method}."
    else:
        message += f" Skip {method}."
    print(message)


def check_run_exists(xdict, degree, method, dim, itr):
    success = False
    if degree in xdict.keys():
        if method in xdict[degree].keys():
            if dim in xdict[degree][method].keys():
                if itr in xdict[degree][method][dim].keys():
                    success = True
    return success


def is_run_required(method, degree, dim):
    if method == 'cavi' and dim > 20000:
        return False
    if method == 'gradvi' and dim > 20000:
        return False
    if method == 'gradvi_tf_direct' and dim > 10000:
        return False
    return True


def run_method(data, method):
    if method == 'cavi':
        res = run_cavi(data)
    elif method == 'gradvi':
        res = run_gradvi(data)
    elif method == 'gradvi_tf':
        res = run_gradvi_tf(data, obj = 'reparametrize', scale_tfbasis = False)
    elif method == 'gradvi_tf_scaled':
        res = run_gradvi_tf(data, obj = 'reparametrize', scale_tfbasis = True)
    elif method == 'gradvi_tf_direct':
        res = run_gradvi_tf(data, obj = 'direct', scale_tfbasis = False)
    elif method == 'gradvi_tf_direct_scaled':
        res = run_gradvi_tf(data, obj = 'direct', scale_tfbasis = True)
    return res


def run_cavi(data):
    n = data.y.shape[0]
    if n > 20000:
        return None
    else:
        prior = toy_priors.get_ash_scaled(k = 20, sparsity = 0.9, skbase = 20)
        cavi  = MrASHR(option = "rds", debug = False)
        cavi.fit(data.H, data.y, prior.sk, winit = prior.w, binit = np.zeros(data.y.shape[0]), maxiter = 500)
        mtime = cavi.fitobj['run_time'] / cavi.niter
        return mtime


def run_gradvi(data):
    n = data.y.shape[0]
    if n > 20000:
        return None
    else:
        prior = toy_priors.get_ash_scaled(k = 20, sparsity = 0.9, skbase = 20)
        gv = LinearRegression(obj = 'reparametrize', maxiter = 200)
        gv.fit(data.H, data.y, prior)
        mtime = gv._res.optim_time / gv.nfev
        return mtime
    


def run_gradvi_tf(data, obj = 'reparametrize', scale_tfbasis = False):
    n = data.y.shape[0]
    if ((obj == 'direct') and (not scale_tfbasis) and (n > 10000)):
        return None
    else:
        prior = toy_priors.get_ash_scaled(k = 20, sparsity = 0.9, skbase = 20)
        gv = Trendfiltering(maxiter = 200, obj = obj, scale_tfbasis = scale_tfbasis)
        gv.fit(data.y, data.degree, prior)
        mtime = gv._res.optim_time / gv.nfev
        return mtime
    

for degree in degrees:
    for n in dims:
        for itr in range(niter):
            seed = 100 + itr
            x = np.linspace(0, 1, n)
            # dummy variable
            data = None
            for method in methods:
                itertime = load_itertime()
                if (not check_run_exists(itertime, degree, method, n, itr)) and is_run_required(method, degree, n):
                    if data is None:
                        data = toy_data.changepoint_from_bspline(x, knots,
                            strue, degree = degree, signal = "normal", 
                            seed = seed, include_intercept = False)
                    print (method, degree, n, itr, seed)
                    mtime = run_method(data, method)
                    if mtime is not None:
                        itertime = append_itertime(itertime, degree, method, n, itr, mtime)
                        save_itertime(itertime)
                if verbose:
                    log_iteration(itertime, degree, method, n, itr)
            del data
            gc.collect()
