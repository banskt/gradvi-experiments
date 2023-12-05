#
import numpy as np
import time
from gradvi.priors import Ash
from gradvi.inference import LinearRegression
from gradvi.inference import Trendfiltering
from gradvi.optimize import moving_average as gv_moving_average

# Used for ELBO calculation
from gradvi.models import basis_matrix as gv_basemat
from mrashpen.inference.mrash_wrapR import MrASHR

gradvi_class_properties = [
    '_dj',
    '_init_params',
    '_invert_method',
    '_invert_options',
    '_is_debug',
    '_is_elbo_calc',
    '_is_intercept',
    '_method',
    '_nclbk',
    '_objtype',
    '_opts',
    'coef',
    'elbo_path',
    'fun',
    'grad',
    'intercept',
    'nfev',
    'niter',
    'njev',
    'obj_path',
    'prior',
    'residual_var',
    'success',
    'theta']

gradvi_trendfiltering_class_properties = gradvi_class_properties + [
    '_tf_standardize_basis',
    '_tf_standardize_y',
    '_tf_scale_basis',
    '_tf_intercept',
    '_tf_ystd',
    '_tf_degree',
    '_tf_fstd',
    '_tf_floc']

prior_property_list = ['smbase', 'sk', 'w', 'wmod', 'w_init', 'wmod_init', 'is_scaled']

def get_ash(k = 20, sparsity = 0.8, skbase = 2.0, skfactor = 1.0, is_scaled = False, **kwargs):
    wk = np.zeros(k)
    wk[0] = 1.0 / k if sparsity is None else sparsity
    wk[1:(k-1)] = np.repeat((1 - wk[0])/(k-1), (k - 2))
    wk[k-1] = 1 - np.sum(wk)
    sk = skfactor * np.abs(np.power(skbase, np.arange(k) / k) - 1)
    prior = Ash(sk, wk = wk, scaled = is_scaled)
    return prior


def get_ash_scaled(k = 20, sparsity = 0.8, skbase = 2.0, skfactor = 1.0, **kwargs):
    return get_ash(k = k, sparsity = sparsity, skbase = skbase, skfactor = skfactor, is_scaled = True, **kwargs)


def fit_ash_gradvi(X, y, objtype, ncomp = 20, sparsity = 0.8, skbase = 2.0, skfactor = 1.0, binit = None, s2init = None, winit = None, return_pip = False, run_initialize = False):

    # Start time
    stwall = time.time()
    stcpu  = time.process_time()

    # initialization / prior
    if s2init is None: s2init = 1.0
    prior = get_ash_scaled(k = ncomp, sparsity = sparsity, skbase = skbase, skfactor = skfactor)

    # Initialization
    if run_initialize:
        # estimate g given b and s2
        gv0 = LinearRegression(obj = 'direct', optimize_b = False, optimize_s = False, optimize_w = True)
        gv0.fit(X, y, prior, b_init = binit, s2_init = s2init)


    # run Gradvi
    if run_initialize:
        # use the previous gv0 fit for initialization
        gv = LinearRegression(obj = objtype, optimize_b = True, optimize_s = True, optimize_w = True)
        if objtype == 'direct':
            gv.fit(X, y, gv0.prior, b_init = gv0.coef,  s2_init = gv0.residual_var)
        elif objtype == 'reparametrize':
            gv.fit(X, y, gv0.prior, t_init = gv0.theta, s2_init = gv0.residual_var)
    else:
        gv = LinearRegression(obj = objtype)
        gv.fit(X, y, prior, b_init = binit, s2_init = s2init)

    # End time
    etwall = time.time()
    etcpu  = time.process_time()
    runtime_wall = etwall - stwall
    runtime_cpu  = etcpu  - stcpu

    # convert class to dict    
    gvdict = class_to_dict(gv, gradvi_class_properties)
    gvdict["runtime_wall"] = runtime_wall
    gvdict["runtime_cpu"]  = runtime_cpu

    # get NormalMeans
    if return_pip:
        gvnm = gv.get_res_normal_means()
        phi, _, _ = gvnm.posterior()
        pip  = 1 - phi[:, 0]
        return gvdict, gv.intercept, gv.coef, pip

    else:
        return gvdict, gv.intercept, gv.coef


def fit_ash_trendfiltering_gradvi(y, objtype, degree = 0, ncomp = 20, sparsity = 0.9, skbase = 2.0, skfactor = 1.0, 
                                  yinit = None, s2init = None, winit = None, run_initialize = False, tol = 1e-8,
                                  standardize_basis = False, scale_basis = False, standardize = True, maxiter = 2000,
                                  return_mrash_elbo = False):
    # Start time
    stwall = time.time()
    stcpu  = time.process_time()

    # initialization
    n = y.shape[0]
    prior_init = get_ash(k = ncomp, sparsity = sparsity, skbase = skbase, skfactor = skfactor)

    gv = Trendfiltering(
            obj = objtype, maxiter = maxiter, tol = tol, 
            standardize_basis = standardize_basis, scale_basis = scale_basis, 
            standardize = standardize)
    gv.fit(y, degree, prior_init, y_init = yinit, s2_init = s2init)

    # End time
    etwall = time.time()
    etcpu  = time.process_time()
    runtime_wall = etwall - stwall
    runtime_cpu  = etcpu  - stcpu

    # convert class to dict    
    gvdict = class_to_dict(gv, gradvi_trendfiltering_class_properties)
    gvdict["runtime_wall"] = runtime_wall
    gvdict["runtime_cpu"]  = runtime_cpu
    if return_mrash_elbo:
        gvdict["elbo"] = get_elbo_from_mrash(gv, y)

    return gvdict, gv.intercept, gv.coef, gv.ypred


def get_elbo_from_mrash(gvcls, y):
    n = y.shape[0]
    degree = gvcls._tf_degree
    prior_sk = gvcls.prior.sk
    prior_wk = gvcls.prior.w
    y_opt = gvcls.ypred
    s2_opt = gvcls.residual_var
    
    X = gv_basemat.trendfiltering(n, degree)
    Xinv = gv_basemat.trendfiltering_inverse(n, degree)
    b_opt = np.dot(Xinv, y_opt)
    
    # Run Mr.Ash
    mrash_r = MrASHR(option = "rds")
    mrash_r.fit(X, y, prior_sk,
                binit = b_opt,
                winit = prior_wk,
                s2init = s2_opt,
                maxiter = 1,
                update_pi = False, 
                update_sigma2 = False)
    return mrash_r.elbo_path[0]

## Backward compatibility
def fit_mrash_gradvi_direct(X, y, ncomp = 20, sparsity = 0.8, skbase = 2.0, binit = None, s2init = None, winit = None):
    return fit_ash_gradvi(X, y, 'direct', ncomp = ncomp, sparsity = sparsity, skbase = skbase, binit = binit, s2init = s2init, winit = winit, return_pip = False)
    

def fit_mrash_gradvi_compound(X, y, ncomp = 20, sparsity = 0.8, skbase = 2.0, binit = None, s2init = None, winit = None):
    return fit_ash_gradvi(X, y, 'reparametrize', ncomp = ncomp, sparsity = sparsity, skbase = skbase, binit = binit, s2init = s2init, winit = winit, return_pip = False)


# This is a placeholder until I find a proper
# dict converter of the class.
# [(p, type(getattr(classname, p))) for p in dir(classname)]
# shows the types, but how to extract the @property methods?
def class_to_dict(classname, property_list):
    model = dict()
    for info in property_list:
        if info == 'prior':
            prcls = getattr(classname, info)
            model[info] = class_to_dict(prcls, prior_property_list)
        else:
            model[info] = getattr(classname, info)
    return model
