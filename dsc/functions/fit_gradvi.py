#
import numpy as np
from gradvi.priors import Ash
from gradvi.inference import LinearRegression
from gradvi.optimize import moving_average as gv_moving_average

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
    'fitobj',
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

prior_property_list = ['smbase', 'sk', 'w', 'wmod', 'w_init', 'wmod_init', 'is_scaled']

def get_ash(k = 20, sparsity = 0.8, skbase = 2.0, is_scaled = False, **kwargs):
    wk = np.zeros(k)
    wk[0] = 1.0 / k if sparsity is None else sparsity
    wk[1:(k-1)] = np.repeat((1 - wk[0])/(k-1), (k - 2))
    wk[k-1] = 1 - np.sum(wk)
    sk = np.abs(np.power(skbase, np.arange(k) / k) - 1)
    prior = Ash(sk, wk = wk, scaled = is_scaled)
    return prior


def get_ash_scaled(k = 20, sparsity = 0.8, skbase = 2.0, **kwargs):
    return get_ash(k = k, sparsity = sparsity, skbase = skbase, is_scaled = True, **kwargs)


def fit_ash_gradvi(X, y, objtype, ncomp = 20, sparsity = 0.8, skbase = 2.0, binit = None, s2init = None, winit = None, return_pip = False, run_initialize = False):
    # initialization / prior
    if s2init is None: s2init = 1.0
    prior = get_ash_scaled(k = ncomp, sparsity = sparsity, skbase = skbase)

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

    # convert class to dict    
    gvdict = class_to_dict(gv, gradvi_class_properties)

    # get NormalMeans
    if return_pip:
        gvnm = gv.get_res_normal_means()
        phi, _, _ = gvnm.posterior()
        pip  = 1 - phi[:, 0]
        return gvdict, gv.intercept, gv.coef, pip

    else:
        return gvdict, gv.intercept, gv.coef


def fit_ash_trendfiltering_gradvi(X, y, objtype, degree = 0, ncomp = 20, sparsity = 0.9, skbase = 20.0, binit = None, s2init = None, winit = None):
    # initialization
    n = y.shape[0]
    prior_init = get_ash_scaled(k = ncomp, sparsity = sparsity, skbase = (degree + 1) * skbase)

    if binit is None:
        y0 = gv_moving_average.moving_average(y)
        binit = np.zeros_like(y0)
    else:
        y0 = np.dot(X, binit)

    if s2init is None:
        s2init = np.var(y - y0) / 10.0

    gv0 = LinearRegression(optimize_s = False, maxiter = 1000, obj = 'direct', tol = 1e-7)
    gv0.fit(X, y, prior_init, b_init = binit, s2_init = s2init)

    # run GradVI
    prior = get_ash_scaled(k = ncomp, sparsity = sparsity, skbase = (degree + 1) * skbase)
    gv = LinearRegression(obj = objtype)
    if objtype == 'direct':
        gv.fit(X, y, prior, b_init = gv0.coef, s2_init = gv0.residual_var)
    elif objtype == 'reparametrize':
        gv.fit(X, y, prior, t_init = gv0.theta, s2_init = gv0.residual_var)

    # convert class to dict    
    gvdict = class_to_dict(gv, gradvi_class_properties)

    return gvdict, gv.intercept, gv.coef



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
