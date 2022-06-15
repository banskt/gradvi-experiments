#
import numpy as np
from gradvi.priors import Ash
from gradvi.inference import LinearRegression

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


def fit_mrash_gradvi_direct(X, y, ncomp = 20, sparsity = 0.8, skbase = 2.0, binit = None, s2init = None, winit = None):
    prior = get_ash_scaled(k = ncomp, sparsity = sparsity, skbase = skbase)
    gv = LinearRegression(obj = 'direct')
    gv.fit(X, y, prior, b_init = binit, s2_init = s2init)
    gvdict = class_to_dict(gv, gradvi_class_properties)
    return gvdict, gv.intercept, gv.coef
    

def fit_mrash_gradvi_compound(X, y, ncomp = 20, sparsity = 0.8, skbase = 2.0, binit = None, s2init = None, winit = None):
    prior = get_ash_scaled(k = ncomp, sparsity = sparsity, skbase = skbase)
    gv = LinearRegression()
    gv.fit(X, y, prior, b_init = binit, s2_init = s2init)
    gvdict = class_to_dict(gv, gradvi_class_properties)
    return gvdict, gv.intercept, gv.coef

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
