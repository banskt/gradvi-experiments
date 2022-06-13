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

def fit_mrash_gradvi_direct(X, y, sk, binit = None, s2init = None, winit = None):
    prior = Ash(sk, wk = winit, scaled = True)
    gv = LinearRegression(obj = 'direct')
    gv.fit(X, y, prior, b_init = binit, s2_init = s2init)
    gvdict = class_to_dict(gv, gradvi_class_properties)
    return gvdict, gv.intercept, gv.coef
    

def fit_mrash_gradvi_compound(X, y, sk, binit = None, s2init = None, winit = None):
    prior = Ash(sk, wk = winit, scaled = True)
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
        model[info] = getattr(classname, info)
    return model
