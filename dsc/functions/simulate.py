import numpy as np
import collections
import patsy
from gradvi.models import basis_matrix as gv_basemat

def parse_input_params(dims, sfrac=0.5, sfix=None):
    n = dims[0]
    p = dims[1]
    if sfix is not None:
        s = sfix
    else:
        s = max(1, int(sfrac * p))
    return n, p, s


def center_and_scale(Z):
    dim = Z.ndim
    if dim == 1:
        Znew = Z / np.std(Z)
        Znew = Znew - np.mean(Znew)
    elif dim == 2:
        Znew = Z / np.std(Z, axis = 0)
        Znew = Znew - np.mean(Znew, axis = 0).reshape(1, -1)
    return Znew


def sample_coefs (p, bidx, method="normal", bfix=None, options = {}):
    ''' 
    Sample coefficientss from a distribution (method = normal / gamma)
    or use a specified value for all betas:
        bfix = const -> all betas will have beta = const
        bfix = [a, b, c, ...] -> all betas can be specified using an array
    Note: 
        when sampling from the gamma distribution,
        a random sign (+/-) will be assigned
    '''
    beta = np.zeros(p)
    s = bidx.shape[0]

    # helper function to obtain random sign (+1, -1) with equal proportion (f = 0.5)
    def sample_sign(n, f = 0.5):
        return np.random.choice([-1, 1], size=n, p=[f, 1 - f])

    # sample beta from Gaussian(mean = 0, sd = 1)
    if method == "normal":
        loc = options.get('loc', 0.0)
        scale = options.get('scale', 1.0)
        beta[bidx] = np.random.normal(loc, scale, size = s)

    # receive fixed beta input
    elif method == "fixed":
        assert bfix is not None, "bfix is not specified for fixed signal"
        if isinstance(bfix, (collections.abc.Sequence, np.ndarray)):
            assert len(bfix) == s, "Length of input coefficient sequence is different from the number of non-zero coefficients"
            beta[bidx] = bfix
        else:
            beta[bidx] = np.repeat(bfix, s)

    # sample beta from a Gamma(40, 0.1) distribution and assign random sign
    elif method == "gamma":
        shape = options.get('shape', 40)
        scale = options.get('scale', 0.1)
        beta[bidx] = np.random.gamma(shape, scale, size = s)
        beta[bidx] = np.multiply(beta[bidx], sample_sign(s))

    return beta


def get_responses (x, b, std):
    return np.dot(x, b) + std * np.random.normal(size = x.shape[0])


def get_sd_from_pve (x, b, pve):
    return np.sqrt(np.var(np.dot(x, b)) * (1 - pve) / pve)


def linear_model (n, p, s, pve, ntest = 1000,
        signal = "normal", signal_params = {}, bfix = None,
        rho = [0.0], corr_method = 'iid', min_block_size = 100,
        seed = None,
        standardize = True):
    
    # set seed
    if seed is not None: np.random.seed(seed)

    # sample predictors
    # ensure rho is a list, array, tuple, set, or any other iterable
    if not isinstance(rho, (collections.abc.Sequence, np.ndarray)): rho = [rho]
    # seed is already set
    xtrain, xtest = predictor_factory(n, ntest, p, rholist = rho, 
                            corr_method = corr_method,
                            min_block_size = min_block_size,
                            standardize = standardize, seed = None)

    # sample coefficients
    bidx = np.random.choice(p, s, replace = False)
    beta = sample_coefs(p, bidx, method = signal, bfix = bfix, options = signal_params)

    # standard deviation from PVE
    std = get_sd_from_pve(xtrain, beta, pve)

    # responses
    ytrain = get_responses(xtrain, beta, std)
    ytest  = get_responses(xtest,  beta, std)
    return xtrain, ytrain, xtest, ytest, beta, std


def predictor_factory(n0, n1, p, rholist = [0.8], 
        corr_method = 'iid', seed = None,
        min_block_size = 100,
        standardize = True):
    if corr_method == 'iid':
        x0, x1 = equicorr_predictors(n0, n1, p, rho = 0.0, seed = seed, standardize = standardize)
    elif corr_method == 'equicorr':
        x0, x1 = equicorr_predictors(n0, n1, p, rho = rholist[0], seed = seed, standardize = standardize)
    elif corr_method == 'blockdiag':
        x0, x1 = blockdiag_predictors(n0, n1, p, rholist, min_block_size = min_block_size, seed = seed, standardize = standardize)
    return x0, x1


def equicorr_predictors(n0, n1, p, rho = 0.8, seed = None, standardize = True):
    '''
    X is sampled from a multivariate normal, with covariance matrix S.
    S has unit diagonal entries and constant off-diagonal entries rho.
    '''
    if seed is not None: np.random.seed(seed)
    ntot  = n0 + n1
    iidx  = np.random.normal(size = ntot * p).reshape(ntot, p)
    comR  = np.random.normal(size = ntot).reshape(ntot, 1)
    allx  = comR * np.sqrt(rho) + iidx * np.sqrt(1 - rho)

    # split into training and test data
    x0 = allx[:n0, :]
    x1 = allx[n0:, :]

    # standardize if required
    if standardize:
        x0 = center_and_scale(x0)
        x1 = center_and_scale(x1)

    return x0, x1


def blockdiag_predictors(n0, n1, p, rholist, min_block_size = 100, seed = None, standardize = True):
    '''
    X is sampled from a multivariate normal, with block-diagonal covariance matrix S.
    S has unit diagonal entries and k blocks of matrices, whose off-diagonal entries 
    are specified by elements of `rholist`.
    '''
    if seed is not None: np.random.seed(seed)
    ntot  = n0 + n1
    iidx  = np.random.normal(size = ntot * p).reshape(ntot, p)

    # number of blocks
    k = len(rholist)
    # do we need to change min_block_size?
    c = min(min_block_size, int(p / k))
    if c == 0: c = 1 # no minimum block size: for numerical reasons, each block must have atleast 1 variable
    if c < 0:  c = int(p / k) # negative input: equal block size
    # choose k-1 boundaries from p/c and multiply by c (to ensure minimum block size)
    bdr = c * np.sort(np.random.choice(int(p / c) - 1, k - 1, replace = False) + 1)

    #
    allx = np.zeros_like(iidx)
    if k > 1:
        for i, rho in enumerate(rholist):
            comR = np.random.normal(0., 1., size = ntot).reshape(ntot, 1)
            if i == 0:
                allx[:, :bdr[i]]  = comR * np.sqrt(rho) + iidx[:, :bdr[i]] * np.sqrt(1 - rho)
            elif i == k - 1:
                allx[:, bdr[i-1]:] = comR * np.sqrt(rho) + iidx[:, bdr[i-1]:] * np.sqrt(1 - rho)
            elif i > 0 and i < k - 1:
                allx[:, bdr[i-1]:bdr[i]] = comR * np.sqrt(rho) + iidx[:, bdr[i-1]:bdr[i]] * np.sqrt(1 - rho)
    elif k == 1:
        rho   = rholist[0]
        comR  = np.random.normal(0., 1., size = ntot).reshape(ntot, 1)
        allx  = comR * np.sqrt(rho) + iidx * np.sqrt(1 - rho)

    # split into training and test data
    x0 = allx[:n0, :]
    x1 = allx[n0:, :]

    # standardize if required
    if standardize:
        x0 = center_and_scale(x0)
        x1 = center_and_scale(x1)

    return x0, x1


def timeseries(x, sfix, degree, snr = 2.0, signal = 'normal', bfix = None, seed = None):

    if seed is not None: np.random.seed(seed)
    n = x.shape[0]
    bidx = np.floor(np.linspace(0, n, sfix + 2)[1:-1]).astype(int)
    btrue_nz = sample_coefs(bidx.shape[0], np.arange(bidx.shape[0]), method = signal, bfix = bfix)

    btrue = np.zeros(n)
    btrue[bidx] = btrue_nz

    # Select a random initial slope, next ones are created from changepoints.
    slope = np.ones(x.shape[0]) * np.random.normal(0, 1, 1)
    for i, idx in enumerate(bidx):
        slope[idx:] = btrue_nz[i]

    # Generate time series data
    ytrue = slope.copy()
    if degree == 1:
        ytrue[0] = slope[0]
        for i in range(1, n):
            ytrue[i] = ytrue[i - 1] + slope[i]
    elif degree == 2:
        ytrue[0] = slope[0]
        ytrue[1] = slope[1]
        for i in range(2, n):
            ytrue[i] = 2 * ytrue[i-1] - ytrue[i-2] + slope[i]

    # Generate noise
    signal = np.sum(np.square(ytrue)) / n
    std = np.sqrt(signal) / snr
    noise = np.random.normal(0, std, n * 2)
    y = ytrue + noise[:n]
    ytest = ytrue + noise[n:]
    return y, ytest, ytrue, btrue, std


def changepoint_design (x, sfix, snr,
                  degree = 0, signal = "gamma", signal_params = {}, 
                  seed = None,
                  include_intercept = False, bfix = None,
                  dummy = False,
                  eps = 1e-8):
    y, ytest, ytrue, btrue, strue = timeseries(x, sfix, degree, 
            snr = snr, signal = signal, bfix = bfix, seed = seed)
    if dummy:
        Xdummy = np.zeros((2,2))
        return Xdummy, Xdummy, Xdummy, Xdummy, y, ytest, ytrue, btrue, strue
    else:
        n = x.shape[0]
        H = gv_basemat.trendfiltering(n, degree)
        Hinv = gv_basemat.trendfiltering_inverse(n, degree)
        Hscale = gv_basemat.trendfiltering_scaled(n, degree)
        Hinvscale = gv_basemat.trendfiltering_inverse_scaled(n, degree)
        b_linreg = np.dot(Hinv, ytrue)
        b_linreg[np.abs(b_linreg) <= 1e-8] = 0.
        return H, Hinv, Hscale, Hinvscale, y, ytest, ytrue, b_linreg, strue


### def changepoint_from_bspline (x, knots, std,
###                  degree = 0, signal = "gamma", seed = None,
###                  include_intercept = False, bfix = None,
###                  eps = 1e-8, get_bsplines = False):
###     if seed is not None: np.random.seed(seed)
###     # ------------------------------
###     n = x.shape[0]
###     # ------------------------------
###     # Generate B-spline bases given the knots and degree
###     bspline_bases = patsy.bs(x, knots = knots, degree = degree, include_intercept = include_intercept)
###     nbases = knots.shape[0] + degree + int(include_intercept)
###     assert bspline_bases.shape[1] == nbases, "Number of B-spline bases does not match the number of knots + degree + interecept"
###     # ------------------------------
###     # Generate coefficients for the bases
###     beta  = sample_coefs(nbases, np.arange(nbases), method = signal, bfix = bfix)
###     # ------------------------------
###     # Generate the function without noise 
###     ytrue = np.dot(bspline_bases, beta)
###     # ------------------------------
###     # Map the data to trendfiltering bases
###     # set low values of beta to zero and regenerate y
###     H     = gv_basemat.trendfiltering(n, degree)
###     Hinv  = gv_basemat.trendfiltering_inverse(n, degree)
###     Hscale     = gv_basemat.trendfiltering_scaled(n, degree)
###     Hinvscale  = gv_basemat.trendfiltering_inverse_scaled(n, degree)
###     btrue = np.dot(Hinv, ytrue)
###     btrue[np.abs(btrue) <= eps] = 0.
###     noise = np.random.normal(0, std, size = n * 2)
###     ytrue = np.dot(H, btrue)
###     y     = ytrue + noise[:n]
###     # ------------------------------
###     # Some test data?
###     ytest = ytrue + noise[n:]
###     # ------------------------------
###     # Signal to noise ratio 
###     # (experimental)
###     signal = np.mean(np.square(btrue[btrue != 0]))
###     snr    = signal / np.square(std)
###     #
###     return H, Hinv, Hscale, Hinvscale, y, ytest, ytrue, btrue, snr
### 
### 
### def changepoint_from_btrue (x, sfix, std,
###                  degree = 0, signal = "gamma", signal_params = {}, 
###                  seed = None,
###                  include_intercept = False, bfix = None,
###                  dummy = False,
###                  eps = 1e-8):
###     if seed is not None: np.random.seed(seed)
###     n = x.shape[0]
###     bidx  = np.floor(np.linspace(0, n, sfix + 2)[1:-1]).astype(int)
###     btrue = sample_coefs(n, bidx, method = signal, bfix = bfix, options = signal_params)
###     ytrue = Hdotv_unscaled(btrue, degree)
###     noise = np.random.normal(0, std, size = n * 2)
###     y     = ytrue + noise[:n]
###     ytest = ytrue + noise[n:]
###     signal = np.mean(np.square(btrue[btrue != 0]))
###     snr    = signal / np.square(std)
###     Xdummy = np.zeros((2,2))
###     if dummy:
###         return Xdummy, Xdummy, Xdummy, Xdummy, y, ytest, ytrue, btrue, snr
###     else:
###         H     = gv_basemat.trendfiltering(n, degree)
###         Hinv  = gv_basemat.trendfiltering_inverse(n, degree)
###         Hscale     = gv_basemat.trendfiltering_scaled(n, degree)
###         Hinvscale  = gv_basemat.trendfiltering_inverse_scaled(n, degree)
###         return H, Hinv, Hscale, Hinvscale, y, ytest, ytrue, btrue, snr
### 
### 
### def Hdotv_unscaled(v, d):
###     X3v1 = np.zeros_like(v)
###     v0 = v[:d]
###     v1 = v[d:]
###     X3v1[d:] = np.cumsum(v1)
###     for i in range(d):
###         X3v1 = np.cumsum(X3v1)
###     if d == 0:
###         Xv = X3v1
###     elif d == 1:
###         Xv = v[0] + X3v1
###     else:
###         X0v0 = np.dot(self._tf_X[:, :d], v0)
###         Xv = X0v0 + X3v1
###     return Xv
###     
### 
### def equicorr_predictors_old (n, p, s, pve, ntest = 1000, 
###         signal = "normal", seed = None, 
###         rho = 0.5, bfix = None,
###         standardize = True):
###     '''
###     X is sampled from a multivariate normal, with covariance matrix S.
###     S has unit diagonal entries and constant off-diagonal entries rho.
###     '''
###     if seed is not None: np.random.seed(seed)
###     ntot  = n + ntest
###     iidX  = np.random.normal(size = ntot * p).reshape(ntot, p)
###     comR  = np.random.normal(size = ntot).reshape(ntot, 1)
###     Xall  = comR * np.sqrt(rho) + iidX * np.sqrt(1 - rho)
###     # split into training and test data
###     X     = Xall[:n, :]
###     Xtest = Xall[n:, :]
###     if standardize:
###         X = center_and_scale(X)
###         Xtest = center_and_scale(Xtest)
###     # sample betas
###     bidx  = np.random.choice(p, s, replace = False)
###     beta  = sample_coefs(p, bidx, method = signal, bfix = bfix)
###     # obtain sd from pve
###     se    = get_sd_from_pve(X, beta, pve)
###     # calculate the responses
###     y     = get_responses(X,     beta, se)
###     ytest = get_responses(Xtest, beta, se)
###     return X, y, Xtest, ytest, beta, se
