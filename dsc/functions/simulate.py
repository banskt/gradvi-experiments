import numpy as np
import collections

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


def sample_coefs (p, bidx, method="normal", bfix=None):
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
        beta[bidx] = np.random.normal(size = s)

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
        params = [40, 0.1]
        beta[bidx] = np.random.gamma(params[0], params[1], size = s)
        beta[bidx] = np.multiply(beta[bidx], sample_sign(s))

    return beta


def get_responses (X, b, sd):
    return np.dot(X, b) + sd * np.random.normal(size = X.shape[0])


def get_sd_from_pve (X, b, pve):
    return np.sqrt(np.var(np.dot(X, b)) * (1 - pve) / pve)
    

def equicorr_predictors (n, p, s, pve, ntest = 1000, 
        signal = "normal", seed = None, 
        rho = 0.5, bfix = None,
        standardize = True):
    '''
    X is sampled from a multivariate normal, with covariance matrix S.
    S has unit diagonal entries and constant off-diagonal entries rho.
    '''
    if seed is not None: np.random.seed(seed)
    ntot  = n + ntest
    iidX  = np.random.normal(size = ntot * p).reshape(ntot, p)
    comR  = np.random.normal(size = ntot).reshape(ntot, 1)
    Xall  = comR * np.sqrt(rho) + iidX * np.sqrt(1 - rho)
    # split into training and test data
    X     = Xall[:n, :]
    Xtest = Xall[n:, :]
    if standardize:
        X = center_and_scale(X)
        Xtest = center_and_scale(Xtest)
    # sample betas
    bidx  = np.random.choice(p, s, replace = False)
    beta  = sample_coefs(p, bidx, method = signal, bfix = bfix)
    # obtain sd from pve
    se    = get_sd_from_pve(X, beta, pve)
    # calculate the responses
    y     = get_responses(X,     beta, se)
    ytest = get_responses(Xtest, beta, se)
    return X, y, Xtest, ytest, beta, se
