#
import simulate

n, p, s = \
    simulate.parse_input_params(dims, sfrac=sfrac, sfix=sfix)

X, y, Xtest, ytest, beta, sigma = \
    simulate.equicorr_predictors(
        n, p, s, pve,
        ntest = ntest, signal = signal, 
        seed = None, rho = rho, bfix = bfix)
