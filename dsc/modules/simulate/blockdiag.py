#
import simulate

n, p, s = \
    simulate.parse_input_params(dims, sfrac=sfrac, sfix=sfix)

X, y, Xtest, ytest, beta, sigma = \
    simulate.linear_model(
        n, p, s, pve, ntest = ntest,
        corr_method = 'blockdiag', rho = rholist, min_block_size = min_block_size,
        signal = signal, bfix = bfix,
        seed = None)
