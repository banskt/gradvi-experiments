#
import simulate

x = np.linspace(0, 1, n)
#knots = np.linspace(0, 1, sfix + 2)[1:-1]

H, Hinv, Hscale, Hinvscale, y, ytest, ytrue, beta, std = \
    simulate.changepoint_design(
        x, sfix, snr, degree = dtrue,
        signal = signal, include_intercept = False,
        dummy = lowmem,
        bfix = bfix, seed = None)
