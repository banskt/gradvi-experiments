#
import simulate

x = np.linspace(0, 1, n)
knots = np.linspace(0, 1, sfix + 2)[1:-1]

H, Hinv, Hscale, Hinvscale, y, ytest, ytrue, beta, snr = \
    simulate.changepoint_from_bspline(
        x, knots, strue, degree = dtrue,
        signal = signal, include_intercept = False,
        bfix = bfix, seed = None)
