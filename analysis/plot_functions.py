import numpy as np
import pandas as pd
import methodprops

def get_outlier_truncation_limits(data, factor = 1.0):
    q3, q1 = np.percentile(data, [75 ,25])
    iqr = q3 - q1
    # In ANY normal distribution: IQR = Q3 - Q1 = 0.67448σ - (-0.67448σ) = 1.34896σ
    iqr_sigma = iqr / 1.34896
    median = np.median(data)
    xmin = median - factor * iqr_sigma
    xmax = median + factor * iqr_sigma
    return xmin, xmax

def get_list_without_outliers(xlist, factor = 1.0):
    data = np.concatenate(xlist)
    xmin, xmax = get_outlier_truncation_limits(data, factor = factor)
    xfilter  = [x[(x >= xmin) & (x <= xmax)] for x in xlist]
    xremoved = [x[(x < xmin) | (x > xmax)] for x in xlist]
    return xfilter, xremoved

def norm_jitter(x, n, d = 0.1):
    jit = np.random.normal(0, 1, n) * d
    return x + jit

def get_truncated_list(ylist, ylim = None, d = 0.04):
    if ylim is None:
        data = np.concatenate(ylist)
        ylim = list(get_outlier_truncation_limits(data, factor = 1.0))
    ynew = ylist.copy()
    
    d = (ylim[1] - ylim[0]) / 10
    efflim = ylim.copy()
    efflim[0] = ylim[0] + d if ylim[0] < 0 else ylim[0] - d
    efflim[1] = ylim[1] + d if ylim[1] < 0 else ylim[1] - d

    for y in ynew:
        nmin = np.sum(y < efflim[0])
        nmax = np.sum(y > efflim[1])
        dmin = np.random.normal(0, 1, size = nmin) * d / 10
        dmax = np.random.normal(0, 1, size = nmax) * d / 10
        y[y < efflim[0]] = efflim[0] + dmin
        y[y > efflim[1]] = efflim[1] - dmax
    return ynew


def compare_methods_with_boxplots(
        ax, df, targets,
        xcol = 'simulate.pve', 
        remove_outliers = True,
        is_truncate = True,
        outlier_factor = 1.4, 
        ylim = None):


    xvals = df[xcol].unique()
    plotmeta = methodprops.plot_metainfo()
    nbox  = len(targets)
    yremoved = dict()

    if not isinstance(outlier_factor, list):
        outlier_factor = [outlier_factor for x in targets]
    if len(outlier_factor) == 1:
        outlier_factor = outlier_factor * len(targets)

    for i, target in enumerate(targets):

        # Boxplot properties
        boxcolor = plotmeta[target].color
        boxface = f'#{boxcolor[1:]}16' #https://stackoverflow.com/questions/15852122/hex-transparency-in-colors
        medianprops = dict(linewidth = 0, color = boxcolor)
        whiskerprops = dict(linewidth = 2, color = boxcolor)
        boxprops = dict(linewidth = 2, color = boxcolor, facecolor = boxface)
        flierprops = dict(marker = 'o', markerfacecolor = boxface, markersize = 3, markeredgecolor = boxcolor)

        # Position of each boxplot
        positions = np.array(range(len(xvals))) * (nbox + 1) + i

        # List of arrays 
        ylist_all = [df[df[xcol] == x][target].to_numpy() for x in xvals]

        # Remove outliers from boxplot
        # but keep them in scatterplot
        if remove_outliers:
            ylist, yremoved[target] = get_list_without_outliers(ylist_all, factor = outlier_factor[i])
        else:
            ylist = ylist_all.copy()
            yremoved[target] = [np.array([]) for y in ylist]

        ax.boxplot(ylist, positions = positions,
                   showcaps = False, showfliers = False,
                   widths = 0.8, patch_artist = True, notch = False,
                   flierprops = flierprops, boxprops = boxprops,
                   medianprops = medianprops, whiskerprops = whiskerprops)

        # Truncate the outlier values to show all points
        # within the visible axes.
        if is_truncate:
            ytrunc = get_truncated_list(ylist_all, ylim = ylim)
        else:
            ytrunc = ylist_all.copy()
        for pos, y in zip(positions, ytrunc):
            x = norm_jitter(pos, len(y))
            ax.scatter(x, y, alpha = 0.5,
                       color = plotmeta[target].facecolor, 
                       marker = plotmeta[target].marker,
                       s = plotmeta[target].size * 3)

    label_positions = np.array(range(len(xvals))) * (nbox + 1) + (nbox - 1) / 2.0
    ax.set_xticks(label_positions)
    ax.set_xticklabels(xvals)

    return yremoved
