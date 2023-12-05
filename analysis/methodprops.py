import collections

class PlotInfo(collections.namedtuple('_PLOT_FIELDS',
                                      ['color', 'facecolor', 'label', 'marker', 'size',
                                       'linewidth', 'linestyle', 'zorder'])):
    __slots__ = ()


def do_modify_namedtuple(x, properties = {}):
    x_dict = x._asdict()
    for k, v in properties.items():
        x_dict[k] = v
    return PlotInfo(**x_dict)


'''
PlotInfo for all penalized regression methods
'''
def plot_metainfo():
    plotmeta = dict()
    plotmeta['mr_ash'] = \
        PlotInfo(color     = "#908D91", # Light gray (reduced from Kelly 7)
                 facecolor = "#A8A7A8",
                 label     = "CAVI",
                 marker    = "o",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 1,
                )
    plotmeta['mr_ash_lasso_init'] = \
        PlotInfo(color     = "#535154", # Medium Gray (kelly 7)
                 facecolor = "#A8A7A8",
                 label     = "CAVI (init)",
                 marker    = "o",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 1,
                )
    plotmeta['gradvi_compound'] = \
        PlotInfo(color     = "#FF6800",
                 facecolor = "#FCC19A",
                 label     = "GradVI Compound",
                 marker    = "v",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 19,
                )
    plotmeta['gradvi_compound_lasso_init'] = \
        PlotInfo(color     = "#CC2529",
                 facecolor = "#F59D9E",
                 label     = "GradVI Compound (init)",
                 marker    = "^",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 29,
                )
    plotmeta['gradvi_direct'] = \
        PlotInfo(color     = "#2D69C4",
                 facecolor = "#AFC8ED",
                 label     = "GradVI Direct",
                 marker    = "<",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 18,
                )
    plotmeta['gradvi_direct_lasso_init'] = \
        PlotInfo(color     = "#00538A",
                 facecolor = "#7898AD",
                 label     = "GradVI Direct (init)",
                 marker    = ">",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 28,
                )
    plotmeta['example'] = \
        PlotInfo(color     = "black",
                 facecolor = None,
                 label     = "Example",
                 marker    = "o",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 1,
                )
    # Trendfiltering methods
    plotmeta['mr_ash_init'] = do_modify_namedtuple(
            plotmeta['mr_ash'])

    plotmeta['mr_ash_scaled_init'] = do_modify_namedtuple(
            plotmeta['mr_ash_lasso_init'], properties = {'label': 'CAVI (scaled H)'})

    plotmeta['gradvi_compound_init'] = do_modify_namedtuple(
            plotmeta['gradvi_compound'])

    plotmeta['gradvi_compound_scaled_init'] = do_modify_namedtuple(
            plotmeta['gradvi_compound_lasso_init'], properties = {'label': 'GradVI Compound (scaled H)'})

    plotmeta['gradvi_direct_scaled_init'] = do_modify_namedtuple(
            plotmeta['gradvi_direct_lasso_init'], properties = {'label': 'GradVI Direct (scaled H)'})

    return plotmeta
