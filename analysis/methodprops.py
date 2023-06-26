import collections

class PlotInfo(collections.namedtuple('_PLOT_FIELDS',
                                      ['color', 'facecolor', 'label', 'marker', 'size',
                                       'linewidth', 'linestyle', 'zorder'])):
    __slots__ = ()



'''
PlotInfo for all penalized regression methods
'''
def plot_metainfo():
    plotmeta = dict()
    plotmeta['mr_ash'] = \
        PlotInfo(color     = "#535154", # Medium Gray (kelly 7)
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
                 label     = "CAVI",
                 marker    = "o",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 1,
                )
    plotmeta['gradvi_compound'] = \
        PlotInfo(color     = "#FF6800",
                 facecolor = "#FCC19A",
                 label     = "Compound",
                 marker    = "v",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 19,
                )
    plotmeta['gradvi_compound_lasso_init'] = \
        PlotInfo(color     = "#CC2529",
                 facecolor = "#F59D9E",
                 label     = "Compound (init)",
                 marker    = "^",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 29,
                )
    plotmeta['gradvi_direct'] = \
        PlotInfo(color     = "#2D69C4",
                 facecolor = "#AFC8ED",
                 label     = "Direct",
                 marker    = "<",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 18,
                )
    plotmeta['gradvi_direct_lasso_init'] = \
        PlotInfo(color     = "#00538A",
                 facecolor = "#7898AD",
                 label     = "Direct (init)",
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
    return plotmeta
