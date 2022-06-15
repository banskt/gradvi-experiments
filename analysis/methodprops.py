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
    plotmeta['ridge'] = \
        PlotInfo(color     = "#817066", # Medium Gray (kelly 7)
                 #facecolor = "#817066",
                 facecolor = "None",
                 label     = "Ridge",
                 marker    = "s",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 10,
                )
    plotmeta['l0learn'] = \
        PlotInfo(color     = "#93BFEB", # Perano Blue
                 #facecolor = "#93BFEB",
                 facecolor = "None",
                 label     = "L0Learn",
                 marker    = "<",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 30,
                )
    plotmeta['lasso'] = \
        PlotInfo(color     = "#367DC4", # Boston Blue
                 #facecolor = '#367DC4',
                 facecolor = "None",
                 label     = "Lasso",
                 marker    = "^",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 20,
                )
    plotmeta['lasso_1se'] = \
        PlotInfo(color     = "#367DC4", # Boston Blue
                 #facecolor = "white",
                 facecolor = "None",
                 label     = "Lasso (1se)",
                 marker    = "^",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 20,
                )
    plotmeta['elastic_net'] = \
        PlotInfo(color     = "#803E75", # Strong Purple (kelly 2)
                 #facecolor = "#803E75",
                 facecolor = "None",
                 label     = "Elastic Net",
                 marker    = ">",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 30,
                )
    plotmeta['elastic_net_1se'] = \
        PlotInfo(color     = "#00538A", # Strong Blue (Smalt)
                 #facecolor = "white",
                 facecolor = "None",
                 label     = "Elastic Net (1se)",
                 marker    = ">",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 30,
                )
    plotmeta['scad'] = \
        PlotInfo(color     = "#007D34", # Medium Purple
                 #facecolor = "#007D34",
                 facecolor = "None",
                 label     = "SCAD",
                 marker    = "v",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 30,
                )
    plotmeta['mcp'] = \
        PlotInfo(color     = "#00C45C", # Medium Purple
                 #facecolor = "None",
                 facecolor = "None",
                 label     = "MCP",
                 marker    = "v",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 30,
                )
    plotmeta['susie'] = \
        PlotInfo(color     = "#FF6800", # Vivid Orange (kelly 3)
                 #facecolor = "#FF6800",
                 facecolor = "None",
                 label     = "Susie",
                 marker    = "+",
                 size      = 12,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 30,
                )
    plotmeta['varbvs'] = \
        PlotInfo(color     = "#232C16",
                 #facecolor = "#232C16",
                 facecolor = "None",
                 label     = "varbvs",
                 marker    = "X",
                 size      = 12,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 30,
                )
    plotmeta['varbvsmix'] = \
        PlotInfo(color     = "#948B3D", # Olive
                 #facecolor = "#948B3D",
                 facecolor = "None",
                 label     = "varvbs (mix)",
                 marker    = "x",
                 size      = 12,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 30,
                )
    plotmeta['bayesb'] = \
        PlotInfo(color     = "#FFB300", # Strong Yellow (kelly 1)
                 #facecolor = "#FFB300",
                 facecolor = "None",
                 label     = "BayesB",
                 marker    = "D",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 30,
                )
    plotmeta['blasso'] = \
        PlotInfo(color     = "#CEA262", # Grayish Yellow (kelly 6)
                 #facecolor = "#CEA262",
                 facecolor = "None",
                 label     = "BLasso",
                 marker    = "*",
                 size      = 12,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 30,
                )
    plotmeta['mr_ash'] = \
        PlotInfo(color     = "#C10020", # Vivid Red (kelly 5)
                 facecolor = "#C10020",
                 label     = "Mr.Ash",
                 marker    = "o",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 200,
                )
    plotmeta['mr_ash_init'] = \
        PlotInfo(color     = "#C10020", # Vivid Red (kelly 5)
                 facecolor = "white",
                 label     = "Mr.Ash(init)",
                 marker    = "o",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 200,
                )
    plotmeta['em_vamp'] = \
        PlotInfo(color     = "#000000", # Black
                 facecolor = "None",
                 label     = "EM-VAMP",
                 marker    = "o",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 30,
                )
    plotmeta['em_vamp_ash'] = \
        PlotInfo(color     = "#000000", # Black
                 facecolor = "None",
                 label     = "EM-VAMP (ash)",
                 marker    = "*",
                 size      = 12,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 30,
                )
    plotmeta['em_iridge'] = \
        PlotInfo(color     = "#93AA00", # Vivid Yellowish Green
                 facecolor = "#93AA00",
                 label     = "IRidge-EM",
                 marker    = "D",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 30,
                )
    plotmeta['ebmr_lasso'] = \
        PlotInfo(color     = "#3A6200", # Dark Green
                 facecolor = "None",
                 label     = "EBMR (DExp, Py)",
                 marker    = "v",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 30,
                )
    plotmeta['ebmr_ashR'] = \
        PlotInfo(color     = "#6DB802", # Light Green
                 facecolor = "#6DB802",
                 label     = "EBMR (mix-DExp, R)",
                 marker    = "<",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 30,
                )
    plotmeta['ebmr_ash'] = \
        PlotInfo(color     = "#A5BD82", # Light Gray Green
                 facecolor = "#A5BD82",
                 label     = "EBMR (mix-point, Py)",
                 marker    = "<",
                 size      = 8,
                 linewidth = 2,
                 linestyle = "solid",
                 zorder    = 30,
                )
    
    return plotmeta
