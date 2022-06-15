# Script for calling dscrutils
# and saving the result as RDS file
#

library(optparse)
library(dscrutils)

get_input_modules <- function (x, s, cmark) {
  xnew <- gsub(cmark, "$", x)
  return (c(strsplit(xnew, s)[[1]]))
}

options = list(
  make_option(c("-o", "--outdir"), type="character", default = NULL,
              help="output directory of DSC results", metavar="character"),
  make_option(c("-t", "--targets"), type="character", default = NULL,
              help="list of dsc-query targets", metavar="character"),
  make_option(c("-c", "--conditions"), type="character", default = NULL,
              help="list of dsc-query conditions", metavar="character"),
  make_option(c("-s", "--separator"), type="character", default = ",",
              help="separator for items in conditions and/or targets", metavar="character"),
  make_option(c("-m", "--cmarker"), type="character", default = "##",
              help="placeholder marker for the $ sign required for conditions", 
              metavar="character"),
  make_option(c("-g", "--groups"), type="character", default = NULL,
              help="list of dsc-query groups", metavar="character"),
  make_option(c("-r", "--rdsfile"), type="character", default = NULL,
              help="name of output RDS file", metavar="character")
)

opt_parser = OptionParser(option_list=options);
opt = parse_args(opt_parser);

targets = get_input_modules(opt$targets, opt$separator, opt$cmarker)
if (!is.null(opt$conditions)) {
  conditions = get_input_modules(opt$conditions, opt$separator, opt$cmarker)
} else {
  conditions = NULL
}
# print(opt$groups)
if (!is.null(opt$groups)) {
  groups = get_input_modules(opt$groups, opt$separator, opt$cmarker)
} else {
  groups = NULL
}



dscout <- dscquery(dsc.outdir = opt$outdir,
                   targets = targets,
                   conditions = conditions,
                   groups = groups,
                   verbose = TRUE)
saveRDS(dscout, file = opt$rdsfile)

# Some debug scripts --
# opt$outdir
# print("Targets")
# targets
# length(targets)
# nchar(targets)
# opt$rds
# conditions
#
# setwd("/home/saikat/Documents/work/ebmr/simulation/mr-ash-dsc/dsc")
# dscout <- dscquery(dsc.outdir = "dsc_result", 
#                    targets = c("simulate", "simulate.se", "simulate.dims", "simulate.sfrac", "simulate.pve"),
#                    conditions = c("$(simulate) == 'indepgauss'", "$(simulate.dims) == '(100,200)'"),
#                    verbose = TRUE)
