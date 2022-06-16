library("dscrutils")

get_input_modules <- function (x, s, cmark) {
  xnew <- gsub(cmark, "$", x)
  return (c(strsplit(xnew, s)[[1]]))
}

is.empty.result <- function (dat)
  any(sapply(dat,length)) == 0

is.output.column <- function (x) {
  n <- nchar(x)
  if (n < 7)
    return(FALSE)
  else if (length(unlist(strsplit(x,"[:]"))) != 2)
    return(FALSE)
  else
    return(substr(x,n - 6,n) == ":output")
}

import.dsc.output <- function (outfile, outdir, ignore.missing.files) {
  out <- dscrutils::dscread(outdir,outfile)
  if (is.null(out) & !ignore.missing.files)
    stop(sprintf(paste("Unable to read from DSC output file %s. You can set",
                       "ignore.missing.files = TRUE to ignore this issue."),
                 outfile))
  return(out)
}

read.dsc.outputs <- function (dat, dsc.outdir, ignore.missing.files, verbose) {
  
  # Convert the DSC query result to a nested list. Here we use a
  # "trick", setting all missing values to NA with logical type. This
  # helps later on when using "unlist" to combine several values that
  # are of different types, some of which are NA; with NAs set to
  # logical, "unlist" should do a better job getting the best type.
  #
  # Note that the "as.logical" part is redundant, this is helpful to
  # make it explicit that the NA is of type logical.
  dat <- as.list(dat)
  n   <- length(dat)
  for (i in 1:n) {
    x           <- as.list(dat[[i]])
    x[is.na(x)] <- as.logical(NA)
    dat[[i]]    <- x
  }
  
  # Determine which columns contain names of files that should be
  # read; these are columns of the form "module.variable:output". If
  # there are no such columns, there is nothing to do here.
  cols <- which(sapply(as.list(names(dat)),is.output.column))
  if (length(cols) == 0)
    return(dat)
  
  # Create a new nested list data structure in which each element
  # corresponds to a single file containing DSC results; each of these
  # list elements is also a list, in which each of these elements
  # corresponds to a single variable extracted from the DSC results
  # file.
  #
  # Here we need to be careful to skip missing (NA) files.
  #
  files      <- unique(do.call(c,dat[cols]))
  files      <- files[!is.na(files)]
  n          <- length(files)
  out        <- rep(list(list()),n)
  vars       <- rep(as.character(NA),n)
  names(out) <- files
  for (i in cols) {
    
    # Get the name of the variable to extract.
    x <- names(dat)[i]
    x <- unlist(strsplit(x,"[.]"))[2]
    x <- substr(x,1,nchar(x) - 7)
    vars[i] <- x
    
    for (j in dat[[i]])
      if (!is.na(j))
        out[[j]][[x]] <- NA
  }
  
  # Extract the outputs.
  if (verbose)
    pb <-
    progress::progress_bar$new(format = "- Loading targets [:bar] :percent eta: :eta",
                     total = n,clear = FALSE,width = 60,show_after = 0)
  else
    pb <- null_progress_bar$new()
  pb$tick(0)
  for (i in files) {
    pb$tick()
    x <- import.dsc.output(i,dsc.outdir,ignore.missing.files)
    if (!is.null(x))
      for (j in names(out[[i]]))
        if (j == "DSC_TIME")
          if ("elapsed" %in% names(x$DSC_DEBUG$time)) {
            out[[i]][[j]] <- x$DSC_DEBUG$time$elapsed
          } else {
            out[[i]][[j]] <- x$DSC_DEBUG$time
          }
    else if (!is.element(j,names(x)))
      # https://github.com/stephenslab/dsc/issues/202
      out[[i]][j] <- NA
    else
      out[[i]][j] <- list(x[[j]])
  }
  
  # Copy the DSC outputs from the intermediate nexted list to final
  # nested list, "dat". The names of the DSC output files are replaced
  # by the extracted values of the requested targets.
  for (i in cols) {
    n <- length(dat[[i]])
    v <- vars[i]
    for (j in 1:n) {
      file <- dat[[i]][[j]]
      if (!is.na(file))
        dat[[i]][j] <- list(out[[file]][[v]])
    }
  }
  
  return(dat)
}
