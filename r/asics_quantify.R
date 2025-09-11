#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse); library(jsonlite); library(ASICS)
})

opt <- parse_args(OptionParser(option_list=list(
  make_option(c("--mixture_csv"), type="character"),             # CSV: first col 'ppm', then one col per mixture (use one col)
  make_option(c("--library_csv"), type="character"),             # CSV: first col 'ppm', then one col per pure reference
  make_option(c("--nb_protons_json"), type="character", default=NULL), # JSON array, same order as ref columns
  make_option(c("--exclusion_json"), type="character", default=NULL),  # JSON [[start,end], ...]
  make_option(c("--max_shift"), type="double", default=0.02),
  make_option(c("--quant_method"), type="character", default="FWER")   # "FWER" or "Lasso"
)))

# import mixture
mix_data <- importSpectra(opt$mixture_csv, type.import="csv",
                          baseline.correction=TRUE, alignment=FALSE, normalisation=TRUE)
mix_obj  <- createSpectra(mix_data)

# import library
lib_data <- importSpectra(opt$library_csv, type.import="csv",
                          baseline.correction=TRUE, alignment=FALSE, normalisation=TRUE)

nbp <- NULL
if (!is.null(opt$nb_protons_json)) nbp <- fromJSON(opt$nb_protons_json)
pure_lib <- if (is.null(nbp)) createPureLibrary(lib_data) else createPureLibrary(lib_data, nb.protons=nbp)

# exclusions
excl <- matrix(c(4.5,5.1), ncol=2) # default water region
if (!is.null(opt$exclusion_json)) {
  e <- fromJSON(opt$exclusion_json)
  excl <- matrix(unlist(e), ncol=2, byrow=TRUE)
}

# run
res <- ASICS(mix_obj,
             pure.library = pure_lib,
             exclusion.areas = excl,
             max.shift = opt$max_shift,
             joint.align = FALSE,
             quantif.method = opt$quant_method,
             ncores = 1, verbose = FALSE)

q <- getQuantification(res) # rows: mixtures, cols: reference names
cat(toJSON(list(
  quantification = q,
  colnames = colnames(q),
  rownames = rownames(q)
), auto_unbox = TRUE, digits = NA))
