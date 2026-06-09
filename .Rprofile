source("renv/activate.R")
local({
  lib <- .libPaths()[1]
  conda_lib <- sub("x86_64-pc-linux-gnu", "x86_64-conda-linux-gnu", lib)
  if (conda_lib != lib && dir.exists(conda_lib)) {
    .libPaths(c(conda_lib, .libPaths()))
  }
})
