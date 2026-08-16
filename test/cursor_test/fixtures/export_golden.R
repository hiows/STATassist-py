# Export golden JSON fixtures from R STATassist for Python cross-validation.
# Run from repo root:
#   Rscript test/cursor_test/fixtures/export_golden.R

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Install jsonlite: install.packages('jsonlite')")
}
if (!requireNamespace("STATassist", quietly = TRUE)) {
  pkg_root <- normalizePath(file.path(dirname(sys.frame(1)$ofile), "../../../STATassist"))
  if (file.exists(file.path(pkg_root, "DESCRIPTION"))) {
    devtools::load_all(pkg_root)
  } else {
    stop("STATassist package not found")
  }
}

out_dir <- normalizePath(
  file.path(dirname(sys.frame(1)$ofile)),
  mustWork = FALSE
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_golden <- function(name, obj) {
  path <- file.path(out_dir, paste0(name, ".json"))
  jsonlite::write_json(obj, path, auto_unbox = TRUE, na = "null", pretty = TRUE)
  message("Wrote ", path)
}

# Two-group comparison
sim <- STATassist::simulate_two_groups(n_feats = 10, n_up = 2, n_down = 2, seed = 2026)
comp <- do.call(STATassist::compare_two_groups, c(sim$args, list(diagnose = FALSE)))
sig <- STATassist::estimate_significance(comp, test = "t_test")
write_golden("compare_two_groups", comp)
write_golden("estimate_significance", sig)

# Multi-group
sim_m <- STATassist::simulate_multiple_groups(
  n_feats = 8, n_control = 20, n_treat = c(20, 20), seed = 2026
)
comp_m <- do.call(STATassist::compare_multiple_groups, c(sim_m$args, list(diagnose = FALSE)))
write_golden("compare_multiple_groups", comp_m)

# Descriptive
desc <- STATassist::summarize_descriptive_stats(
  sim$args$data, sim$args$feats[1:3], sim$args$group, sim$args$group_lv
)
write_golden("summarize_descriptive_stats", desc)

message("Golden export complete.")
