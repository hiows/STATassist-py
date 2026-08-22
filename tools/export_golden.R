#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Golden fixtures: the numbers the Python port is graded against.
#
# Run once, from the package root:
#   Rscript tools/export_golden.R
#
# Each case writes testdata/golden/<name>/input.csv and expected.json. Both are
# committed, so the port is graded against R without R having to be installed.
#
# Two details are load-bearing.
#
# The input is rounded to six decimals before anything is computed from it. A CSV
# does not round-trip a double exactly, so computing the expected value from the
# unrounded draw would grade the port against a number its own input file does
# not hold. Rounding first makes the CSV text an exact record of the double that
# both languages then parse.
#
# `digits = NA` on the way out is not optional. jsonlite defaults to four
# significant digits, which would make an rtol of 1e-8 meaningless.
# ---------------------------------------------------------------------------

stopifnot(requireNamespace("jsonlite", quietly = TRUE))
stopifnot(requireNamespace("STATassist", quietly = TRUE))

ROOT <- file.path("testdata", "golden")
DIGITS <- 6L

write_case <- function(name, input, expected) {
  dir <- file.path(ROOT, name)
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(input, file.path(dir, "input.csv"),
                   row.names = FALSE, na = "")
  # `na = "string"` rather than "null". JSON has no infinity, so under "null" an
  # `Inf` and an `NA` both come out as `null` - and a one-sided interval leaves
  # its untested end at `Inf` on purpose. Writing them as "Inf", "-Inf", "NA" and
  # "NaN" keeps the four apart, and tests/golden.py reads them back.
  writeLines(
    jsonlite::toJSON(expected, digits = NA, na = "string", null = "null",
                     auto_unbox = TRUE, dataframe = "columns",
                     matrix = "rowmajor"),
    file.path(dir, "expected.json")
  )
  cat("wrote", name, "\n")
}

r6 <- function(x) round(x, DIGITS)

# jsonlite drops the names of an atomic vector and writes a bare array, which
# would throw away half of what a kernel's contract says. A list keeps them, so
# every named numeric vector goes out as a JSON object and the port is graded on
# the names as well as the numbers.
nv <- function(v) as.list(v)
nvs <- function(...) lapply(list(...), nv)

# Columns of one input.csv have to be the same length even when the samples they
# hold are not.
pad <- function(v, n) c(v, rep(NA_real_, n - length(v)))

# A matrix carries its column names as a data.frame, which the column-oriented
# JSON then keys by. Row order is the column order, so no row names are needed -
# and they are dropped, because jsonlite writes a non-automatic set of them as an
# extra `_row` column that the port has no counterpart for. A table whose row
# labels are part of the contract exports them under a name of its own.
as_cols <- function(m) {
  df <- as.data.frame(m, stringsAsFactors = FALSE)
  rownames(df) <- NULL
  df
}


# --------------------------------------------------------------------------- #
# Fixed inputs
#
# Drawn once here rather than per case, so several kernels are graded on the very
# same numbers and a disagreement between two of them cannot be blamed on the
# input.
# --------------------------------------------------------------------------- #

set.seed(20260820)

# Four groups, unequal sizes and unequal spreads: what separates the pooled tests
# from the Welch family.
long_group <- factor(rep(c("ctrl", "low", "mid", "high"),
                         times = c(9, 7, 8, 6)),
                     levels = c("ctrl", "low", "mid", "high"))
long_value <- r6(c(
  stats::rnorm(9, 10, 1.0),
  stats::rnorm(7, 11, 2.5),
  stats::rnorm(8, 12, 0.8),
  stats::rnorm(6, 13, 3.0)
))
long_df <- data.frame(group = as.character(long_group), value = long_value,
                      stringsAsFactors = FALSE)
samples <- split(long_value, long_group)

# Two independent samples of very different spread, for the Behrens-Fisher pair.
bm_x <- r6(stats::rnorm(14, 0, 1))
bm_y <- r6(stats::rnorm(11, 0.8, 3))
two_df <- data.frame(x = pad(bm_x, 14), y = pad(bm_y, 14))

# Complete pairs, with two stray values so trimming has something to do.
pair_x <- r6(stats::rnorm(16, 5, 1))
pair_y <- r6(stats::rnorm(16, 5.6, 1))
pair_x[3] <- pair_x[3] + 9
pair_y[12] <- pair_y[12] - 7
pair_df <- data.frame(x = pair_x, y = pair_y)

# Subjects by conditions, correlated within subject, for the repeated family.
n_subj <- 12L
subject_offset <- stats::rnorm(n_subj, 0, 2)
rm_mat <- r6(cbind(
  t1 = subject_offset + stats::rnorm(n_subj, 0.0, 1.0),
  t2 = subject_offset + stats::rnorm(n_subj, 0.7, 1.2),
  t3 = subject_offset + stats::rnorm(n_subj, 1.5, 0.9)
))

# One plain vector with a single value pushed out, for the screening rules.
out_value <- r6(stats::rnorm(20, 100, 5))
out_value[7] <- out_value[7] + 40

# Missing and non-finite values are never flagged, and fewer than three usable
# observations means no rule runs at all.
holed <- out_value
holed[c(4, 9)] <- NA_real_
holed[15] <- Inf
short <- c(1, NA, 2, rep(NA_real_, length(out_value) - 3L))
flat <- rep(3, length(out_value))

# A tie-heavy sample. The rank tests behave differently there and the tie
# correction is where a port most easily goes wrong.
tied_group <- factor(rep(c("a", "b", "c"), each = 6), levels = c("a", "b", "c"))
tied_value <- c(1, 2, 2, 3, 3, 3, 2, 3, 3, 4, 4, 5, 3, 4, 4, 5, 5, 6)
tied_samples <- split(tied_value, tied_group)

rank_df <- data.frame(
  group = c(as.character(long_group), as.character(tied_group)),
  value = c(long_value, tied_value),
  block = rep(c("continuous", "tied"),
              c(length(long_value), length(tied_value))),
  stringsAsFactors = FALSE
)

# A wide frame of three features and a group, for the public functions. gene_3 is
# built from gene_1 so that one pair of the association screen is strongly
# correlated and the rest are not.
wide <- data.frame(
  gene_1 = r6(stats::rnorm(24, 8, 1.5)),
  gene_2 = r6(stats::rnorm(24, 5, 0.7)),
  group = rep(c("ctrl", "treat_a", "treat_b"), each = 8),
  stringsAsFactors = FALSE
)
wide$gene_3 <- r6(wide$gene_1 * 1.3 + stats::rnorm(24, 0, 0.4))
wide <- wide[c("gene_1", "gene_2", "gene_3", "group")]

# Strictly positive, so the geometric centre and the log2 pipeline both exist.
fc_wide <- data.frame(
  prot_1 = r6(stats::runif(18, 4, 9)),
  prot_2 = r6(stats::runif(18, 1, 3)),
  group = rep(c("ctrl", "case", "other"), each = 6),
  stringsAsFactors = FALSE
)

# A three-by-two crossed design, deliberately unbalanced, so that Type I, II and
# III separate and Type I can be matched against aov().
fact_lv <- list(treatment = c("control", "treat_A", "treat_B"),
                sex = c("male", "female"))
fact_cells <- STATassist:::sa_fact_grid(fact_lv)
fact_label <- STATassist:::sa_fact_cell_labels(fact_lv, fact_cells)
fact_n <- c(6L, 5L, 4L, 7L, 3L, 5L)
fact_value <- r6(
  stats::rnorm(sum(fact_n), 10, 1.3) +
    rep(c(0, 0.4, 0.6, 1.5, 3.5, 2.1), times = fact_n)
)
fact_df <- data.frame(
  cell      = rep(fact_label, times = fact_n),
  treatment = rep(fact_lv$treatment[fact_cells$treatment], times = fact_n),
  sex       = rep(fact_lv$sex[fact_cells$sex], times = fact_n),
  value     = fact_value,
  stringsAsFactors = FALSE
)
fact_samples <- split(
  fact_value,
  factor(rep(fact_label, times = fact_n), levels = fact_label)
)

# Two by two by two, for the column order of a three-way interaction block.
cube_lv <- list(a = c("a1", "a2"), b = c("b1", "b2"), c = c("c1", "c2"))
cube_cells <- STATassist:::sa_fact_grid(cube_lv)

# The same three-by-two design with one cell never observed, which is the one
# case where the cell matrix is not of full rank and `df` has to come off the two
# ranks rather than off a formula.
holed_cells <- fact_cells[-3, , drop = FALSE]
holed_label <- fact_label[-3]
holed_samples <- fact_samples[-3]

# Contingency tables. Written out rather than drawn, so each one is the shape it
# is here for a reason: `t2x2` has every cell filled, `t2x2_zero` has one empty
# and sends the odds ratio through the Haldane-Anscombe correction, `t2x4` is the
# one that separates R's Yates rule (2 x 2 only) from scipy's (one degree of
# freedom), `t3x4` is a plain r x c and `t3x3_small` is sparse enough that the
# exact test is the one to read.
mkt <- function(values, rows, cols, nrow) {
  matrix(as.integer(values), nrow = nrow,
         dimnames = list(row_level = rows, col_level = cols))
}
tab_2x2 <- mkt(c(21, 9, 8, 22), c("no", "yes"), c("no", "yes"), 2L)
tab_2x2_zero <- mkt(c(14, 0, 5, 11), c("no", "yes"), c("no", "yes"), 2L)
tab_2x4 <- mkt(c(12, 7, 9, 14, 6, 11, 13, 5),
               c("no", "yes"), c("q1", "q2", "q3", "q4"), 2L)
tab_3x4 <- mkt(c(14, 9, 6, 11, 17, 8, 7, 6, 15, 5, 12, 10),
               c("low", "mid", "high"), c("q1", "q2", "q3", "q4"), 3L)
tab_3x3_small <- mkt(c(3, 1, 0, 2, 4, 1, 1, 2, 5),
                     c("low", "mid", "high"), c("x", "y", "z"), 3L)
# Twelve thousand tables hold these margins, which is few enough for the Python
# port to enumerate and many enough that it has to prune rather than walk them
# all. The exact p-value here is what pins that pruning down.
tab_3x3_mid <- mkt(c(8, 4, 2, 5, 9, 4, 3, 5, 10),
                   c("low", "mid", "high"), c("x", "y", "z"), 3L)
# Large enough that the network algorithm's workspace is exhausted, which the
# kernel reports as `enumerated = 0` rather than as an error.
tab_big <- mkt(c(18, 12, 9, 7, 11, 14, 16, 8, 6, 13, 10, 15,
                 9, 17, 12, 11, 14, 8, 13, 10, 7, 16, 11, 9, 12),
               paste0("r", 1:5), paste0("c", 1:5), 5L)

# Matched pairs: the same thing measured twice, crossed against itself. The
# first has few enough discordant pairs to send `exact = NULL` down the exact
# branch, the second has enough to send it down the chi-square one.
pair_small <- mkt(c(28, 9, 3, 14), c("no", "yes"), c("no", "yes"), 2L)
pair_large <- mkt(c(64, 31, 17, 48), c("no", "yes"), c("no", "yes"), 2L)
pair_one_way <- mkt(c(30, 11, 0, 15), c("no", "yes"), c("no", "yes"), 2L)

# One long frame per table, so several tables share one input file and the port
# rebuilds each by name.
tab_long <- function(tables) {
  do.call(rbind, lapply(names(tables), function(nm) {
    m <- tables[[nm]]
    grid <- expand.grid(row_level = rownames(m), col_level = colnames(m),
                        KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
    data.frame(table = nm, row_level = grid$row_level,
               col_level = grid$col_level, count = as.numeric(m),
               stringsAsFactors = FALSE)
  }))
}
cat_df <- tab_long(list(
  t2x2 = tab_2x2, t2x2_zero = tab_2x2_zero, t2x4 = tab_2x4, t3x4 = tab_3x4,
  t3x3_small = tab_3x3_small, t3x3_mid = tab_3x3_mid, t_big = tab_big,
  pair_small = pair_small, pair_large = pair_large, pair_one_way = pair_one_way
))

# Repeated binary conditions, for Cochran's Q and Kendall's W. Subject 1 answered
# the same way throughout, which contributes nothing to the numerator and is kept
# in the denominator where the formula puts it.
cochran_mat <- rbind(
  c(1, 1, 1), c(1, 0, 0), c(1, 1, 0), c(0, 1, 1), c(1, 0, 1), c(0, 0, 1),
  c(1, 1, 0), c(0, 1, 0), c(1, 1, 1), c(0, 0, 0), c(1, 0, 1), c(1, 1, 0)
)
colnames(cochran_mat) <- c("cond_1", "cond_2", "cond_3")

# A two-class outcome with two sets of predicted probabilities. Ties are planted
# on purpose: the ROC steps diagonally through them and the placement values
# count them as half, and a port that sorts them differently disagrees here
# first.
perf_response <- c(0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0)
perf_old <- c(0.12, 0.55, 0.31, 0.62, 0.44, 0.31, 0.18, 0.77, 0.55, 0.24,
              0.68, 0.09, 0.71, 0.40, 0.40, 0.22, 0.83, 0.15)
perf_new <- c(0.08, 0.66, 0.28, 0.74, 0.51, 0.35, 0.14, 0.81, 0.49, 0.19,
              0.72, 0.11, 0.79, 0.46, 0.33, 0.26, 0.88, 0.10)
perf_df <- data.frame(response = perf_response, predictor_old = perf_old,
                      predictor_new = perf_new)

# One event only, where a class-wise variance has nothing to be about and the
# standard error is absent rather than zero.
thin_response <- c(0, 0, 1, 0, 0, 0)
thin_old <- c(0.10, 0.22, 0.61, 0.35, 0.18, 0.27)
thin_new <- c(0.08, 0.25, 0.72, 0.31, 0.20, 0.24)

# Points in two dimensions with a cluster label each. Cluster 3 is a singleton,
# rows 11 and 12 are noise, and rows 5 and 6 sit on top of each other so that the
# coincident case is covered.
sil_x <- c(0.0, 0.4, 0.2, 0.6, 1.0, 1.0, 5.0, 5.4, 5.2, 5.6, 2.7, 3.1, 9.0)
sil_y <- c(0.0, 0.3, 0.5, 0.1, 0.8, 0.8, 5.0, 5.3, 4.8, 5.1, 2.5, 2.9, 9.0)
sil_cluster <- c(1L, 1L, 1L, 1L, 1L, 1L, 2L, 2L, 2L, 2L, 0L, 0L, 3L)
sil_df <- data.frame(x = sil_x, y = sil_y, cluster = sil_cluster)


# --------------------------------------------------------------------------- #
# kernel_robust.R
# --------------------------------------------------------------------------- #

t_params <- data.frame(
  stat = c(2.31, -2.31, 0.4, 2.31, 2.31, -1.1, 0),
  df = c(7, 7, 13.4, 7, 7, 20, 5),
  alternative = c("two.sided", "two.sided", "two.sided", "greater", "less",
                  "greater", "two.sided"),
  stringsAsFactors = FALSE
)
write_case(
  "robust_t_pval",
  t_params,
  list(pval = vapply(seq_len(nrow(t_params)), function(i) {
    STATassist:::sa_t_pval(t_params$stat[i], t_params$df[i],
                           t_params$alternative[i])
  }, numeric(1)))
)

ci_params <- data.frame(
  est = c(1.25, 1.25, 1.25, 0.62, 0.62),
  se = c(0.4, 0.4, 0.4, 0.11, 0.11),
  df = c(9, 9, 9, 15.7, 15.7),
  alternative = c("two.sided", "greater", "less", "greater", "less"),
  conf_level = c(0.95, 0.95, 0.95, 0.9, 0.9),
  lower_bound = c(-Inf, -Inf, -Inf, 0, 0),
  upper_bound = c(Inf, Inf, Inf, 1, 1),
  stringsAsFactors = FALSE
)
write_case(
  "robust_t_ci",
  ci_params,
  as_cols(t(vapply(seq_len(nrow(ci_params)), function(i) {
    ci <- STATassist:::sa_t_ci(
      ci_params$est[i], ci_params$se[i], ci_params$df[i],
      ci_params$alternative[i], ci_params$conf_level[i],
      bounds = c(ci_params$lower_bound[i], ci_params$upper_bound[i])
    )
    c(lower_conf = ci[1], upper_conf = ci[2])
  }, numeric(2))))
)

write_case(
  "robust_winsorize",
  data.frame(value = out_value),
  list(
    tr_0 = STATassist:::sa_winsorize(out_value, 0),
    tr_10 = STATassist:::sa_winsorize(out_value, 0.1),
    tr_20 = STATassist:::sa_winsorize(out_value, 0.2),
    tr_45 = STATassist:::sa_winsorize(out_value, 0.45)
  )
)

win_var_tr <- c(0, 0.05, 0.1, 0.2, 0.3, 0.45)
write_case(
  "robust_winsorized_normal_var",
  data.frame(tr = win_var_tr),
  list(value = vapply(win_var_tr, STATassist:::sa_winsorized_normal_var,
                      numeric(1)))
)

write_case(
  "robust_brunner_munzel",
  two_df,
  nvs(
    two_sided = STATassist:::sa_brunner_munzel(bm_x, bm_y),
    greater = STATassist:::sa_brunner_munzel(bm_x, bm_y, "greater"),
    less = STATassist:::sa_brunner_munzel(bm_x, bm_y, "less"),
    conf_90 = STATassist:::sa_brunner_munzel(bm_x, bm_y, "two.sided", 0.90),
    tied = STATassist:::sa_brunner_munzel(tied_samples$a, tied_samples$c)
  )
)

write_case(
  "robust_yuen_paired",
  pair_df,
  nvs(
    tr_20 = STATassist:::sa_yuen_paired(pair_x, pair_y),
    tr_10 = STATassist:::sa_yuen_paired(pair_x, pair_y, tr = 0.1),
    tr_0 = STATassist:::sa_yuen_paired(pair_x, pair_y, tr = 0),
    greater = STATassist:::sa_yuen_paired(pair_x, pair_y,
                                         alternative = "greater"),
    less = STATassist:::sa_yuen_paired(pair_x, pair_y, alternative = "less"),
    conf_90 = STATassist:::sa_yuen_paired(pair_x, pair_y, conf_level = 0.90)
  )
)


# --------------------------------------------------------------------------- #
# kernel_wilcox.R
#
# Graded against stats::wilcox.test() rather than against a STATassist kernel,
# because that is what the Python side ports: SciPy reports no location estimate
# and no interval for either test, so the whole of R's is written out there.
#
# Four paths have to be covered and each one is a different piece of code in R.
# `exact` is decided by sample size, `n < 50`; ties then decide whether the exact
# distribution is the classical one over the ranks 1..n or the one induced by the
# observed rank vector, which may hold halves. The asymptotic interval is a
# numerical search over a step function, so it is the case where the port has to
# follow R's root finder and not merely R's formula.
# --------------------------------------------------------------------------- #

# Two samples large enough to take R off the exact distribution.
#
# Drawn under a seed of their own, with the main stream put back afterwards. The
# fixed inputs above are all drawn from one stream, so taking values out of it
# here would renumber every case added after this section rather than only the
# ones this section writes.
wilcox_stream <- .Random.seed
set.seed(20260822)

big_x <- r6(stats::rnorm(55, 0.4, 1))
big_y <- r6(stats::rnorm(60, 0, 1))
big_df <- data.frame(x = pad(big_x, 60), y = big_y)
big_pair <- r6(stats::rnorm(55, 0.3, 1))

.Random.seed <- wilcox_stream

# Heavy ties on both paths: small enough for the tie-aware exact distribution,
# and a longer one that lands on the asymptotic branch instead.
tie_x <- c(2, 3, 3, 4, 1, 5, 3, 4, 4, 2, 5, 3, 1, 4)
tie_y <- c(1, 2, 2, 3, 3, 1, 4, 2, 5, 3, 2, 1, 3, 4, 2, 3)
tie_two_df <- data.frame(x = pad(tie_x, 16), y = tie_y)

long_tie_x <- rep(c(1, 2, 3, 4, 5), times = 11L)
long_tie_y <- rep(c(1, 2, 2, 3, 4), times = 12L)
long_tie_df <- data.frame(x = pad(long_tie_x, 60), y = long_tie_y)

wilcox_two <- function(x, y, ...) {
  res <- suppressWarnings(
    stats::wilcox.test(x, y, conf.int = TRUE, ...)
  )
  list(w_stat = unname(res$statistic), hl_shift = unname(res$estimate),
       pval = res$p.value,
       lower_conf = res$conf.int[1], upper_conf = res$conf.int[2])
}

wilcox_one <- function(x, ...) {
  res <- suppressWarnings(
    stats::wilcox.test(x, conf.int = TRUE, ...)
  )
  list(v_stat = unname(res$statistic), hl_shift = unname(res$estimate),
       pval = res$p.value,
       lower_conf = res$conf.int[1], upper_conf = res$conf.int[2])
}

write_case(
  "wilcox_rank_sum_exact",
  two_df,
  list(
    two_sided = wilcox_two(bm_x, bm_y),
    greater = wilcox_two(bm_x, bm_y, alternative = "greater"),
    less = wilcox_two(bm_x, bm_y, alternative = "less"),
    conf_90 = wilcox_two(bm_x, bm_y, conf.level = 0.90),
    shifted = wilcox_two(bm_x, bm_y, mu = 0.5)
  )
)

write_case(
  "wilcox_rank_sum_exact_tied",
  tie_two_df,
  list(
    two_sided = wilcox_two(tie_x, tie_y),
    greater = wilcox_two(tie_x, tie_y, alternative = "greater"),
    less = wilcox_two(tie_x, tie_y, alternative = "less"),
    conf_99 = wilcox_two(tie_x, tie_y, conf.level = 0.99)
  )
)

write_case(
  "wilcox_rank_sum_asymptotic",
  big_df,
  list(
    two_sided = wilcox_two(big_x, big_y),
    greater = wilcox_two(big_x, big_y, alternative = "greater"),
    less = wilcox_two(big_x, big_y, alternative = "less"),
    no_correction = wilcox_two(big_x, big_y, correct = FALSE)
  )
)

write_case(
  "wilcox_rank_sum_asymptotic_tied",
  long_tie_df,
  list(
    two_sided = wilcox_two(long_tie_x, long_tie_y),
    greater = wilcox_two(long_tie_x, long_tie_y, alternative = "greater"),
    less = wilcox_two(long_tie_x, long_tie_y, alternative = "less")
  )
)

write_case(
  "wilcox_signed_rank_exact",
  pair_df,
  list(
    two_sided = wilcox_one(pair_x - pair_y),
    greater = wilcox_one(pair_x - pair_y, alternative = "greater"),
    less = wilcox_one(pair_x - pair_y, alternative = "less"),
    conf_90 = wilcox_one(pair_x - pair_y, conf.level = 0.90),
    against_mu = wilcox_one(pair_x, mu = 5.2)
  )
)

write_case(
  "wilcox_signed_rank_exact_tied",
  data.frame(value = tied_value),
  list(
    # A zero difference carries no sign, so it leaves the sample: mu is set on
    # the median to make sure that path is taken.
    with_zeros = wilcox_one(tied_value, mu = 3),
    greater = wilcox_one(tied_value, mu = 3, alternative = "greater"),
    off_centre = wilcox_one(tied_value, mu = 2.5)
  )
)

write_case(
  "wilcox_signed_rank_asymptotic",
  data.frame(value = big_pair, tied = pad(long_tie_x, 55)),
  list(
    two_sided = wilcox_one(big_pair),
    greater = wilcox_one(big_pair, alternative = "greater"),
    less = wilcox_one(big_pair, alternative = "less"),
    no_correction = wilcox_one(big_pair, correct = FALSE),
    tied = wilcox_one(long_tie_x, mu = 3)
  )
)


# --------------------------------------------------------------------------- #
# kernel_anova.R
# --------------------------------------------------------------------------- #

split_value <- long_value
split_value[c(2, 11, 20)] <- NA_real_
write_case(
  "anova_split_groups",
  data.frame(group = as.character(long_group), value = split_value,
             stringsAsFactors = FALSE),
  STATassist:::sa_split_groups(split_value, long_group)
)

write_case("anova_oneway", long_df, nv(STATassist:::sa_oneway_anova(samples)))
write_case("anova_welch", long_df, nv(STATassist:::sa_welch_anova(samples)))

write_case(
  "anova_yuen",
  long_df,
  nvs(
    tr_20 = STATassist:::sa_yuen_anova(samples),
    tr_10 = STATassist:::sa_yuen_anova(samples, tr = 0.1),
    tr_0 = STATassist:::sa_yuen_anova(samples, tr = 0)
  )
)

write_case(
  "anova_kruskal",
  rank_df,
  nvs(
    continuous = STATassist:::sa_kruskal(samples),
    tied = STATassist:::sa_kruskal(tied_samples)
  )
)

write_case("anova_rm", as_cols(rm_mat), nv(STATassist:::sa_rm_anova(rm_mat)))

write_case(
  "anova_sphericity",
  as_cols(rm_mat),
  nvs(
    full = STATassist:::sa_sphericity(rm_mat),
    # n <= k, where the condition covariance is singular and both epsilons fall
    # back to the lower bound.
    singular = STATassist:::sa_sphericity(rm_mat[1:3, , drop = FALSE])
  )
)

write_case("anova_friedman", as_cols(rm_mat),
           nv(STATassist:::sa_friedman(rm_mat)))


# --------------------------------------------------------------------------- #
# kernel_posthoc.R
# --------------------------------------------------------------------------- #

write_case(
  "posthoc_columns",
  data.frame(unused = 0),
  list(columns = STATassist:::sa_posthoc_columns())
)

write_case(
  "posthoc_tukey",
  long_df,
  list(
    conf_95 = as_cols(STATassist:::sa_tukey(samples)),
    conf_99 = as_cols(STATassist:::sa_tukey(samples, conf_level = 0.99))
  )
)

write_case(
  "posthoc_games_howell",
  long_df,
  list(
    conf_95 = as_cols(STATassist:::sa_games_howell(samples)),
    conf_90 = as_cols(STATassist:::sa_games_howell(samples, conf_level = 0.90))
  )
)

write_case(
  "posthoc_dunn",
  rank_df,
  list(
    continuous = as_cols(STATassist:::sa_dunn(samples)),
    continuous_99 = as_cols(STATassist:::sa_dunn(samples, conf_level = 0.99)),
    tied = as_cols(STATassist:::sa_dunn(tied_samples))
  )
)

write_case(
  "posthoc_yuen_independent",
  two_df,
  nvs(
    tr_20 = STATassist:::sa_yuen_independent(bm_x, bm_y),
    tr_10 = STATassist:::sa_yuen_independent(bm_x, bm_y, tr = 0.1),
    tr_0 = STATassist:::sa_yuen_independent(bm_x, bm_y, tr = 0),
    greater = STATassist:::sa_yuen_independent(bm_x, bm_y,
                                              alternative = "greater"),
    less = STATassist:::sa_yuen_independent(bm_x, bm_y, alternative = "less"),
    conf_90 = STATassist:::sa_yuen_independent(bm_x, bm_y, conf_level = 0.90)
  )
)

write_case(
  "posthoc_pairwise_yuen",
  long_df,
  list(
    tr_20 = as_cols(STATassist:::sa_pairwise_yuen(samples)),
    tr_10 = as_cols(STATassist:::sa_pairwise_yuen(samples, tr = 0.1,
                                                 conf_level = 0.99))
  )
)

write_case(
  "posthoc_pairwise_paired_t",
  as_cols(rm_mat),
  list(
    conf_95 = as_cols(STATassist:::sa_pairwise_paired_t(rm_mat)),
    conf_90 = as_cols(STATassist:::sa_pairwise_paired_t(rm_mat,
                                                       conf_level = 0.90))
  )
)

write_case(
  "posthoc_conover",
  as_cols(rm_mat),
  list(
    conf_95 = as_cols(STATassist:::sa_conover(rm_mat)),
    conf_99 = as_cols(STATassist:::sa_conover(rm_mat, conf_level = 0.99))
  )
)


# --------------------------------------------------------------------------- #
# kernel_diagnostic.R
# --------------------------------------------------------------------------- #

write_case(
  "diag_shapiro",
  data.frame(value = long_value, outlier = pad(out_value, 30),
             tied = pad(tied_value, 30)),
  nvs(
    all = STATassist:::sa_shapiro(long_value),
    small = STATassist:::sa_shapiro(long_value[1:8]),
    tiny = STATassist:::sa_shapiro(long_value[1:3]),
    outlier = STATassist:::sa_shapiro(out_value),
    tied = STATassist:::sa_shapiro(tied_value)
  )
)

write_case(
  "diag_ks_normal",
  data.frame(value = long_value, outlier = pad(out_value, 30),
             tied = pad(tied_value, 30)),
  nvs(
    all = STATassist:::sa_ks_normal(long_value),
    small = STATassist:::sa_ks_normal(long_value[1:8]),
    outlier = STATassist:::sa_ks_normal(out_value),
    # Ties push R off the exact p-value and onto the asymptotic one.
    tied = STATassist:::sa_ks_normal(tied_value)
  )
)

write_case(
  "diag_levene",
  long_df,
  nvs(
    median = STATassist:::sa_levene(samples),
    mean = STATassist:::sa_levene(samples, center = "mean"),
    trimmed = STATassist:::sa_levene(samples, center = "trimmed"),
    trimmed_25 = STATassist:::sa_levene(samples, center = "trimmed",
                                        trim = 0.25)
  )
)

write_case("diag_bartlett", long_df, nv(STATassist:::sa_bartlett(samples)))

write_case(
  "diag_grubbs",
  data.frame(value = pad(out_value, 30), clean = long_value),
  nvs(
    outlier = STATassist:::sa_grubbs(out_value),
    clean = STATassist:::sa_grubbs(long_value),
    tiny = STATassist:::sa_grubbs(long_value[1:3])
  )
)

write_case(
  "diag_flag_outliers",
  data.frame(value = out_value, holed = holed, short = short, flat = flat),
  list(
    iqr = STATassist:::sa_flag_outliers(out_value),
    iqr_3 = STATassist:::sa_flag_outliers(out_value, iqr_multiplier = 3),
    robust_z = STATassist:::sa_flag_outliers(out_value, criterion = "robust_z"),
    robust_z_2 = STATassist:::sa_flag_outliers(out_value,
                                              criterion = "robust_z",
                                              z_threshold = 2),
    grubbs = STATassist:::sa_flag_outliers(out_value, criterion = "grubbs"),
    grubbs_strict = STATassist:::sa_flag_outliers(out_value,
                                                 criterion = "grubbs",
                                                 alpha = 1e-6),
    holed = STATassist:::sa_flag_outliers(holed),
    holed_z = STATassist:::sa_flag_outliers(holed, criterion = "robust_z"),
    holed_grubbs = STATassist:::sa_flag_outliers(holed, criterion = "grubbs"),
    short = STATassist:::sa_flag_outliers(short),
    flat = STATassist:::sa_flag_outliers(flat),
    flat_z = STATassist:::sa_flag_outliers(flat, criterion = "robust_z"),
    flat_grubbs = STATassist:::sa_flag_outliers(flat, criterion = "grubbs")
  )
)


# --------------------------------------------------------------------------- #
# utils_describe.R and summarize_descriptive_stats.R
# --------------------------------------------------------------------------- #

describe_vectors <- list(
  plain = long_value,
  outlier = out_value,
  tied = tied_value,
  four = long_value[1:4],
  three = long_value[1:3],
  single = long_value[1],
  flat = flat,
  holed = holed,
  empty = numeric(0),
  all_missing = rep(NA_real_, 5)
)
write_case(
  "describe_vector",
  data.frame(value = long_value),
  c(
    list(columns = STATassist:::sa_describe_columns()),
    lapply(describe_vectors,
           function(v) nv(STATassist:::sa_describe_vector(v))),
    list(
      skewness = nv(vapply(describe_vectors, function(v) {
        v <- v[is.finite(v)]
        if (length(v) == 0L) NA_real_ else STATassist:::sa_skewness(v)
      }, numeric(1))),
      kurtosis = nv(vapply(describe_vectors, function(v) {
        v <- v[is.finite(v)]
        if (length(v) == 0L) NA_real_ else STATassist:::sa_kurtosis(v)
      }, numeric(1)))
    )
  )
)

write_case(
  "descriptive_ungrouped",
  wide,
  list(
    all = STATassist::summarize_descriptive_stats(
      wide, c("gene_1", "gene_2", "gene_3")
    ),
    one = STATassist::summarize_descriptive_stats(wide, "gene_2"),
    reordered = STATassist::summarize_descriptive_stats(
      wide, c("gene_3", "gene_1")
    )
  )
)

write_case(
  "descriptive_grouped",
  wide,
  list(
    all = STATassist::summarize_descriptive_stats(
      wide, c("gene_1", "gene_2"), wide$group
    ),
    ordered = STATassist::summarize_descriptive_stats(
      wide, c("gene_2", "gene_1"), wide$group,
      c("treat_b", "ctrl", "treat_a")
    ),
    subset = suppressMessages(STATassist::summarize_descriptive_stats(
      wide, "gene_1", wide$group, c("ctrl", "treat_a")
    ))
  )
)


# --------------------------------------------------------------------------- #
# utils_associate.R and summarize_association_stats.R
# --------------------------------------------------------------------------- #

assoc <- wide[c("gene_1", "gene_2", "gene_3")]
assoc$gene_4 <- r6(c(stats::rnorm(20, 3, 1), rep(NA_real_, 4)))
assoc$gene_5 <- rep(7, 24)

# A short, tie-free pair for the exact branches and a tied one for the fallback.
assoc_small <- data.frame(
  a = c(1.5, 2.25, 3.75, 0.5, 4.5, 2.75, 6.25, 5.5),
  b = c(2.5, 1.25, 4.25, 1.5, 5.5, 3.75, 5.25, 6.75),
  c = c(1, 1, 2, 2, 3, 3, 4, 4)
)

write_case(
  "association_cor_test_pvalue",
  assoc_small,
  list(
    pearson_ab = STATassist:::sa_cor_test_pvalue(assoc_small$a, assoc_small$b,
                                                 "pearson"),
    # n = 8 is inside the enumerated range of R's exact Spearman.
    spearman_ab = STATassist:::sa_cor_test_pvalue(assoc_small$a, assoc_small$b,
                                                  "spearman"),
    kendall_ab = STATassist:::sa_cor_test_pvalue(assoc_small$a, assoc_small$b,
                                                 "kendall"),
    # Ties: R leaves the exact path for the asymptotic one.
    spearman_ac = STATassist:::sa_cor_test_pvalue(assoc_small$a, assoc_small$c,
                                                  "spearman"),
    kendall_ac = STATassist:::sa_cor_test_pvalue(assoc_small$a, assoc_small$c,
                                                 "kendall"),
    # n = 24 leaves the enumerated range for the Edgeworth series.
    pearson_long = STATassist:::sa_cor_test_pvalue(wide$gene_1, wide$gene_2,
                                                  "pearson"),
    spearman_long = STATassist:::sa_cor_test_pvalue(wide$gene_1, wide$gene_2,
                                                    "spearman"),
    kendall_long = STATassist:::sa_cor_test_pvalue(wide$gene_1, wide$gene_2,
                                                   "kendall"),
    pearson_corr = STATassist:::sa_cor_test_pvalue(wide$gene_1, wide$gene_3,
                                                   "pearson"),
    spearman_corr = STATassist:::sa_cor_test_pvalue(wide$gene_1, wide$gene_3,
                                                    "spearman"),
    kendall_corr = STATassist:::sa_cor_test_pvalue(wide$gene_1, wide$gene_3,
                                                   "kendall"),
    # Refused: no variance on one side, and fewer than three shared values.
    flat = STATassist:::sa_cor_test_pvalue(assoc$gene_1, assoc$gene_5,
                                          "pearson"),
    tiny = STATassist:::sa_cor_test_pvalue(c(1, 2), c(3, 5), "pearson")
  )
)

# The two exact distributions scipy has no equivalent for, tabulated straight
# from the C routines cor.test calls.
rho_grid <- do.call(rbind, lapply(2:14, function(n) {
  top <- (n^3 - n) / 3
  data.frame(n = n,
             s = unique(c(-1, 0, 2, 4, round(top / 2), top - 2, top, top + 2)))
}))
write_case(
  "association_p_rho",
  rho_grid,
  list(
    lower = vapply(seq_len(nrow(rho_grid)), function(i) {
      .Call(stats:::C_pRho, round(rho_grid$s[i]) + 2L, rho_grid$n[i], TRUE)
    }, numeric(1)),
    upper = vapply(seq_len(nrow(rho_grid)), function(i) {
      .Call(stats:::C_pRho, round(rho_grid$s[i]), rho_grid$n[i], FALSE)
    }, numeric(1))
  )
)

kendall_grid <- do.call(rbind, lapply(2:12, function(n) {
  top <- n * (n - 1) / 2
  data.frame(n = n, q = unique(c(-1, 0, 1, round(top / 2), top, top + 1)))
}))
write_case(
  "association_p_kendall",
  kendall_grid,
  list(value = vapply(seq_len(nrow(kendall_grid)), function(i) {
    .Call(stats:::C_pKendall, kendall_grid$q[i], kendall_grid$n[i])
  }, numeric(1)))
)

assoc_matrix <- as.matrix(assoc)
assoc_matrix[!is.finite(assoc_matrix)] <- NA
write_case(
  "association_matrices",
  assoc,
  list(
    pearson = STATassist:::sa_association_matrices(assoc_matrix, "pearson",
                                                   "BH"),
    spearman = STATassist:::sa_association_matrices(assoc_matrix, "spearman",
                                                    "holm"),
    kendall = STATassist:::sa_association_matrices(assoc_matrix, "kendall",
                                                   "bonferroni"),
    pairwise_n = STATassist:::sa_pairwise_n(assoc_matrix)
  )
)

write_case(
  "association_public",
  assoc,
  list(
    pairwise = suppressMessages(STATassist::summarize_association_stats(
      assoc, c("gene_1", "gene_2", "gene_3", "gene_4", "gene_5")
    )),
    complete = suppressMessages(STATassist::summarize_association_stats(
      assoc, c("gene_1", "gene_2", "gene_4"), use = "complete.obs",
      adj_type = "holm"
    )),
    spearman_only = suppressMessages(STATassist::summarize_association_stats(
      assoc, c("gene_1", "gene_3", "gene_4"), methods = "spearman",
      adj_type = "bonferroni"
    ))
  )
)


# --------------------------------------------------------------------------- #
# screen_outliers.R and diagnose_distribution.R
# --------------------------------------------------------------------------- #

screen_wide <- wide
screen_wide$gene_1[c(2, 19)] <- screen_wide$gene_1[c(2, 19)] + 12
screen_wide$gene_2[5] <- NA_real_

screen_case <- function(...) {
  out <- suppressMessages(STATassist::screen_outliers(...))
  list(
    table = out,
    settings = list(criterion = attr(out, "criterion"),
                    iqr_multiplier = attr(out, "iqr_multiplier"),
                    z_threshold = attr(out, "z_threshold"),
                    alpha = attr(out, "alpha")),
    n_rows = nrow(out)
  )
}

write_case(
  "screen_outliers",
  screen_wide,
  list(
    ungrouped = screen_case(screen_wide, c("gene_1", "gene_2")),
    grouped = screen_case(screen_wide, c("gene_1", "gene_2"),
                          screen_wide$group),
    subset = screen_case(screen_wide, "gene_1", screen_wide$group,
                         c("treat_a", "treat_b")),
    robust_z = screen_case(screen_wide, "gene_1", criterion = "robust_z",
                           z_threshold = 2),
    grubbs = screen_case(screen_wide, "gene_1", criterion = "grubbs"),
    # A fence wide enough that nothing is flagged: the zero-row table.
    wide_fence = screen_case(screen_wide, "gene_2", iqr_multiplier = 10)
  )
)

diag_case <- function(...) {
  res <- suppressMessages(STATassist::diagnose_distribution(...))
  list(
    analysis = res$analysis,
    features = res$features,
    design = res$design,
    parameters = res$parameters,
    normality = res$normality,
    variance = res$variance,
    outliers = res$outliers,
    summary = res$summary,
    n_variance_rows = nrow(res$variance)
  )
}

write_case(
  "diagnose_ungrouped",
  screen_wide,
  list(
    plain = diag_case(screen_wide, c("gene_1", "gene_2", "gene_3")),
    one = diag_case(screen_wide, "gene_1", alpha = 0.1)
  )
)

write_case(
  "diagnose_grouped",
  screen_wide,
  list(
    plain = diag_case(screen_wide, c("gene_1", "gene_2"), screen_wide$group),
    mean_centre = diag_case(screen_wide, "gene_1", screen_wide$group,
                            center = "mean", alpha = 0.1),
    trimmed = diag_case(screen_wide, "gene_1", screen_wide$group,
                        center = "trimmed", trim = 0.25),
    robust_z = diag_case(screen_wide, "gene_1", screen_wide$group,
                         criterion = "robust_z", z_threshold = 2)
  )
)


# --------------------------------------------------------------------------- #
# utils_foldchange.R and center_by_control.R
# --------------------------------------------------------------------------- #

fc_feats <- c("prot_1", "prot_2")
fc_log2 <- fc_wide
fc_log2[fc_feats] <- r6(log2(fc_wide[fc_feats]))

write_case(
  "foldchange_center",
  fc_wide,
  list(
    arith_raw = STATassist:::sa_fc_center(fc_wide$prot_1, "case", "arith"),
    geom_raw = STATassist:::sa_fc_center(fc_wide$prot_1, "case", "geom"),
    arith_log2 = STATassist:::sa_fc_center(r6(log2(fc_wide$prot_1)), "case",
                                          "arith", input_scale = "log2"),
    geom_log2 = STATassist:::sa_fc_center(r6(log2(fc_wide$prot_1)), "case",
                                         "geom", input_scale = "log2"),
    resolved = list(
      default_raw = STATassist:::sa_resolve_fc_mean(c("arith", "geom"), "raw",
                                                   TRUE),
      default_log2 = STATassist:::sa_resolve_fc_mean(c("arith", "geom"),
                                                     "log2", TRUE),
      explicit_arith = STATassist:::sa_resolve_fc_mean("arith", "log2", FALSE),
      explicit_geom = STATassist:::sa_resolve_fc_mean("geom", "raw", FALSE)
    )
  )
)

fc_pairs <- lapply(fc_feats, function(f) {
  list(x = fc_wide[[f]][fc_wide$group == "case"],
       y = fc_wide[[f]][fc_wide$group == "ctrl"])
})
names(fc_pairs) <- fc_feats
fc_pairs_log2 <- lapply(fc_feats, function(f) {
  list(x = fc_log2[[f]][fc_log2$group == "case"],
       y = fc_log2[[f]][fc_log2$group == "ctrl"])
})
names(fc_pairs_log2) <- fc_feats

write_case(
  "foldchange_table",
  fc_wide,
  list(
    arith = suppressMessages(STATassist:::sa_fold_change(
      fc_pairs, fc_feats, c("case", "ctrl"), "arith"
    )),
    geom = suppressMessages(STATassist:::sa_fold_change(
      fc_pairs, fc_feats, c("case", "ctrl"), "geom"
    )),
    geom_log2 = suppressMessages(STATassist:::sa_fold_change(
      fc_pairs_log2, fc_feats, c("case", "ctrl"), "geom",
      input_scale = "log2"
    ))
  )
)

write_case(
  "center_by_control",
  fc_wide,
  list(
    raw_arith = suppressMessages(STATassist::center_by_control(
      fc_wide, fc_feats, fc_wide$group, c("ctrl", "case")
    ))[fc_feats],
    raw_geom = suppressMessages(STATassist::center_by_control(
      fc_wide, fc_feats, fc_wide$group, c("ctrl", "case"), fc_mean = "geom"
    ))[fc_feats],
    other_control = suppressMessages(STATassist::center_by_control(
      fc_wide, fc_feats, fc_wide$group, c("ctrl", "case", "other"),
      control_label = "case"
    ))[fc_feats],
    one_feature = suppressMessages(STATassist::center_by_control(
      fc_wide, "prot_2", fc_wide$group, c("other", "ctrl")
    ))["prot_2"]
  )
)

write_case(
  "center_by_control_log2",
  fc_log2,
  list(
    # The default centre is scale dependent: log2 data centres on the geometric
    # mean unless the caller says otherwise.
    default_geom = suppressMessages(STATassist::center_by_control(
      fc_log2, fc_feats, fc_log2$group, c("ctrl", "case"),
      input_scale = "log2"
    ))[fc_feats],
    explicit_arith = suppressMessages(STATassist::center_by_control(
      fc_log2, fc_feats, fc_log2$group, c("ctrl", "case"),
      fc_mean = "arith", input_scale = "log2"
    ))[fc_feats]
  )
)


# --------------------------------------------------------------------------- #
# compare_two_groups.R
#
# The whole object rather than one table: what is being graded is the assembly as
# much as the numbers, so `effect` and all three test tables go out together and
# `design` goes with them, since which rows were paired and which were dropped is
# what the numbers rest on.
# --------------------------------------------------------------------------- #

two_case <- function(...) {
  res <- suppressMessages(suppressWarnings(STATassist::compare_two_groups(...)))
  list(
    group_lv      = res$design$group_lv,
    pairing       = res$design$pairing,
    n_dropped     = res$design$n_dropped,
    unmatched_ids = res$design$unmatched_ids,
    tr            = res$parameters$tr,
    fc_mean       = res$parameters$fc_mean,
    effect        = as_cols(res$effect),
    t_test        = as_cols(res$tests$t_test),
    wilcox_test   = as_cols(res$tests$wilcox_test),
    robust_test   = as_cols(res$tests$robust_test)
  )
}

write_case(
  "two_group_independent",
  fc_wide,
  list(
    plain = two_case(fc_wide, fc_feats, fc_wide$group, c("ctrl", "case"),
                     diagnose = FALSE),
    greater = two_case(fc_wide, fc_feats, fc_wide$group, c("ctrl", "case"),
                       alternative = "greater", diagnose = FALSE),
    less_90 = two_case(fc_wide, fc_feats, fc_wide$group, c("ctrl", "case"),
                       alternative = "less", conf_level = 0.90,
                       diagnose = FALSE),
    # `control_label` names the second level, which reverses every difference
    # and ratio without the pair being rewritten.
    reversed = two_case(fc_wide, fc_feats, fc_wide$group, c("ctrl", "case"),
                        control_label = "case", diagnose = FALSE),
    geom = two_case(fc_wide, fc_feats, fc_wide$group, c("ctrl", "case"),
                    fc_mean = "geom", diagnose = FALSE),
    # A third level present in `group` and absent from `group_lv`: its rows are
    # dropped rather than tested.
    dropped = two_case(fc_wide, fc_feats, fc_wide$group,
                       c("other", "case"), p_adjust = "holm",
                       diagnose = FALSE)
  )
)

write_case(
  "two_group_log2",
  fc_log2,
  list(
    # Only `effect` moves onto the raw scale; `x_mean` and `y_mean` in the t-test
    # stay on the log2 scale the tests ran on.
    default_geom = two_case(fc_log2, fc_feats, fc_log2$group,
                            c("ctrl", "case"), input_scale = "log2",
                            diagnose = FALSE),
    explicit_arith = two_case(fc_log2, fc_feats, fc_log2$group,
                              c("ctrl", "case"), fc_mean = "arith",
                              input_scale = "log2", diagnose = FALSE)
  )
)

# Sixteen subjects measured twice. The second feature is built from the two
# samples the other way round, so a port that pairs the wrong way disagrees on it
# and not only on the shared one.
pair_long <- data.frame(
  metab_1 = c(pair_x, pair_y),
  metab_2 = r6(c(pair_y + 0.5, pair_x - 0.3)),
  group   = rep(c("post", "pre"), each = length(pair_x)),
  subject = rep(sprintf("s%02d", seq_along(pair_x)), times = 2L),
  stringsAsFactors = FALSE
)
pair_feats <- c("metab_1", "metab_2")

# The same data with the `pre` block reordered. Row order pairing silently uses
# the wrong partners here and `id` recovers the correct answer, which is the one
# failure mode `id` exists for.
pair_shuffled <- pair_long[c(1:16, 16 + c(4, 9, 1, 7, 2, 10, 3, 6, 8, 5,
                                          14, 12, 16, 11, 15, 13)), ]

# One subject measured only once, so its id is dropped from the pairing.
pair_holed <- pair_long[-20, ]

write_case(
  "two_group_paired",
  pair_long,
  list(
    by_order = two_case(pair_long, pair_feats, pair_long$group,
                        c("pre", "post"), paired = TRUE, diagnose = FALSE),
    by_id = two_case(pair_long, pair_feats, pair_long$group, c("pre", "post"),
                     id = pair_long$subject, paired = TRUE, diagnose = FALSE),
    tr_10 = two_case(pair_long, pair_feats, pair_long$group, c("pre", "post"),
                     id = pair_long$subject, paired = TRUE, tr = 0.1,
                     diagnose = FALSE),
    greater = two_case(pair_long, pair_feats, pair_long$group,
                       c("pre", "post"), id = pair_long$subject, paired = TRUE,
                       alternative = "greater", diagnose = FALSE)
  )
)

write_case(
  "two_group_paired_id",
  pair_shuffled,
  list(
    shuffled_by_order = two_case(pair_shuffled, pair_feats,
                                 pair_shuffled$group, c("pre", "post"),
                                 paired = TRUE, diagnose = FALSE),
    shuffled_by_id = two_case(pair_shuffled, pair_feats, pair_shuffled$group,
                              c("pre", "post"), id = pair_shuffled$subject,
                              paired = TRUE, diagnose = FALSE)
  )
)

write_case(
  "two_group_unmatched",
  pair_holed,
  list(
    holed = two_case(pair_holed, pair_feats, pair_holed$group,
                     c("pre", "post"), id = pair_holed$subject, paired = TRUE,
                     diagnose = FALSE)
  )
)

# A feature with holes and a feature with none left to test, so the per-feature
# missing-value handling and the all-NA row are both graded.
gappy <- fc_wide[fc_wide$group %in% c("ctrl", "case"), ]
gappy$prot_1[c(2, 3, 9)] <- NA_real_
gappy$flat <- 4
gappy$flat[gappy$group == "case"] <- 4

write_case(
  "two_group_gappy",
  gappy,
  list(
    holes = two_case(gappy, c("prot_1", "prot_2", "flat"), gappy$group,
                     c("ctrl", "case"), diagnose = FALSE)
  )
)


# --------------------------------------------------------------------------- #
# compare_one_sample.R
# --------------------------------------------------------------------------- #

one_case <- function(...) {
  res <- suppressMessages(suppressWarnings(STATassist::compare_one_sample(...)))
  list(
    mu          = res$design$mu,
    p           = res$design$p,
    success     = res$design$success,
    fc_mean     = res$parameters$fc_mean,
    effect      = as_cols(res$effect),
    t_test      = as_cols(res$tests$t_test),
    wilcox_test = as_cols(res$tests$wilcox_test),
    prop_test   = as_cols(res$tests$prop_test)
  )
}

# One continuous feature with holes, one strictly positive one, and a 0/1 coded
# one so the proportion test has something it applies to. Built out of the fixed
# inputs rather than drawn, so this section takes nothing out of the stream the
# cases below it are drawn from.
one_df <- data.frame(
  conc  = fc_wide$prot_2,
  level = fc_wide$prot_1,
  flag  = c(1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0)
)
one_df$conc[c(3, 14)] <- NA_real_
one_feats <- c("conc", "level", "flag")

write_case(
  "one_sample",
  one_df,
  list(
    plain = one_case(one_df, one_feats, mu = 5, p = 0.5),
    greater = one_case(one_df, one_feats, mu = 5, p = 0.6,
                       alternative = "greater"),
    less_90 = one_case(one_df, one_feats, mu = 5, p = 0.4,
                       alternative = "less", conf_level = 0.90),
    geom = one_case(one_df, one_feats, mu = 5, fc_mean = "geom"),
    # `mu` at zero is the one value a ratio cannot be taken against, and it is
    # also the default.
    mu_zero = one_case(one_df, one_feats, p = 0.5, p_adjust = "holm"),
    # The success value counted is not the only one a binary feature can hold.
    success_zero = one_case(one_df, "flag", mu = 0.5, p = 0.5, success = 0)
  )
)

one_log2 <- one_df
one_log2$level <- r6(log2(one_df$level))
write_case(
  "one_sample_log2",
  one_log2,
  list(
    # On the log2 scale the reference is 2^mu, positive whatever mu is, so
    # mu = 0 means a reference of 1 rather than an undefined ratio.
    default_geom = one_case(one_log2, "level", mu = 2, input_scale = "log2"),
    mu_zero = one_case(one_log2, "level", mu = 0, input_scale = "log2"),
    explicit_arith = one_case(one_log2, "level", mu = 2, fc_mean = "arith",
                              input_scale = "log2")
  )
)

# The proportion test on its own, over the shapes that separate R's corrections:
# a count exactly at the null, one at a boundary, and one small enough that the
# Yates cap binds.
prop_params <- data.frame(
  n_success = c(14, 9, 18, 1, 5, 5, 5, 1),
  n = c(18, 18, 18, 18, 10, 10, 10, 2),
  p = c(0.5, 0.5, 0.5, 0.05, 0.5, 0.3, 0.8, 0.5),
  alternative = c("two.sided", "two.sided", "two.sided", "two.sided",
                  "greater", "less", "two.sided", "two.sided"),
  conf_level = c(0.95, 0.95, 0.95, 0.95, 0.95, 0.90, 0.99, 0.95),
  stringsAsFactors = FALSE
)
write_case(
  "one_sample_prop",
  prop_params,
  as_cols(t(vapply(seq_len(nrow(prop_params)), function(i) {
    v <- c(rep(1, prop_params$n_success[i]),
           rep(0, prop_params$n[i] - prop_params$n_success[i]))
    STATassist:::sa_one_sample_prop(v, prop_params$p[i], 1,
                                    prop_params$alternative[i],
                                    prop_params$conf_level[i])
  }, numeric(11))))
)


# --------------------------------------------------------------------------- #
# compare_multiple_groups.R
#
# The whole object again, and here the pairwise stage is most of it: `posthoc` is
# the ragged record of what was asked and `pairwise` is the rectangular view of
# the same numbers, so both go out and a port that rectangularises the wrong way
# round disagrees on one of them.
# --------------------------------------------------------------------------- #

multi_case <- function(...) {
  res <- suppressMessages(suppressWarnings(
    STATassist::compare_multiple_groups(...)
  ))
  out <- list(
    group_lv      = res$design$group_lv,
    pairing       = res$design$pairing,
    n_dropped     = res$design$n_dropped,
    unmatched_ids = res$design$unmatched_ids,
    tr            = res$parameters$tr,
    fc_mean       = res$parameters$fc_mean,
    n_posthoc     = nv(res$parameters$n_posthoc),
    effect        = as_cols(res$effect),
    tests         = lapply(res$tests, as_cols)
  )
  if (!is.null(res$posthoc)) {
    out$posthoc <- lapply(res$posthoc, as_cols)
    out$pairwise <- lapply(res$pairwise, function(by_contrast) {
      lapply(by_contrast, as_cols)
    })
  }
  out
}

multi_feats <- c("gene_1", "gene_2", "gene_3")
multi_lv <- c("ctrl", "treat_a", "treat_b")

write_case(
  "multi_group_independent",
  wide,
  list(
    # `posthoc_alpha = 1` on purpose: with three features of noise the default
    # would leave the pairwise stage empty, and an empty table grades nothing.
    all_posthoc = multi_case(wide, multi_feats, wide$group, multi_lv,
                             posthoc_alpha = 1, diagnose = FALSE),
    # The default threshold, so which features qualify is graded too.
    default_alpha = multi_case(wide, multi_feats, wide$group, multi_lv,
                               diagnose = FALSE),
    # `control_label` names the last level, which re-points every ratio and puts
    # that level on the subtracted side of every contrast.
    reversed = multi_case(wide, multi_feats, wide$group, multi_lv,
                          control_label = "treat_b", posthoc_alpha = 1,
                          diagnose = FALSE),
    tuned = multi_case(wide, multi_feats, wide$group, multi_lv,
                       conf_level = 0.90, tr = 0.1, posthoc_alpha = 1,
                       p_adjust = "holm", posthoc_p_adjust = "BH",
                       diagnose = FALSE),
    no_posthoc = multi_case(wide, multi_feats, wide$group, multi_lv,
                            posthoc = FALSE, diagnose = FALSE)
  )
)

# Four levels of unequal size and spread, which is what separates the pooled
# tests from the Welch family, and one feature so the whole object is small
# enough to read.
write_case(
  "multi_group_unbalanced",
  long_df,
  list(
    four_levels = multi_case(long_df, "value", long_df$group,
                             c("ctrl", "low", "mid", "high"),
                             posthoc_alpha = 1, diagnose = FALSE),
    # Three of the four named, so the fourth level's rows are dropped rather
    # than tested.
    dropped = multi_case(long_df, "value", long_df$group,
                         c("ctrl", "mid", "high"), posthoc_alpha = 1,
                         diagnose = FALSE)
  )
)

# Strictly positive, so the geometric centre and the log2 pipeline both exist.
# All three of its levels are named here, where the two-group cases kept two.
fc_lv <- c("ctrl", "case", "other")

write_case(
  "multi_group_effect",
  fc_wide,
  list(
    arith = multi_case(fc_wide, fc_feats, fc_wide$group, fc_lv,
                       posthoc_alpha = 1, diagnose = FALSE),
    geom = multi_case(fc_wide, fc_feats, fc_wide$group, fc_lv,
                      fc_mean = "geom", posthoc_alpha = 1, diagnose = FALSE)
  )
)

write_case(
  "multi_group_log2",
  fc_log2,
  list(
    # Only `effect` moves back onto the raw scale; the tests stay on the log2
    # values they were handed.
    default_geom = multi_case(fc_log2, fc_feats, fc_log2$group, fc_lv,
                              input_scale = "log2", posthoc_alpha = 1,
                              diagnose = FALSE),
    explicit_arith = multi_case(fc_log2, fc_feats, fc_log2$group, fc_lv,
                                fc_mean = "arith", input_scale = "log2",
                                posthoc_alpha = 1, diagnose = FALSE)
  )
)

# Twelve subjects under three conditions, in long format with a subject key.
# Built from the same matrix the repeated kernels are graded on, so a
# disagreement here is about the assembly rather than about the test.
rep_long <- data.frame(
  score   = as.numeric(rm_mat),
  cond    = rep(colnames(rm_mat), each = nrow(rm_mat)),
  subject = rep(sprintf("s%02d", seq_len(nrow(rm_mat))),
                times = ncol(rm_mat)),
  stringsAsFactors = FALSE
)
# A second feature built from the conditions the other way round, so a port that
# lines the rectangle up wrongly disagrees on it and not only on the shared one.
rep_long$shifted <- r6(rev(rep_long$score) + 0.25)
rep_feats <- c("score", "shifted")
rep_lv <- colnames(rm_mat)

# One subject measured under only two of the three conditions, so its rows are
# dropped whole.
rep_holed <- rep_long[!(rep_long$subject == "s05" & rep_long$cond == "t2"), ]

# A hole in one feature only, which costs that feature one subject and leaves
# the other with all twelve.
rep_gappy <- rep_long
rep_gappy$score[rep_gappy$subject == "s03" & rep_gappy$cond == "t3"] <- NA_real_

write_case(
  "multi_group_repeated",
  rep_long,
  list(
    plain = multi_case(rep_long, rep_feats, rep_long$cond, rep_lv,
                       id = rep_long$subject, paired = TRUE,
                       posthoc_alpha = 1, diagnose = FALSE),
    tuned = multi_case(rep_long, rep_feats, rep_long$cond, rep_lv,
                       id = rep_long$subject, paired = TRUE,
                       conf_level = 0.90, posthoc_alpha = 1,
                       posthoc_p_adjust = "BH", diagnose = FALSE),
    reversed = multi_case(rep_long, rep_feats, rep_long$cond, rep_lv,
                          id = rep_long$subject, paired = TRUE,
                          control_label = "t3", posthoc_alpha = 1,
                          diagnose = FALSE)
  )
)

write_case(
  "multi_group_unmatched",
  rep_holed,
  list(
    holed = multi_case(rep_holed, rep_feats, rep_holed$cond, rep_lv,
                       id = rep_holed$subject, paired = TRUE,
                       posthoc_alpha = 1, diagnose = FALSE)
  )
)

write_case(
  "multi_group_gappy",
  rep_gappy,
  list(
    per_feature_holes = multi_case(rep_gappy, rep_feats, rep_gappy$cond, rep_lv,
                                   id = rep_gappy$subject, paired = TRUE,
                                   posthoc_alpha = 1, diagnose = FALSE)
  )
)


# --------------------------------------------------------------------------- #
# estimate_significance.R
#
# The verdict and the attributes together: the cutoffs travel with the table and
# draw_volcano_plot() reads its guides off them, so a port that computes the
# right flags and loses the attributes has not finished the job.
#
# `is_signif` is three-valued and goes out as such. jsonlite writes a logical NA
# as "NA" under `na = "string"`, which keeps "undecided" apart from FALSE.
# --------------------------------------------------------------------------- #

sig_attrs <- function(tbl) {
  keep <- c("analysis", "group_lv", "test", "test_label", "adj_type",
            "log2fc_cutoff", "pval_cutoff", "contrast", "group1", "group2")
  held <- attributes(tbl)[keep]
  names(held) <- keep
  # A NULL attribute - `group_lv` of a one-sample comparison, say - has to
  # survive as a stated absence rather than be dropped from the object.
  held[vapply(held, is.null, logical(1))] <- NA
  held
}

sig_one <- function(tbl) {
  list(table = as_cols(tbl), attrs = sig_attrs(tbl))
}

est_case <- function(...) {
  res <- suppressMessages(suppressWarnings(
    STATassist::estimate_significance(...)
  ))
  out <- list(analysis_type = res$analysis_type)
  if (is.data.frame(res$significance)) {
    out$significance <- sig_one(res$significance)
  } else {
    out$by_contrast <- lapply(res$significance, sig_one)
  }
  out
}

two_res <- suppressMessages(suppressWarnings(STATassist::compare_two_groups(
  fc_wide, fc_feats, fc_wide$group, c("ctrl", "case"), diagnose = FALSE
)))

write_case(
  "estimate_two_group",
  fc_wide,
  list(
    # The default cutoff of one log2 unit, which these two groups are nowhere
    # near, so the verdict is FALSE throughout and the magnitude rule is what
    # decided it.
    default = est_case(two_res),
    loose = est_case(two_res, log2fc_cutoff = 0.05),
    wilcox = est_case(two_res, test = "wilcox_test", log2fc_cutoff = 0.05),
    robust = est_case(two_res, test = "robust_test", log2fc_cutoff = 0.05),
    # Naming a method re-adjusts the raw column rather than the adjusted one, so
    # this is not the comparison's BH followed by Bonferroni.
    bonferroni = est_case(two_res, adj_type = "bonferroni",
                          log2fc_cutoff = 0.05),
    raw = est_case(two_res, adj_type = "none", log2fc_cutoff = 0.05,
                   pval_cutoff = 0.2)
  )
)

# A feature whose two centres have opposite signs gives log2fc = NaN, and one
# whose numerator is zero gives -Inf. The first is undecided and the second
# clears any magnitude cutoff, which is the distinction the three-valued rule
# exists for.
edge_wide <- data.frame(
  moved   = fc_wide$prot_1[1:12],
  flipped = c(fc_wide$prot_2[1:6], -fc_wide$prot_2[7:12]),
  zeroed  = c(rep(0, 6), fc_wide$prot_2[7:12]),
  group   = rep(c("ctrl", "case"), each = 6),
  stringsAsFactors = FALSE
)
edge_res <- suppressMessages(suppressWarnings(STATassist::compare_two_groups(
  edge_wide, c("moved", "flipped", "zeroed"), edge_wide$group,
  c("ctrl", "case"), diagnose = FALSE
)))

write_case(
  "estimate_undecided",
  edge_wide,
  list(
    plain = est_case(edge_res, log2fc_cutoff = 0.05),
    # A cutoff no finite ratio here can clear, which leaves the infinite one
    # clearing it and the undecided one still undecided.
    strict = est_case(edge_res, log2fc_cutoff = 8)
  )
)

one_res <- suppressMessages(suppressWarnings(STATassist::compare_one_sample(
  one_df, one_feats, mu = 5, p = 0.5
)))

write_case(
  "estimate_one_sample",
  one_df,
  list(
    # No group levels to carry, so `group_lv` is a stated absence.
    plain = est_case(one_res, log2fc_cutoff = 0.05),
    prop = est_case(one_res, test = "prop_test", log2fc_cutoff = 0.05)
  )
)

multi_res <- suppressMessages(suppressWarnings(
  STATassist::compare_multiple_groups(wide, multi_feats, wide$group, multi_lv,
                                      posthoc_alpha = 1, diagnose = FALSE)
))

write_case(
  "estimate_multi_group",
  wide,
  list(
    # The omnibus reading carries `extreme_level`, naming the level whose centre
    # produced `log2fc`.
    omnibus = est_case(multi_res, log2fc_cutoff = 0.05),
    kruskal = est_case(multi_res, test = "kruskal_test",
                       log2fc_cutoff = 0.05),
    # One verdict table per contrast, each with its own log2fc and its own
    # adjustment axis.
    by_contrast = est_case(multi_res, by = "contrast", log2fc_cutoff = 0.05),
    # Naming a method here adjusts across the features of one contrast, which is
    # a different family from the one the pairwise stage corrected over.
    by_contrast_bh = est_case(multi_res, by = "contrast", adj_type = "BH",
                              log2fc_cutoff = 0.05)
  )
)

# A strict post-hoc threshold leaves features that were never compared
# pairwise, whose p-value is absent in every contrast table while their ratio
# is not.
multi_strict <- suppressMessages(suppressWarnings(
  STATassist::compare_multiple_groups(wide, multi_feats, wide$group, multi_lv,
                                      posthoc_alpha = 0.001, diagnose = FALSE)
))

write_case(
  "estimate_not_asked",
  wide,
  list(
    by_contrast = est_case(multi_strict, by = "contrast",
                           log2fc_cutoff = 0.05)
  )
)


# --------------------------------------------------------------------------- #
# utils_factorial.R
#
# Every index here is one-based on the way out and the port converts. Which cell
# is which is what is being graded, not how it is counted.
# --------------------------------------------------------------------------- #

write_case(
  "fact_layout",
  fact_df,
  list(
    cells = as_cols(fact_cells),
    labels = fact_label,
    cell_index = STATassist:::sa_fact_cell_index(
      as.matrix(fact_cells), c(3L, 2L)
    ),
    terms = STATassist:::sa_fact_terms(names(fact_lv)),
    term_labels = STATassist:::sa_fact_term_labels(
      STATassist:::sa_fact_terms(names(fact_lv))
    ),
    subsets = STATassist:::sa_fact_subsets(c("treatment", "sex")),
    cube_cells = as_cols(cube_cells),
    cube_labels = STATassist:::sa_fact_cell_labels(cube_lv, cube_cells),
    cube_terms = STATassist:::sa_fact_terms(names(cube_lv)),
    cube_term_labels = STATassist:::sa_fact_term_labels(
      STATassist:::sa_fact_terms(names(cube_lv))
    ),
    cube_cell_index = STATassist:::sa_fact_cell_index(
      as.matrix(cube_cells), c(2L, 2L, 2L)
    ),
    control_first = STATassist:::sa_fact_control_first(
      fact_lv, list(sex = "female")
    ),
    # `expand.grid()` of nothing is one row and no columns, which is the answer
    # every index built on the grid has to survive. `sa_fact_cell_labels()` is
    # not graded there: R's `apply()` over a zero-column matrix raises, and the
    # port answers `""` instead, which is what `paste(character(0))` means.
    empty_grid_rows = nrow(STATassist:::sa_fact_grid(list())),
    empty_grid_cols = ncol(STATassist:::sa_fact_grid(list())),
    empty_cell_index = STATassist:::sa_fact_cell_index(
      matrix(integer(0), nrow = 3L, ncol = 0L), integer(0)
    ),
    tol = STATassist:::sa_fact_tol()
  )
)

# A shift per cell with a main effect, a partner effect and a real interaction in
# it, so every term of the decomposition is non-zero and none of them is the
# whole of it.
fact_eff <- c(0, 1.4, 2.1, 0.5, 0.4, 2.9)
fact_terms_list <- STATassist:::sa_fact_terms(names(fact_lv))
write_case(
  "fact_decompose",
  data.frame(cell = fact_label, eff = fact_eff, stringsAsFactors = FALSE),
  list(
    collapse_none = STATassist:::sa_fact_collapse(fact_eff, fact_cells,
                                                  character(0)),
    collapse_treatment = STATassist:::sa_fact_collapse(fact_eff, fact_cells,
                                                       "treatment"),
    collapse_sex = STATassist:::sa_fact_collapse(fact_eff, fact_cells, "sex"),
    collapse_both = STATassist:::sa_fact_collapse(fact_eff, fact_cells,
                                                  c("treatment", "sex")),
    component = lapply(fact_terms_list, function(t) {
      STATassist:::sa_fact_component(fact_eff, fact_cells, t)
    }),
    term_effect = STATassist:::sa_fact_term_effect(fact_eff, fact_cells,
                                                   fact_terms_list),
    # A flat shift belongs to the grand mean, so every component of it is zero.
    flat_term_effect = STATassist:::sa_fact_term_effect(
      rep(2.5, 6), fact_cells, fact_terms_list
    )
  )
)

fact_skel <- STATassist:::sa_fact_contrast_skeleton(
  list(factor_lv = fact_lv, cells = fact_cells)
)
write_case(
  "fact_contrast_skeleton",
  fact_df,
  list(
    table = fact_skel$table,
    sel1 = fact_skel$sel1,
    sel2 = fact_skel$sel2
  )
)


# --------------------------------------------------------------------------- #
# kernel_factorial.R
# --------------------------------------------------------------------------- #

fact_mat <- STATassist:::sa_fact_cell_matrix(fact_lv, fact_cells)
cube_mat <- STATassist:::sa_fact_cell_matrix(cube_lv, cube_cells)
write_case(
  "factorial_cell_matrix",
  fact_df,
  list(
    x = fact_mat$x,
    assign = fact_mat$assign,
    cube_x = cube_mat$x,
    cube_assign = cube_mat$assign,
    contr_sum_2 = unname(stats::contr.sum(2)),
    contr_sum_4 = unname(stats::contr.sum(4))
  )
)

ss_plan_of <- function(mat, ss_type) {
  plan <- STATassist:::sa_fact_ss_plan(mat$terms, mat$assign, ss_type)
  list(base = lapply(plan, `[[`, "base"), full = lapply(plan, `[[`, "full"))
}
write_case(
  "factorial_ss_plan",
  fact_df,
  list(
    III = ss_plan_of(fact_mat, "III"),
    II = ss_plan_of(fact_mat, "II"),
    I = ss_plan_of(fact_mat, "I"),
    cube_III = ss_plan_of(cube_mat, "III"),
    cube_II = ss_plan_of(cube_mat, "II"),
    cube_I = ss_plan_of(cube_mat, "I")
  )
)

fact_fit_of <- function(samples, cells, ss_type) {
  plan <- STATassist:::sa_factorial_plan(fact_lv, cells, ss_type)
  fit <- STATassist:::sa_factorial_anova(samples, plan)
  list(
    model = nv(fit$model),
    terms = as_cols(fit$terms),
    labels = rownames(fit$terms),
    # Positional on the way out: the port's cell means are an array in cell
    # order, and the cell they belong to is `labels`, not a name on the value.
    means = unname(fit$means),
    n = unname(fit$n),
    ms_error = fit$ms_error,
    df_error = fit$df_error
  )
}
write_case(
  "factorial_anova",
  fact_df,
  list(
    III = fact_fit_of(fact_samples, fact_cells, "III"),
    II = fact_fit_of(fact_samples, fact_cells, "II"),
    I = fact_fit_of(fact_samples, fact_cells, "I"),
    # Type I on unbalanced data is what aov() reports, so the whole kernel has an
    # external check and not only an internal one.
    aov_I = as_cols(as.matrix(
      summary(stats::aov(value ~ treatment * sex, data = fact_df))[[1]]
    )),
    # One cell never observed: the interaction block loses a column of rank and
    # `df` has to come off the two ranks.
    holed = fact_fit_of(holed_samples, holed_cells, "III"),
    holed_labels = holed_label
  )
)

fact_plan_III <- STATassist:::sa_factorial_plan(fact_lv, fact_cells, "III")
fact_fit_III <- STATassist:::sa_factorial_anova(fact_samples, fact_plan_III)
fact_nmeans <- ifelse(fact_skel$table$factor == "treatment", 3L, 2L)
write_case(
  "factorial_tukey",
  fact_df,
  list(
    all = as_cols(STATassist:::sa_factorial_tukey(
      fact_fit_III, fact_skel, fact_nmeans, seq_len(nrow(fact_skel$table))
    )),
    conf_90 = as_cols(STATassist:::sa_factorial_tukey(
      fact_fit_III, fact_skel, fact_nmeans, seq_len(nrow(fact_skel$table)),
      conf_level = 0.90
    )),
    # A subset in a different order, since `rows` fixes the output order.
    picked = as_cols(STATassist:::sa_factorial_tukey(
      fact_fit_III, fact_skel, fact_nmeans, c(10L, 1L, 5L)
    )),
    nmeans = fact_nmeans
  )
)


# --------------------------------------------------------------------------- #
# kernel_categorical.R and the three utils_categorical.R helpers the kernels
# rest on.
# --------------------------------------------------------------------------- #

write_case(
  "cat_expected",
  cat_df,
  list(
    independence_2x2 = STATassist:::sa_expected_independence(tab_2x2),
    independence_3x4 = STATassist:::sa_expected_independence(tab_3x4),
    symmetry_small = STATassist:::sa_expected_symmetry(pair_small),
    symmetry_one_way = STATassist:::sa_expected_symmetry(pair_one_way),
    finite_or_na = STATassist:::sa_finite_or_na(
      c(1.5, Inf, -Inf, NaN, NA_real_, 0)
    )
  )
)

write_case(
  "cat_chisq",
  cat_df,
  nvs(
    t2x2_corrected = STATassist:::sa_chisq(tab_2x2),
    t2x2_plain = STATassist:::sa_chisq(tab_2x2, correct = FALSE),
    # Yates is a 2 x 2 rule in R, whatever `correct` is set to, so these two agree
    # and a port that keys the correction on one degree of freedom disagrees.
    t2x4_corrected = STATassist:::sa_chisq(tab_2x4),
    t2x4_plain = STATassist:::sa_chisq(tab_2x4, correct = FALSE),
    t3x4 = STATassist:::sa_chisq(tab_3x4),
    t3x3_small = STATassist:::sa_chisq(tab_3x3_small)
  )
)

write_case(
  "cat_fisher",
  cat_df,
  nvs(
    t2x2 = STATassist:::sa_fisher(tab_2x2),
    t2x2_conf_90 = STATassist:::sa_fisher(tab_2x2, conf_level = 0.90),
    t2x2_zero = STATassist:::sa_fisher(tab_2x2_zero),
    t3x3_small = STATassist:::sa_fisher(tab_3x3_small),
    t3x3_mid = STATassist:::sa_fisher(tab_3x3_mid),
    t2x4 = STATassist:::sa_fisher(tab_2x4),
    # `t3x4` is inside R's workspace and outside the Python port's own limit, so
    # it is exported to state R's answer and the port asserts `enumerated = 0`
    # against it rather than the p-value. Where the two limits differ is written
    # up at `FISHER_TABLE_LIMIT`.
    t3x4 = STATassist:::sa_fisher(tab_3x4),
    # Past the limit in both, which is not an error in the data: `enumerated` is
    # 0 and the p-value is absent.
    t_big = STATassist:::sa_fisher(tab_big)
  )
)

write_case(
  "cat_mcnemar",
  cat_df,
  nvs(
    small_default = STATassist:::sa_mcnemar(pair_small),
    small_forced_chisq = STATassist:::sa_mcnemar(pair_small, exact = FALSE),
    small_plain_chisq = STATassist:::sa_mcnemar(pair_small, correct = FALSE,
                                                exact = FALSE),
    large_default = STATassist:::sa_mcnemar(pair_large),
    large_forced_exact = STATassist:::sa_mcnemar(pair_large, exact = TRUE),
    large_plain_chisq = STATassist:::sa_mcnemar(pair_large, correct = FALSE),
    one_way = STATassist:::sa_mcnemar(pair_one_way)
  )
)

cochran_fit <- STATassist:::sa_cochran_q(cochran_mat)
write_case(
  "cat_cochran_q",
  as_cols(cochran_mat),
  list(
    q = nv(cochran_fit),
    kendalls_w = STATassist:::sa_assoc_measures_repeated(
      cochran_fit[["statistic"]], nrow(cochran_mat), ncol(cochran_mat)
    )
  )
)

write_case(
  "cat_association",
  cat_df,
  list(
    t2x2 = STATassist:::sa_assoc_measures(tab_2x2),
    t2x2_conf_90 = STATassist:::sa_assoc_measures(tab_2x2, conf_level = 0.90),
    t2x2_zero = STATassist:::sa_assoc_measures(tab_2x2_zero),
    t3x4 = STATassist:::sa_assoc_measures(tab_3x4),
    paired_small = STATassist:::sa_assoc_measures_paired(pair_small),
    paired_conf_90 = STATassist:::sa_assoc_measures_paired(pair_small,
                                                            conf_level = 0.90),
    paired_one_way = STATassist:::sa_assoc_measures_paired(pair_one_way),
    phi_2x2 = STATassist:::sa_phi(tab_2x2),
    phi_zero = STATassist:::sa_phi(tab_2x2_zero),
    odds_ratio = STATassist:::sa_odds_ratio(tab_2x2),
    odds_ratio_zero = STATassist:::sa_odds_ratio(tab_2x2_zero),
    has_zero_2x2 = STATassist:::sa_has_zero_cell(tab_2x2),
    has_zero_zero = STATassist:::sa_has_zero_cell(tab_2x2_zero),
    has_zero_3x4 = STATassist:::sa_has_zero_cell(tab_3x4)
  )
)


# --------------------------------------------------------------------------- #
# The rest of utils_categorical.R, and the contract constants of categorical.R.
#
# Given inputs of their own rather than added to `cat_df`, so the cases already
# frozen above are exported from the very same file they were.
# --------------------------------------------------------------------------- #

# An empty row and an empty column, so every quantity that divides by a margin
# has one to divide by nothing: `prop_row`, `prop_col` and both residuals.
tab_hole <- mkt(c(7, 0, 4, 0, 0, 0, 5, 0, 9),
                c("low", "mid", "high"), c("x", "y", "z"), 3L)

cells_df <- tab_long(list(
  t2x2 = tab_2x2, t3x4 = tab_3x4, pair_small = pair_small, hole = tab_hole
))

write_case(
  "cat_cells",
  cells_df,
  list(
    columns = STATassist:::sa_categorical_cell_columns(),
    nulls = STATassist:::sa_categorical_nulls(),
    test_columns = STATassist:::sa_categorical_test_columns(),
    assoc_columns = STATassist:::sa_association_columns(),
    t2x2_independence = STATassist:::sa_categorical_cells(tab_2x2),
    # Square, so all three nulls can be stated of it. `marginal_homogeneity`
    # takes the independence arithmetic and keeps `std_residual` a number, which
    # a port that branches on "not independence" gets wrong.
    t2x2_symmetry = STATassist:::sa_categorical_cells(tab_2x2, "symmetry"),
    t2x2_marginal = STATassist:::sa_categorical_cells(tab_2x2,
                                                     "marginal_homogeneity"),
    t3x4_independence = STATassist:::sa_categorical_cells(tab_3x4),
    pair_small_symmetry = STATassist:::sa_categorical_cells(pair_small,
                                                           "symmetry"),
    hole_independence = STATassist:::sa_categorical_cells(tab_hole),
    hole_symmetry = STATassist:::sa_categorical_cells(tab_hole, "symmetry")
  )
)

# Mixed storage modes, because `as.character()` is what makes a factor, a
# logical and a 0/1 code all categorical here. `answer` carries both kinds of
# unusable row: one missing value and one level a named `category_lv` leaves out.
cat_input_df <- data.frame(
  answer = c("y", "n", "y", "y", NA, "n", "y", "n", "y", "maybe"),
  grade = factor(c("low", "high", "mid", "low", "high",
                   "mid", "low", "high", "mid", "low"),
                 levels = c("low", "mid", "high")),
  flag = c(TRUE, FALSE, TRUE, TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, TRUE),
  coded = c(1L, 0L, 1L, 0L, 1L, 1L, 0L, 0L, 1L, 0L),
  stringsAsFactors = FALSE
)

# Repeated conditions where one condition never saw a level the other did, which
# is what the union path exists for: without it the table is not square.
cat_paired_df <- data.frame(
  before = c("no", "no", "no", "yes", "no", "no", "yes", "no"),
  after = c("yes", "yes", "no", "yes", "maybe", "yes", "yes", "no"),
  stringsAsFactors = FALSE
)

cat_val_plain <- STATassist:::sa_validate_categorical_input(
  cat_input_df, NULL, NULL, FALSE
)
cat_val_named <- STATassist:::sa_validate_categorical_input(
  cat_input_df, list(answer = c("y", "n"), grade = c("low", "mid")),
  list(answer = "n"), FALSE
)
cat_val_paired <- STATassist:::sa_validate_categorical_input(
  cat_paired_df, NULL, "no", TRUE
)

# The resolved levels are the whole content of the answer, so they are exported
# as the labels rather than as the factor codes underneath them.
val_out <- function(res) {
  list(
    variables = res$variables,
    category_lv = res$category_lv,
    n_used = res$n_used,
    n_dropped = res$n_dropped,
    n_incomplete = res$n_incomplete,
    data = as_cols(lapply(res$data, as.character))
  )
}

write_case(
  "cat_validate",
  cat_input_df,
  list(
    plain = val_out(cat_val_plain),
    # `category_lv` naming three levels of a variable that takes more is the way
    # through `max_levels`, and it drops the rows at the rest.
    named = val_out(cat_val_named),
    paired = val_out(cat_val_paired),
    shared_lv_union = STATassist:::sa_categorical_shared_lv(
      list(before = c("no", "yes"), after = c("maybe", "no", "yes")),
      list(before = c("no", "yes"), after = c("maybe", "no", "yes")),
      FALSE, NULL
    ),
    shared_lv_control = STATassist:::sa_categorical_shared_lv(
      list(before = c("no", "yes"), after = c("maybe", "no", "yes")),
      list(before = c("no", "yes"), after = c("maybe", "no", "yes")),
      FALSE, "yes"
    )
  )
)

cat_counts_plain <- STATassist:::sa_categorical_counts(
  cat_val_plain$data, c("answer", "grade")
)
cat_counts_named <- STATassist:::sa_categorical_counts(
  cat_val_named$data, cat_val_named$variables
)
cat_counts_paired <- STATassist:::sa_categorical_counts(
  cat_val_paired$data, cat_val_paired$variables
)
cat_counts_condition <- STATassist:::sa_categorical_condition_counts(
  cat_val_paired$data, cat_val_paired$variables,
  cat_val_paired$category_lv[[1]]
)

# The labels are the key `cells` and a simulator's `truth_cell` are both read on,
# so they are exported beside the counts rather than left to the row order.
tab_out <- function(m) {
  list(
    counts = as_cols(unclass(m)),
    row_levels = rownames(m),
    col_levels = colnames(m),
    dim_names = names(dimnames(m))
  )
}

write_case(
  "cat_counts",
  cat_input_df,
  list(
    plain = tab_out(cat_counts_plain),
    named = tab_out(cat_counts_named),
    paired = tab_out(cat_counts_paired),
    condition = tab_out(cat_counts_condition),
    paired_cells = STATassist:::sa_categorical_cells(cat_counts_paired,
                                                     "symmetry"),
    condition_cells = STATassist:::sa_categorical_cells(
      cat_counts_condition, "marginal_homogeneity"
    )
  )
)

write_case(
  "cat_diagnose",
  cells_df,
  list(
    expected_ok = STATassist:::sa_diagnose_expected(
      STATassist:::sa_categorical_cells(tab_3x4)
    ),
    expected_sparse = STATassist:::sa_diagnose_expected(
      STATassist:::sa_categorical_cells(tab_3x3_small)
    ),
    # One cell below 5 out of eight, none below 1: the rule's second clause, the
    # one a port that only checks the minimum never reaches.
    expected_one_small = STATassist:::sa_diagnose_expected(
      STATassist:::sa_categorical_cells(tab_2x4)
    ),
    expected_hole = STATassist:::sa_diagnose_expected(
      STATassist:::sa_categorical_cells(tab_hole)
    ),
    discordance_below = STATassist:::sa_diagnose_discordance(12L),
    discordance_at = STATassist:::sa_diagnose_discordance(25L),
    discordance_above = STATassist:::sa_diagnose_discordance(48L),
    repeated_below = STATassist:::sa_diagnose_repeated(5L, 3L),
    repeated_at = STATassist:::sa_diagnose_repeated(8L, 3L),
    repeated_above = STATassist:::sa_diagnose_repeated(12L, 3L)
  )
)


# --------------------------------------------------------------------------- #
# kernel_performance.R
# --------------------------------------------------------------------------- #

write_case(
  "perf_roc",
  perf_df,
  list(
    points_old = STATassist:::sa_roc_points(perf_response, perf_old),
    points_new = STATassist:::sa_roc_points(perf_response, perf_new),
    placement_old = STATassist:::sa_placement_values(perf_response, perf_old),
    placement_new = STATassist:::sa_placement_values(perf_response, perf_new),
    auc_old = STATassist:::sa_auc(perf_response, perf_old),
    auc_new = STATassist:::sa_auc(perf_response, perf_new),
    delong_old = nv(STATassist:::sa_auc_delong(perf_response, perf_old)),
    delong_new = nv(STATassist:::sa_auc_delong(perf_response, perf_new))
  )
)

write_case(
  "perf_compare",
  perf_df,
  nvs(
    delong_test = STATassist:::sa_delong_test(perf_response, perf_new, perf_old),
    delong_reversed = STATassist:::sa_delong_test(perf_response, perf_old,
                                                  perf_new),
    # Two models that rank every row identically differ by exactly zero with a
    # standard error of exactly zero, and the ratio of the two is not a number.
    delong_identical = STATassist:::sa_delong_test(perf_response, perf_old,
                                                   perf_old),
    idi = STATassist:::sa_idi(perf_response, perf_old, perf_new),
    idi_identical = STATassist:::sa_idi(perf_response, perf_old, perf_old),
    nri = STATassist:::sa_nri(perf_response, perf_old, perf_new),
    nri_identical = STATassist:::sa_nri(perf_response, perf_old, perf_old)
  )
)

perf_thresholds <- c(0, 0.25, 0.4, 0.55, 0.83, 1)
write_case(
  "perf_scores",
  perf_df,
  list(
    brier_old = STATassist:::sa_brier(perf_response, perf_old),
    brier_new = STATassist:::sa_brier(perf_response, perf_new),
    thresholds = perf_thresholds,
    at_threshold = as_cols(t(vapply(perf_thresholds, function(cut) {
      STATassist:::sa_threshold_scores(perf_response, perf_old, cut)
    }, numeric(3))))
  )
)

write_case(
  "perf_thin",
  data.frame(response = thin_response, predictor_old = thin_old,
             predictor_new = thin_new),
  nvs(
    delong = STATassist:::sa_auc_delong(thin_response, thin_old),
    delong_test = STATassist:::sa_delong_test(thin_response, thin_new,
                                              thin_old),
    idi = STATassist:::sa_idi(thin_response, thin_old, thin_new),
    nri = STATassist:::sa_nri(thin_response, thin_old, thin_new)
  )
)


# --------------------------------------------------------------------------- #
# kernel_cluster.R
# --------------------------------------------------------------------------- #

sil_points <- cbind(sil_df$x, sil_df$y)
sil_dist <- stats::dist(sil_points)
write_case(
  "cluster_silhouette",
  sil_df,
  list(
    # Two real clusters, a singleton, two noise points and two coincident points.
    mixed = STATassist:::sa_silhouette(sil_dist, sil_cluster),
    # A single cluster is a comparison with nothing to compare against.
    one_cluster = STATassist:::sa_silhouette(
      sil_dist, ifelse(sil_cluster > 0L, 1L, 0L)
    ),
    # Noise takes no part in any other point's a or b.
    no_noise = STATassist:::sa_silhouette(
      sil_dist, ifelse(sil_cluster == 0L, 1L, sil_cluster)
    ),
    all_noise = STATassist:::sa_silhouette(sil_dist, rep(0L, length(sil_cluster)))
  )
)


cat("\ndone:", length(list.dirs(ROOT, recursive = FALSE)), "case(s) in", ROOT,
    "\n")
