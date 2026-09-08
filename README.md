# STATassist-py

**STATassist-py** runs every method that applies to a question in one call and returns standardised tables. A comparison reports parametric, rank-based and robust tests side by side, with fold changes, intervals and multiplicity-adjusted p-values. A model reports one row per term with its estimate and whatever inference that model honestly supports. A dimension reduction reports one row per point with its coordinates. A feature selection reports one row per candidate with what it was ranked by and whether it survived. A contingency table reports one row per cell under the null the test was read against. A performance evaluation reports one row per model on the same held-out rows. A clustering reports one row per point with the label it was assigned. Seven result contracts, and everything downstream reads them rather than the engine underneath.

Two groups, three or more groups and a single sample all return the same object, so `draw_forest_plot()`, `estimate_significance()` and anything else that reads a result works across them without being told which scenario produced it. Five models — linear, logistic, penalized, forest and kernel — return the same object too, so `model.coef()` and `model.predict(newdata=)` are one line each whichever of them was fitted, and `perform_rfe()` and `perform_stepwise()` hand the predictors they kept straight to any of the five.

Every example below runs on simulated data whose answer was planted on purpose, so a verdict can be **scored** rather than trusted: `simulate_two_groups()`, `simulate_multiple_groups()`, `simulate_factorial_groups()`, `simulate_categorical_groups()`, `simulate_regression()` and `simulate_classification()` hand back the effects and coefficients they put in.

This is a port of [R STATassist](https://github.com/hiows/STATassist). The comparison, diagnostic and simulation functions are written against `numpy`, `pandas` and `scipy`. The modelling, selection, reduction and clustering functions wrap `scikit-learn`, which is where the port stands in place of R's `glmnet`, `randomForest`, `kernlab`, `Rtsne` and `dbscan`. The `draw_*` functions render with `matplotlib`. Only `perform_umap()` needs anything else, and it is an optional extra.

## Example

The volcano plot below is what **§1–§2** produce on 30 simulated genes, eight of them planted up and eight planted down in `case`. Since the answer is known, **§3** scores the plot against it: 13 of the 16 planted genes are called, and none of the 14 null ones.

```python
import statassist as sa

sim = sa.simulate_two_groups(n_feats=30, n_up=8, n_down=8, seed=2026)
comp_res = sa.compare_two_groups(**sim.args)
sig = sa.estimate_significance(comp_res, test="t_test")
sa.draw_volcano_plot(sig)
```

![Volcano plot from compare_two_groups and estimate_significance](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-volcano.png)

`draw_forest_plot()` on the same result draws either the estimates against their intervals or the p-values against the threshold, from the same table:

| `type="estimate"` | `type="pvalue"` |
| --- | --- |
| ![Forest plot of differences in log2 means](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-forest-estimate.png) | ![Forest plot of adjusted p-values](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-forest-pvalue.png) |

And the same wide input feeds the plots that look at the data instead of at a result:

| Grouped boxplot | Back-to-back histogram |
| --- | --- |
| ![Grouped boxplot of the first ten genes](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-boxplot.png) | ![Butterfly histogram of gene_29](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-butterfly.png) |

---

## Installation

The package is not on PyPI yet, so install it from GitHub:

```bash
pip install git+https://github.com/hiows/STATassist-py.git
```

`perform_umap()` is the one function whose engine is not a hard dependency. It pulls in `numba`, it is the only public function that needs it, and the other two reductions answer about the same points — so a caller who wants UMAP asks for it and everyone else does not pay for the compiler:

```bash
pip install "statassist[umap] @ git+https://github.com/hiows/STATassist-py.git"
```

Python 3.11 or newer. The core dependencies are `numpy`, `pandas`, `scipy`, `scikit-learn` and `matplotlib`.

## Conventions used throughout

Every example imports the package as `sa`:

```python
import statassist as sa
```

Every result is a `Mapping` whose slots are also attributes, so `comp_res.effect` and `comp_res["effect"]` are the same table. This README uses attribute access for slots, and subscripts only for the plain dictionaries inside them:

```python
comp_res.effect               # a DataFrame
comp_res.tests["t_test"]      # `tests` is a dict of DataFrames
"posthoc" in comp_res         # slots that only some scenarios carry
```

Wide input is one row per observation, numeric columns are features. Direction is fixed by `group_lv`, whose first level is the reference: differences read `group_lv[1] - group_lv[0]` and fold changes `group_lv[1] / group_lv[0]`. The same rule holds for three or more groups, so a control named first stays the reference whichever function reads it.

---

# Part 1 — Comparison

### 1. Compare two groups (all applicable tests)

`simulate_two_groups()` returns data in exactly that shape along with the effects it planted. Thirty features, eight moved up and eight moved down in `case`, the other fourteen null. Its `args` slot is named after the parameters of `compare_two_groups()`, so it can be spread out as below, or handed over whole with `compare_two_groups(**sim.args)`.

```python
sim = sa.simulate_two_groups(n_feats=30, n_up=8, n_down=8, seed=2026)

comp_res = sa.compare_two_groups(
    data=sim.args["data"],
    feats=sim.args["feats"],
    group=sim.args["group"],
    group_lv=sim.args["group_lv"],
    input_scale=sim.args["input_scale"],
)

comp_res
```

```
<SaTwoGroup> two_group_comparison
  groups   : control vs case  (independent)
  features : 30
  settings : alternative = two.sided, conf_level = 0.95, p_adjust = BH

  tests
    $t_test       13 of 30 at pval_adj <= 0.05
                 Welch's t-test
    $wilcox_test  13 of 30 at pval_adj <= 0.05
                 Wilcoxon rank sum test (Mann-Whitney U test)
    $robust_test  13 of 30 at pval_adj <= 0.05
                 Brunner-Munzel test

  $diagnostics attached
```

`group_lv` is `["control", "case"]`, so `control` is the reference and a positive `log2fc` means higher in `case`, which is where the effects were planted.

```python
comp_res.effect.head(4)
```

```
  features    x_center    y_center  fold_change    log2fc
0   gene_1   45.101254   15.072227     2.992342  1.581275
1   gene_2  361.708567  340.676746     1.061735  0.086424
2   gene_3  123.662174   85.328723     1.449244  0.535301
3   gene_4   67.616304   48.285753     1.400337  0.485774
```

The centres are in the tens and hundreds while the features themselves run from about 2 to 12, because the data is on the log2 scale, as gene expression usually is, which is what `input_scale="log2"` says. Dividing two means of logged values is not a fold change and can even come out with the wrong sign: log2 centres of -1 and -2 are a two-fold increase, but their ratio reads as a two-fold decrease. Each observation is raised back through `2**x` before the centres are taken, and `fc_mean` then defaults to `"geom"`, which makes `log2fc` the difference of the two log2 means.

Only `comp_res.effect` is converted. The tests still run on the log2 values, which is the reason for logging them in the first place.

The three tests live under `comp_res.tests`, keyed by name, and each is a table with one row per feature:

```python
comp_res.tests["t_test"]        # Welch or paired t, depending on paired=
comp_res.tests["wilcox_test"]
comp_res.tests["robust_test"]
```

Paired data needs `id` and `paired=True`; the subject column matches the rows up and subjects missing either condition are dropped whole.

### 2. Significance and volcano plot

`estimate_significance()` takes the comparison object and applies cutoffs to `log2fc` and p-values. `adj_type=None`, the default, uses the adjusted p-values already stored in the result and so avoids double adjustment; naming a method re-adjusts from `pval`.

```python
sig = sa.estimate_significance(
    comp_res,
    test="t_test",
    log2fc_cutoff=1,
    pval_cutoff=0.05,
    adj_type="BH",
)
sig

verdict = sig.significance     # one row per feature
sa.draw_volcano_plot(sig)
```

```
<SaSignificance> two_group_comparison
  test     : t_test  (Welch's t-test)
  cutoffs  : abs(log2fc) >= 1, adj_pvalue <= 0.05  (BH)
  verdict  : 13 of 30 significant
```

The verdict comes back as `sig.significance`, a DataFrame of `features`, `log2fc`, `pvalue`, `adj_pvalue` and `is_signif`, beside the `sig.analysis_type` it was read from. The scenario name travels with the table because `log2fc` does not mean the same thing in all of them: with two groups it is the second level over the reference, with three or more it is the level furthest from the reference, which is why a multi-group volcano plot says so on its x axis.

Pass `test="wilcox_test"` or `test="robust_test"` to threshold on a different family; `log2fc` stays the same because it comes from `comp_res.effect`.

### 3. Score the verdict against the planted answer

The comparison above ran on data whose answer is known, so the verdict can be scored rather than trusted. Unplanted features have a true fold change of exactly zero, which makes anything called among them a false positive by definition.

```python
import pandas as pd

planted = sim.truth["direction"].to_numpy() != "none"
called = verdict["is_signif"].fillna(False).to_numpy()

pd.crosstab(
    pd.Series(planted, name="planted"),
    pd.Series(called, name="called"),
)
```

```
called   False  True 
planted              
False       14      0
True         3     13
```

Thirteen of the sixteen planted features come back and none of the fourteen null ones is called. The three that were missed are worth looking up rather than guessing at, which is what the rest of `truth` is for:

```python
missed = planted & ~called
sim.truth[missed]
verdict[missed]
```

```
   features direction    log2fc  baseline   sd_case  sd_control
5    gene_6        up  1.192517  9.905182  3.125548    2.387683
18  gene_19        up  1.049078  4.778992  2.823737    1.228359
29  gene_30        up  1.347547  4.999915  2.911643    2.268551

   features    log2fc    pvalue  adj_pvalue  is_signif
5    gene_6  0.927544  0.073321    0.157117      False
18  gene_19  0.684492  0.195189    0.365980      False
29  gene_30  0.894895  0.092158    0.184317      False
```

All three were planted between 1.05 and 1.35, barely over the cutoff, and all three were estimated under 0.93. Nothing went wrong: an estimate carries a sampling error of its own, so a feature planted near the cutoff lands below it a good share of the time, and the same noise keeps its p-value from clearing 0.05 either. That is the third reason a real volcano plot loses features, next to the p-value cutoff and the multiplicity adjustment, and it is the one a simulation that recovers everything would hide.

Their spreads say the rest of it. `gene_6` was planted at 1.19 with a `sd_case` of 3.13 against a `sd_control` of 2.39: a group whose spread was widened along with its centre is harder to distinguish, not easier, and `truth` records both so the two can be read together.

The same scoring runs over each test family in turn, which is the point of reporting three of them:

```python
{
    name: float(
        sa.estimate_significance(comp_res, test=name)
        .significance["is_signif"]
        .fillna(False)
        .to_numpy()[planted]
        .mean()
    )
    for name in comp_res.tests
}
```

```
{'t_test': 0.8125, 'wilcox_test': 0.8125, 'robust_test': 0.8125}
```

Here all three recover the same thirteen. That is what a clean two-group separation looks like and not something to count on — the families come apart as soon as a group is skewed or its variance is unequal, which is what §11 checks for.

### 4. Grouped boxplot and back-to-back histogram

Both read the same wide input the comparison took, and both draw the levels in the order `group_lv` gives them, so the reference lands on the left.

```python
first_ten = sim.args["feats"][:10]

sa.draw_grouped_boxplot(
    data=sim.args["data"],
    feats=first_ten,
    group=sim.args["group"],
    group_lv=sim.args["group_lv"],
)
```

![Grouped boxplot of the first ten genes](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-boxplot.png)

```python
strongest = verdict.sort_values("adj_pvalue")["features"].iloc[0]

sa.draw_butterfly_hist(
    data=sim.args["data"],
    feat=strongest,
    group=sim.args["group"],
    group_lv=sim.args["group_lv"],
    type="both",     # or "freq" for bars only, "dens" for the curve only
)
```

![Butterfly histogram of gene_29, bars and density](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-butterfly.png)

The call returns bin summaries and per-group histogram data for further plotting, plus per-group densities when a density is drawn. A density and a bar can only be read against one axis when the bar is a density too: a count or a proportion per bin scales with the bin width, which the curve knows nothing about. So `type="dens"` and `type="both"` move `scale` to `"density"`, and reject a `scale` that was asked for explicitly and says otherwise rather than drawing two incomparable shapes.

### 5. `draw_forest_plot()`: one function for every scenario

`draw_forest_plot()` reads only the columns the result contract guarantees, which is why one function covers two groups, three or more, and a single sample. `type="auto"`, the default, picks the first view the chosen table can support.

```python
sa.draw_forest_plot(comp_res)                                    # estimates with intervals
sa.draw_forest_plot(comp_res, test="wilcox_test", sort_by="pvalue")
sa.draw_forest_plot(comp_res, dark=True)
```

`feats` picks the features to draw and the order to draw them in, from the top of the plot down, and `sort_by` reorders whatever `feats` selected. `xlim` fixes the axis instead of deriving it, so two plots can be read against each other.

```python
sa.draw_forest_plot(
    comp_res, test="t_test", type="estimate",
    feats=first_ten, sort_by="pvalue",
)
```

![Forest plot of mean differences for the first ten genes](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-forest-estimate.png)

The p-value view is the fallback for a table with no interval to draw, and marks the `alpha` threshold. It is also worth asking for on purpose, since it puts the whole selection on one scale:

```python
sa.draw_forest_plot(
    comp_res, test="t_test", type="pvalue",
    feats=first_ten, sort_by="pvalue",
)
```

![Forest plot of adjusted p-values for the first ten genes](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-forest-pvalue.png)

`use_adjusted=False` reads the `pval` column instead of `pval_adj`. The colouring, the sorting, the p-value view and the legend all follow it, so the plot always names the p-value it actually used. The call returns the table it drew, so what the picture claims can be checked against numbers.

### 6. Clustered heatmap

The same wide input, transposed internally so that features run down the rows. `group` labels the samples, which become the columns, and the strip above them is drawn from it.

```python
drawn = sa.draw_heatmap(
    data=sim.args["data"],
    group=sim.args["group"],
    group_lv=sim.args["group_lv"],
    scale="feature",               # or "sample" / "none"
    hclust_method="ward.D2",
    show_sample_names=False,       # 100 samples, no room for 100 labels
)
```

![Clustered heatmap of the simulated two-group data](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-heatmap.png)

Features are z-scored across the samples by default. One colour scale is shared by every cell and features are not measured on a common scale, so without it a single high-abundance feature takes the whole range and the rest of the plot is left white.

The clustering comes back on the result rather than staying inside the picture, so what the plot claims can be checked:

```python
sorted(drawn)
```

```
['feat_hclust', 'feat_order', 'group_colors', 'matrix', 'sample_hclust', 'sample_order', 'zlim']
```

Nothing in the plot was told where the groups are. Whether it put them together anyway is a question about `sample_order`, and the answer is the longest unbroken run of each group along it:

```python
ordered = pd.Series(pd.Series(sim.args["group"]).to_numpy()[drawn["sample_order"]])
runs = ordered.groupby((ordered != ordered.shift()).cumsum())
pd.DataFrame(
    {"group": [g.iloc[0] for _, g in runs], "run": [len(g) for _, g in runs]}
).groupby("group")["run"].max()
```

```
group
case       29
control    30
Name: run, dtype: int64
```

Thirty of the fifty controls land in one block and twenty-nine of the fifty cases in another, out of a clustering that was never told the groups exist. It is not two clean blocks, and it should not be: fourteen of these genes were planted with no effect at all, and the planted half carries enough noise that **§3** misses three of them.

`drawn["matrix"]` is the scaled data in the order it was drawn, and `feat_hclust` and `sample_hclust` are the linkages behind the two dendrograms. Missing values are drawn as grey cells rather than dropped, and a feature with no variance is centred instead of divided by zero.

### 7. Compare three or more groups, with the matching post-hoc stage

Four omnibus tests run side by side, each paired with the pairwise procedure that shares its assumptions: ANOVA with Tukey HSD, Welch's ANOVA with Games-Howell, Yuen's trimmed-mean ANOVA with pairwise Yuen, Kruskal-Wallis with Dunn's test. Pairing them in the result is what makes it impossible to follow a rank-based omnibus test with a parametric comparison by accident.

`simulate_multiple_groups()` builds one control group and any number of treatment groups, and `n_treat` states how many by its length.

```python
sim_multi = sa.simulate_multiple_groups(
    n_feats=10, n_control=50, n_treat=(50, 50, 50),
    n_up=3, n_down=3, seed=2026,
)

multi = sa.compare_multiple_groups(**sim_multi.args)
multi
```

```
<SaMultiGroup> multi_group_comparison
  groups   : control vs treat_1 vs treat_2 vs treat_3  (independent)
  features : 10
  settings : alternative = two.sided, conf_level = 0.95, p_adjust = BH

  tests
    $anova_test    6 of 10 at pval_adj <= 0.05
                  One-way ANOVA
                  post-hoc: 14 of 36 contrast(s) over 6 feature(s), Tukey HSD
    $welch_test    7 of 10 at pval_adj <= 0.05
                  Welch's one-way ANOVA
                  post-hoc: 15 of 42 contrast(s) over 7 feature(s), Games-Howell post-hoc test
    $robust_test   6 of 10 at pval_adj <= 0.05
                  Yuen's trimmed mean one-way ANOVA
                  post-hoc: 14 of 36 contrast(s) over 6 feature(s), Pairwise Yuen tests
    $kruskal_test  5 of 10 at pval_adj <= 0.05
                  Kruskal-Wallis test
                  post-hoc: 14 of 30 contrast(s) over 5 feature(s), Dunn's post-hoc test

  $diagnostics attached
```

The four families disagree here in a way the two-group example did not: Welch's finds seven, Kruskal-Wallis five. Unequal group variances are exactly what `simulate_multiple_groups()` plants along with the centres, and they are what Welch's treatment of the same data is for.

```python
multi.tests["anova_test"][["features", "n_used", "f_stat", "eta_sq", "pval_adj"]]
```

```
  features  n_used     f_stat    eta_sq      pval_adj
0   prot_1   200.0  10.105083  0.133951  1.602715e-05
1   prot_2   200.0   7.878447  0.107612  1.819613e-04
2   prot_3   200.0  16.344921  0.200113  1.605685e-08
3   prot_4   200.0   3.179460  0.046407  4.188040e-02
4   prot_5   200.0   6.483346  0.090276  8.302507e-04
5   prot_6   200.0   0.085859  0.001312  9.677125e-01
6   prot_7   200.0   1.501181  0.022461  2.677685e-01
7   prot_8   200.0   2.718432  0.039947  6.542276e-02
8   prot_9   200.0   4.773929  0.068095  6.210807e-03
9  prot_10   200.0   1.410364  0.021131  2.677685e-01
```

An omnibus test reports that the levels are not all alike, not by how much, so its `lower_conf` and `upper_conf` are missing throughout and the intervals live in `posthoc` instead. `estimate` there reads as `group1 - group2`, and the reference is the level being subtracted, so a contrast against it points the same way the fold change does:

```python
lead = multi.tests["anova_test"].sort_values("pval_adj")["features"].iloc[0]
ph = multi.posthoc["anova_test"]
ph.loc[ph["features"] == lead, ["contrast", "estimate", "pval_adj"]]
```

```
             contrast  estimate      pval_adj
12  treat_1 - control  0.412947  8.286613e-01
13  treat_2 - control -2.597809  1.326606e-06
14  treat_3 - control  0.012608  9.999936e-01
15  treat_2 - treat_1 -3.010756  1.739896e-08
16  treat_3 - treat_1 -0.400339  8.414159e-01
17  treat_3 - treat_2  2.610417  1.169681e-06
```

Only `treat_2` moved this feature, which is one of the three shapes `simulate_multiple_groups()` plants: `"all"` moves every treatment group alike, `"gradient"` moves them in a ramp, and `"single"` moves one and leaves the rest at exactly zero. They are recovered at visibly different rates by the same omnibus test, which is the point of planting more than one.

`draw_forest_plot()` reaches the same rows with `type="posthoc"`, which is what `type="auto"` falls through to on an omnibus table:

```python
sa.draw_forest_plot(multi, test="anova_test", type="posthoc", feats=lead, sort_by="pvalue")
```

![Tukey HSD contrasts for prot_3](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-multi-posthoc.png)

The pairwise stage runs only for features whose omnibus test cleared `posthoc_alpha`. A feature that did not qualify is **absent** from the post-hoc table rather than present with a missing row, because "never asked" and "asked and unanswerable" are different facts.

`multi.pairwise` holds the same numbers one contrast at a time, keyed by test and then by contrast label:

```python
list(multi.pairwise["anova_test"])
```

```
['treat_1 - control', 'treat_2 - control', 'treat_3 - control',
 'treat_2 - treat_1', 'treat_3 - treat_1', 'treat_3 - treat_2']
```

These tables are rectangular where `posthoc` is ragged: each holds every feature, in the order the rest of the object uses, so a feature that did not qualify is present with its inference columns missing. They add `log2fc`, which no post-hoc procedure reports, being the ratio of the two group centres rather than anything a test produced.

The omnibus verdict and its volcano plot work the same way they did with two groups, except that `log2fc` is now the level furthest from the reference, which the x axis says:

```python
sig_multi = sa.estimate_significance(multi, test="anova_test", pval_cutoff=0.05, adj_type="BH")
sa.draw_volcano_plot(sig_multi)
```

![Volcano plot of the multi-group verdict](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-multi-volcano.png)

`estimate_significance(multi, by="contrast")` reads the `pairwise` tables instead, and its `significance` is then one verdict table per contrast.

Repeated conditions need `id` and a complete rectangle; `simulate_multiple_groups(paired=True)` builds one, and subjects missing any condition are dropped whole. Repeated measures swap in Friedman as `tests["kruskal_test"]`, with Conover's pairwise comparisons behind it, and report Mauchly's sphericity test and both epsilon corrections on the same row as the uncorrected F.

### 8. Factorial crossed design

Crossing two factors asks three questions at once — each main effect and their interaction — and the answer is planted **per model term**, not per cell. `simulate_factorial_groups()` returns `truth_term` beside the wide data so each row of the ANOVA table can be scored.

```python
sim_fact = sa.simulate_factorial_groups(seed=2026)

# Naming a reference level per factor fixes what `effect` is measured against.
control_label = {name: list(levels)[0] for name, levels in sim_fact.args["factor_lv"].items()}
control_label
#> {'treatment': 'control', 'sex': 'male'}

fact = sa.compare_factorial_groups(**sim_fact.args, control_label=control_label)
fact
```

```
<SaFactorial> factorial_comparison
  factors  : treatment (4) x sex (2)  (8 cells, independent)
  anova    : two-way, Type III sums of squares
  features : 100
  settings : alternative = two.sided, conf_level = 0.95, p_adjust = BH

  tests
    $anova_test  21 of 100 at pval_adj <= 0.05
                Two-way ANOVA (Type III sums of squares)
                post-hoc: 85 of 216 contrast(s) over 24 feature(s), Tukey HSD on marginal means and simple effects

  terms
    treatment      16 of 100 at pval_adj <= 0.05
    sex            8 of 100 at pval_adj <= 0.05
    treatment:sex  7 of 100 at pval_adj <= 0.05

  $diagnostics attached
```

```python
fact.effect.head(3)
```

```
  features  n_used  n_cells  ref_center    extreme_cell  extreme_center  fold_change    log2fc
0   prot_1   160.0      8.0   13.229816  control.female       36.956599     2.793432  1.482039
1   prot_2   160.0      8.0  225.820769    treat_B.male      587.785393     2.602885  1.380111
2   prot_3   160.0      8.0  119.120058  control.female       50.195071     0.421382 -1.246799
```

The whole-model F in `fact.tests["anova_test"]` is the same test `estimate_significance()` and `draw_forest_plot()` already know from multi-group comparisons. Term-wise inference lives in `fact.terms`, one row per feature and model term, with `pval_adj` corrected **across features within each term** rather than across terms. `fact.cells` holds the cell means the design was built from.

```python
sig_term = sa.estimate_significance(fact, by="term")
sig_term
sa.draw_volcano_plot(sig_term)
```

```
<SaSignificance> factorial_comparison
  test     : anova_test  (Two-way ANOVA (Type III sums of squares))
  cutoffs  : abs(log2_effect) >= 1, adj_pvalue <= 0.05  (BH)

  $significance, one table per term
    treatment      16 of 100 significant
    sex            4 of 100 significant
    treatment:sex  6 of 100 significant
```

![Term-wise volcano plot of the factorial design](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-factorial-volcano.png)

`by="term"` puts each term's `log2_effect` on its own panel and plots the magnitude, so the axis is a size rather than an up/down fold change.

The shape worth looking for is a **crossover**: a feature whose main effects are exactly zero and whose interaction alone was moved. It is invisible in a one-factor read of treatment, and visible the moment the lines cross. `truth_term` is what says which features those are:

```python
truth_term = sim_fact.truth_term
is_interaction = truth_term["term_order"] > 1

crossovers = [
    name
    for name, rows in truth_term.groupby("features", sort=False)
    if rows.loc[is_interaction[rows.index], "is_effect"].any()
    and not rows.loc[~is_interaction[rows.index], "is_effect"].any()
]
crossover = crossovers[0]

truth_term[truth_term["features"] == crossover]
```

```
   features          terms  term_order  is_within  max_abs_delta  is_effect
9    prot_4      treatment           1      False       0.000000      False
10   prot_4            sex           1      False       0.000000      False
11   prot_4  treatment:sex           2      False       1.210508       True
```

The fitted terms recover exactly that:

```python
fact.terms.loc[
    fact.terms["features"] == crossover,
    ["terms", "f_stat", "partial_eta_sq", "log2_effect", "pval_adj"],
]
```

```
            terms    f_stat  partial_eta_sq  log2_effect  pval_adj
9       treatment  0.407388        0.007976     0.316236  0.973394
10            sex  0.353675        0.002321    -0.106964  0.888042
11  treatment:sex  6.496097        0.113642     1.136998  0.012147
```

Both main effects sit at p of 0.97 and 0.89, the interaction at 0.012, and the planted magnitude of 1.21 comes back as 1.14. The interaction panel is the place to read it: the treatment profile runs one way in `male` and the opposite way in `female`, inside a single feature panel rather than by comparing panels across the page.

```python
sa.draw_interaction_plot(fact, feats=crossover, errorbar="se")
```

![Interaction plot of cell means for prot_4](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-factorial-interaction.png)

### 9. Categorical contingency tables

A contingency table has no feature axis. The question is about the **table as a whole**, or about one cell at a time, and the result is `SaCategorical` rather than a comparison. `design["null"]` names the hypothesis the expected counts and residuals were read under — independence here — and every downstream function reads that same null.

```python
sim_cat = sa.simulate_categorical_groups(seed=2026)
cat = sa.compare_categorical_groups(**sim_cat.args)
cat
```

```
<SaCategorical> categorical_comparison
  table    : cat_1 (2) x cat_2 (3)  (6 cells, independent)
  null     : independence -- a cell is expected at the product of its margins
  observed : 200 row(s)
  settings : conf_level = 0.95, correct = True

  tests
    $chisq_test   pval = 0.012  (null rejected at 0.05)
                 Chi-square test of independence
    $fisher_test  pval = 0.0115  (null rejected at 0.05)
                 Fisher's exact test

  association
    cramers_v                0.21
    contingency_coefficient  0.206

  $diagnostics attached, rule expected_count_min: met
```

`cells` is the canonical form, one row per cell, because that is the shape which survives being written out as JSON with its labels attached. A table is the shape to read it in, so it is built on request:

```python
cat.as_table()
```

```
cat_2  high  mid  low
cat_1                
y        36   27   29
n        26   53   29
```

`estimate_significance()` refuses a contingency result rather than reading it as a comparison, since there is no feature axis for it to threshold. `estimate_categorical_significance()` is the one that takes this object, and it answers two different questions depending on `by=`. The cell reading scores each `(row_level, col_level)` pair by how far `observed / expected` sits from one, with p-values from the Pearson residual:

```python
sig_cell = sa.estimate_categorical_significance(cat, by="cell")
sig_cell.significance
```

```
  row_level col_level  observed  expected      lift  log2_lift  std_residual    pvalue  adj_pvalue  is_signif
0         y      high      36.0     28.52  1.262272   0.336023      2.294592  0.021757    0.032635      False
1         n      high      26.0     33.48  0.776583  -0.364788     -2.294592  0.021757    0.032635      False
2         y       mid      27.0     36.80  0.733696  -0.446746     -2.838113  0.004538    0.013614      False
3         n       mid      53.0     43.20  1.226852   0.294961      2.838113  0.004538    0.013614      False
4         y       low      29.0     26.68  1.086957   0.120294      0.725386  0.468215    0.468215      False
5         n       low      29.0     31.32  0.925926  -0.111031     -0.725386  0.468215    0.468215      False
```

At the default cutoffs every cell misses — the omnibus test already rejected independence, and four of the six cells clear an adjusted p-value, but none of them also clears the lift cutoff. That is a different verdict from the table reading, which is one row and one association measure:

```python
sa.estimate_categorical_significance(cat, by="table", test="chisq_test")
```

```
<SaCategoricalSignificance> categorical_comparison
  reading  : table  (2 x 3 table)
  null     : independence -- a cell is expected at the product of its margins
  test     : chisq_test  (Chi-square test of independence)
  cutoffs  : pvalue <= 0.05
  verdict  : cramers_v = 0.21  (significant)
```

`draw_mosaic_plot()` shades each tile by the residual under the same null, and draws the expected conditional proportion as a dashed line inside each strip so the eye reads distance from the null rather than distance from the neighbouring strip:

```python
sa.draw_mosaic_plot(cat)
```

![Mosaic plot shaded by Pearson residuals under independence](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-mosaic.png)

Repeated binary conditions swap the null and the tests: `paired=True` reads McNemar for two conditions and Cochran's Q for more, and `design["null"]` says so.

### 10–13. One sample, assumptions, descriptives and bars

Four functions that need less space than the sections above, in the order a real analysis reaches them.

`compare_one_sample()` returns the same contract as the two-group call, so `draw_forest_plot()` reads it without being told which scenario produced it. It runs a one-sample t-test, a signed-rank test with a Hodges-Lehmann pseudo-median, and a score test with a Wilson interval for binary features:

```python
one = sa.compare_one_sample(sim.args["data"], strongest, mu=8)
one.tests["t_test"]
```

`diagnose_distribution()` checks each assumption twice, by tests that fail differently: Shapiro-Wilk against Kolmogorov-Smirnov for normality, median-centred Levene against Bartlett for homogeneity of variance. A failed check never blocks an analysis and never swaps one test for another — it changes which member of the reported family deserves the most weight. The same checks are attached to every comparison as `diagnostics` unless `diagnose=False`:

```python
d = sa.diagnose_distribution(sim.args["data"], sim.args["feats"], sim.args["group"])
d.normality    # one row per feature and level
d.variance     # one row per feature
d.summary      # normal_ok / variance_ok flags per feature
```

`screen_outliers()` flags observations and **does not remove them**. `row` is the row number in the original `data`, so a flagged point can be looked up:

```python
sa.screen_outliers(sim.args["data"], first_ten, sim.args["group"])          # 1.5 x IQR fences
sa.screen_outliers(sim.args["data"], first_ten, criterion="robust_z")
sa.screen_outliers(sim.args["data"], first_ten, criterion="grubbs", alpha=0.05)
```

`summarize_descriptive_stats()` returns one row per feature, or per feature and level when a group is given, with counts, centres, spreads, quartiles, IQR fences, MAD, skewness and excess kurtosis:

```python
sa.summarize_descriptive_stats(sim.args["data"], first_ten[:3])
sa.summarize_descriptive_stats(
    sim.args["data"], strongest, sim.args["group"], group_lv=sim.args["group_lv"]
)
```

`draw_grouped_boxplot()` shows the spread a group's observations have. `draw_grouped_barplot()` shows one number standing for them — a mean, a median, or any column `summarize_descriptive_stats()` already computed — with an error bar whose meaning follows the height. The bar and the table row are the same number because the heights are read from that function rather than recomputed, and `errorbar` is rejected when it cannot be read under `mainbar`: a mean takes a standard error or an interval, a median takes a notch, a count takes none.

```python
sa.draw_grouped_barplot(
    data=sim.args["data"],
    feats=first_ten,
    group=sim.args["group"],
    group_lv=sim.args["group_lv"],
    errorbar="se",
)
```

`center_by_control()` sits beside them: it divides, or on the log2 scale subtracts, the control centre out of every feature, so each value reads as its distance from the control. It takes the arguments the comparisons take, in the same order, so the two steps are one set of arguments.

### 14. Feature-pair association

Nothing in Part 1 yet asked how **two** features move together. `summarize_association_stats()` is a screen, not a contract: Pearson, Spearman and Kendall come back side by side on the same pairs, each as four square matrices — coefficient, p-value, adjusted p-value and the observations the pair shared.

```python
assoc_cor = sa.make_block_cor(
    n_features=10,
    blocks=[
        {"features": range(0, 3), "cor": 0.9},
        {"features": range(3, 5), "cor": 0.5, "against": range(5, 7)},
    ],
)

assoc_sim = sa.simulate_regression(n_pred=10, n_factor_pred=0, cor_mat=assoc_cor, seed=2026)
assoc = sa.summarize_association_stats(
    data=assoc_sim.args["data"], feats=assoc_sim.args["predictors"]
)

assoc["design"]
```

```
{'feats': ['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10'],
 'n_obs': 200,
 'methods': ['pearson', 'spearman', 'kendall'],
 'adj_type': 'BH',
 'use': 'pairwise.complete.obs'}
```

```python
assoc["pearson"]["corr"].iloc[:3, :3]
```

```
          x_1       x_2       x_3
x_1  1.000000  0.910209  0.893620
x_2  0.910209  1.000000  0.927179
x_3  0.893620  0.927179  1.000000
```

The `against` block planted a positive correlation inside `x_1`–`x_3` and `x_4`–`x_5`, and a negative one between those two sides, which is why the upper-left block reads near 0.9 and the cross-block cells read near -0.5. `make_block_cor()` refuses a matrix that is symmetric with a unit diagonal but describes no data that could exist, which is why `against` exists at all: `-1/(k - 1)` is the floor on one value shared by `k` predictors, so three of them cannot disagree past -0.5 while a split block has no such limit.

`draw_corrplot()` is three decisions on top of `draw_heatmap()`: nothing is standardised, the colours are fixed at -1 to 1, and both axes share one clustering order so the diagonal stays diagonal.

```python
sa.draw_corrplot(assoc["pearson"]["corr"])
sa.draw_corrplot(assoc["pearson"]["corr"], pvalue=assoc["pearson"]["adj_pvalue"])
```

| All pairs | Pairs that cleared BH at 0.05 |
| --- | --- |
| ![Correlation heatmap of ten predictors](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-corrplot.png) | ![Correlation heatmap with non-significant cells blanked](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-corrplot-masked.png) |

Blanking happens **after** clustering, so the tree is built on the full matrix the reader is being shown. The distance is `1 - cor()`, the same rule `cluster_hclust()` and `draw_heatmap(dist_method="correlation")` use.

---

# Part 2 — Modelling

A model has no feature axis. Every table in Part 1 repeats `features` in the same order; a model has one outcome and a set of **terms**, and the terms are not the columns that were handed in, since one factor predictor becomes several. `terms` takes the place of `features`, `coefficients["terms"]` repeats that order, and the ten slots of an `SaModel` are the same ten whichever of the five models produced it. §18 and §19 are searches rather than fits, and they have an axis of their own again: `candidates`, the columns they were asked to choose between. §23 and §24 are the first sections that **score** fitted models on held-out rows rather than fit or search on training ones.

### 15. Data whose coefficients are known, and a split that does not leak

`make_block_cor()` builds the correlation the predictors are drawn with. The blocks below straddle the planted predictors on purpose, and `noise_sd` is set below the simulator's default of 3 so that there is something for Part 2 to find on 200 rows:

```python
cor_mat = sa.make_block_cor(
    n_features=8,
    blocks=[
        {"features": range(1, 3), "cor": 0.8},
        {"features": range(4, 7), "cor": 0.5},
    ],
)

sim_reg = sa.simulate_regression(cor_mat=cor_mat, noise_sd=2, seed=2026)
sim_reg.truth
```

```
  predictors    role      beta direction  value_mean  value_sd  max_cor_signal
0        x_1  signal  1.685777        up         0.0       1.0             0.0
1        x_2  signal  1.857716        up         0.0       1.0             0.0
2        x_3    null  0.000000      none         0.0       1.0             0.8
3        x_4    null  0.000000      none         0.0       1.0             0.0
4        x_5    null  0.000000      none         0.0       1.0             0.5
5        x_6  signal -0.766030      down         0.0       1.0             0.5
6        x_7  signal -1.479177      down         0.0       1.0             0.5
7        x_8    null  0.000000      none         0.0       1.0             0.0
8    x_cat_1  factor       NaN       NaN         NaN       NaN             NaN
```

Four predictors carry a coefficient and four are exactly zero, so a false positive is a count rather than an estimate. `max_cor_signal` is why the correlation blocks are there at all: `x_3` is null but correlates with the planted `x_2` at 0.8, and a null predictor that correlates with a planted one is pulled off zero by data alone. No number of rows fixes that, and every section below runs into it.

Which predictors get planted is itself drawn from the random stream, so the same `cor_mat` does not land the same way in the R package as it does here. `truth` is the record of what this run actually planted, which is why every claim below points at it rather than at the call that produced it.

`truth` has one row per predictor; `truth_term` has one row per term, aligned with `coefficients` by position, since a three-level factor is two terms and a constant predictor is none.

`split_data()` defines what "the training half" means, and closes the two ways a training set learns what it must not. `stratified` keeps the balance of the whole data on both sides, and `id` sends every row of one sampling unit to the same side.

```python
reg_data = sim_reg.args["data"]
outcome = sim_reg.args["outcome"]
predictors = sim_reg.args["predictors"]

dataset = sa.split_data(
    data=reg_data,
    stratified=reg_data["x_cat_1"],
    p_train=0.75,
    times=1,
    seed=2026,
)
dataset

train_data = dataset.datasets["Resample1"]["train_data"]
test_data = dataset.datasets["Resample1"]["test_data"]
```

```
<SaSplit> train/test partition
  rows     : 200
  stratify : <vector>
             high 66, low 67, mid 67
  settings : p_train = 0.75, times = 1, seed = 2026

  splits
    $Resample1  train 152 / test 48  (p = 0.760)
```

`p_train` is a proportion of rows, or of units when `id` is given, and the row proportion actually reached is reported as `p` above and stored in `parameters["achieved_p"]`. The shape does not depend on `times`: `datasets` is a dict of one when one split was asked for.

`sim_reg.split_args` is named after the parameters of `split_data()` for the same reason `args` is named after the parameters of the model, so either can be spread out with `**`.

### 16. Linear regression

Every model takes `data`, `outcome` and `predictors`, and every one resamples the same way. Cross-validation here scores the fit and does not choose it: the final model is fitted on all usable rows either way, so `cv=True` and `cv=False` give identical coefficients and differ only in `performance` and `resampling`.

```python
lin = sa.fit_linear_regression(
    data=train_data,
    outcome=outcome,
    predictors=predictors,
    cv=True,
    cv_method="repeated_kfold",
    n_fold=10,
    n_repeat=3,
    seed=2026,
)
lin
```

```
<SaModel> linear_regression
  outcome  : y  (continuous)
  rows     : 152 used
  terms    : 11 over 9 predictor(s)
  settings : repeated_kfold, 10 fold(s) x 3 repeat(s), conf_level = 0.95

  coefficients
    (Intercept)     -0.0959  [-0.643, 0.451]  p = 0.729
    x_1                1.85  [1.54, 2.16]  p = <1e-16
    x_2                1.96  [1.43, 2.49]  p = 1.88e-11
    x_3              -0.102  [-0.585, 0.381]  p = 0.677
    x_4               0.163  [-0.152, 0.478]  p = 0.307
    x_5             -0.0408  [-0.459, 0.377]  p = 0.847
    x_6               -0.75  [-1.17, -0.328]  p = 0.000594
    x_7               -1.46  [-1.81, -1.1]  p = 3.02e-13
    x_8             -0.0155  [-0.328, 0.297]  p = 0.922
    x_cat_1mid         1.64  [0.862, 2.42]  p = 5.29e-05
    ... and 1 more term(s) in $coefficients

  fit      : r_squared = 0.797, adj_r_squared = 0.783, sigma = 1.94, f_stat = 55.4, df1 = 10, df2 = 141, pval = 7.31e-44, aic = 646, bic = 682
  resample : RMSE = 2.02 (SD 0.331), Rsquared = 0.758 (SD 0.104), MAE = 1.66 (SD 0.291) over 30 resample(s)
```

`lin.coef()` is the whole table. `lin.fit` is the fitted estimator underneath, for anything the contract does not carry:

```python
lin.coef()[["terms", "estimate", "pval"]]
```

```
          terms  estimate          pval
0   (Intercept) -0.095886  7.293041e-01
1           x_1  1.853455  8.900721e-23
2           x_2  1.957413  1.878594e-11
3           x_3 -0.102058  6.769894e-01
4           x_4  0.163168  3.071302e-01
5           x_5 -0.040843  8.471814e-01
6           x_6 -0.749665  5.941897e-04
7           x_7 -1.456682  3.017039e-13
8           x_8 -0.015465  9.221503e-01
9    x_cat_1mid  1.639358  5.293973e-05
10  x_cat_1high -1.169092  4.585224e-03
```

All four planted coefficients clear 0.05 and land within 0.17 of the values §15 planted, and the four planted zeros come back between p = 0.31 and p = 0.85. `x_3` — the null that correlates with the planted `x_2` at 0.8 — is estimated at -0.102 rather than at zero, which is `max_cor_signal` doing exactly what it records. It did not fool the p-value here, and no amount of `n_fold` guarantees that it will not.

`fit_stats` is a dict rather than a table, because these are quantities per model and not per term: `r_squared`, `adj_r_squared`, `sigma`, the F test, `aic` and `bic`.

### 17. Logistic regression

The same call with a two-class outcome. `outcome_lv` follows the `group_lv` rule — the first level is the reference — so the coefficients describe the odds of `outcome_lv[1]`, and a list handed to both `compare_two_groups()` and this function points the same way in both.

```python
sim_cls = sa.simulate_classification(cor_mat=cor_mat, seed=2026)
cls_data = sim_cls.args["data"]
cls_outcome = sim_cls.args["outcome"]
cls_predictors = sim_cls.args["predictors"]
outcome_lv = sim_cls.args["outcome_lv"]

cls = sa.split_data(
    data=cls_data,
    stratified=cls_data[cls_outcome],   # about one row in three is an event
    p_train=0.75,
    times=1,
    seed=2026,
)
cls_train = cls.datasets["Resample1"]["train_data"]
cls_test = cls.datasets["Resample1"]["test_data"]

log_fit = sa.fit_logistic_regression(
    data=cls_train,
    outcome=cls_outcome,
    predictors=cls_predictors,
    outcome_lv=outcome_lv,
    cv=True, cv_method="repeated_kfold", n_fold=10, n_repeat=3, seed=2026,
)
log_fit.coef()[["terms", "odds_ratio", "pval"]]
```

```
          terms  odds_ratio      pval
0   (Intercept)    0.051750  0.000095
1           x_1    7.693181  0.000091
2           x_2   14.591653  0.001969
3           x_3    0.643857  0.435961
4           x_4    1.285238  0.421701
5           x_5    1.236341  0.620916
6           x_6    0.336219  0.030619
7           x_7    0.121111  0.000114
8           x_8    0.806454  0.537624
9    x_cat_1mid    8.578163  0.012309
10  x_cat_1high    0.163621  0.056831
```

All four planted numeric predictors clear 0.05 here, and the two that were planted `down` come back with an odds ratio under one. This half of Part 2 is a separate simulation with its own outcome, and it has 47 events to spend on eleven terms where §16 had 152 continuous responses, so read the p-values as the thinner evidence they are — the `x_6` that arrived at p = 0.0006 there is at p = 0.031 here on the same planted coefficient.

The interval columns are `or_lower_conf` and `or_upper_conf`, exponentiated from the Wald interval on the log-odds scale rather than profiled, so the two numbers always agree with the standard error in the same row.

`x_cat_1` is where a term stops being a predictor: one of its two dummy terms clears 0.05 and the other does not. Keeping "the columns behind the significant terms" would put the factor on both sides of any comparison built that way, which is the trap §18 avoids.

### 18. Recursive feature elimination

§17 selected predictors the way most analyses do: fit once, read the p-values, keep what cleared 0.05. Two things are wrong with that even when the answer comes out right. The threshold is arbitrary, and the p-values that chose the predictors came from all the training rows, so the resampled score of the model that follows describes a fit whose predictors were already chosen — the choosing sits outside the resampling that reports on it.

`perform_rfe()` asks the same question with the elimination **inside** the resampling: rank the candidates, drop the weakest, score what is left, and repeat until one predictor is standing. There is no `cv` argument, because an elimination with nothing held out has no score to choose a size by.

```python
rfe = sa.perform_rfe(
    data=cls_train,
    outcome=cls_outcome,
    predictors=cls_predictors,
    outcome_lv=outcome_lv,
    model="logistic",
    seed=2026,
)
rfe
```

```
<SaSelection> rfe
  outcome  : y  (two classes)
             modelling case against control, 47 of 151 row(s)
  rows     : 151 used
  search   : Binomial logistic regression over 9 candidate(s), size(s) 1, 2, 3, 4, 5, 6, 7, 8, 9
  settings : repeated_kfold, 5 fold(s) x 5 repeat(s), Accuracy maximised
  selected : 5 of 9  (Accuracy = 0.879 (SD 0.049) over 25 resample(s))

  ranking  (absolute Wald z)
    x_1           3.368  selected
    x_7           3.322  selected
    x_2           2.665  selected
    x_cat_1       2.249  selected
    x_6           1.911  selected
    x_3           0.734  dropped
    x_4          0.7287  dropped
    x_8          0.6957  dropped
    x_5          0.5353  dropped
```

This is `SaSelection`. `candidates` takes the place `features` holds in a comparison, `terms` in a model and `points` in a reduction, and two tables hang off it because "which predictors" and "how many" are two different answers: `ranking` has one row per candidate, `profile` one row per subset size.

Two details of the ranking are worth the space. It is the absolute Wald z rather than the coefficient, so a predictor measured in grams and the same predictor in kilograms are eliminated in the same order — a coefficient is an effect per unit, and ranking by its size ranks by the units. And `x_cat_1` is ranked as one candidate rather than as its two dummy terms, which is exactly the trap §17 pointed at. Here a factor is kept or dropped as a column, which is the only thing a later `predictors=` could accept.

```python
rfe.selected
#> ['x_1', 'x_7', 'x_2', 'x_cat_1', 'x_6']

list(sim_cls.truth.loc[sim_cls.truth["role"] != "null", "predictors"])
#> ['x_1', 'x_2', 'x_6', 'x_7', 'x_cat_1']
```

The five it kept are the five that were planted, and no threshold was named anywhere in the call. `x_3` is the interesting one at the other end: it is null but correlates with the planted `x_2` at 0.8, which is what `max_cor_signal` in §15 warned about, and it still lands in the bottom four rather than being carried in by that correlation.

How many to keep is a resampled number like any other, and `profile` is where it is kept honest:

```python
rfe.profile
```

```
   n_vars  Accuracy  AccuracySD     Kappa   KappaSD  chosen
0       1  0.725720    0.051547  0.304931  0.119653   False
1       2  0.821462    0.054564  0.572216  0.127458   False
2       3  0.849032    0.052631  0.632521  0.132385   False
3       4  0.875226    0.070375  0.704349  0.170701   False
4       5  0.879312    0.049142  0.713644  0.114515    True
5       6  0.871355    0.053122  0.693965  0.125829   False
6       7  0.864774    0.052758  0.677853  0.126061   False
7       8  0.867441    0.052732  0.683007  0.129953   False
8       9  0.866108    0.051401  0.680508  0.125887   False
```

Accuracy climbs steeply to four predictors, peaks at five, and then falls back and stays flat: everything after the fifth is a parameter spent on noise. Five wins over four by 0.004 against standard deviations of 0.05 to 0.07 on those same rows, so the top of this curve is a near-tie and reading it is the point of it being a table. `selected` is a list of column names and nothing else, so it goes straight back into a fit.

### 19. Stepwise selection by information criteria

§18 bought its answer with resampling, and paid the full price: nine subset sizes, twenty-five resamples each. `perform_stepwise()` asks the same question and settles the bill another way. It walks one term at a time — drop the one whose absence costs least, refit, stop when no move helps — and judges every move by an information criterion, which is the likelihood of the model with a flat charge levied against the number of parameters it spent. Nothing is held out, so there is no `cv` argument and no `seed` either: the path is a deterministic consequence of the data and the charge.

```python
step = sa.perform_stepwise(
    data=cls_train,
    outcome=cls_outcome,
    predictors=cls_predictors,
    outcome_lv=outcome_lv,
    model="logistic",
    criterion="AIC",
)
step
```

```
<SaSelection> stepwise
  outcome  : y  (two classes)
             modelling case against control, 47 of 151 row(s)
  rows     : 151 used
  search   : Binomial logistic regression over 9 candidate(s), 4 step(s)
  settings : backward search, AIC minimised at 2 per parameter
  selected : 5 of 9  (AIC = 81.4698)

  ranking  (AIC increase when the predictor is left out)
    x_1           31.84  selected
    x_7           24.86  selected
    x_2           22.96  selected
    x_cat_1       14.41  selected
    x_6           3.516  selected
    x_8          -1.309  dropped
    x_3          -1.433  dropped
    x_4          -1.496  dropped
    x_5          -1.986  dropped
```

Same contract, same `candidates` axis, and two slots that mean something different here. `ranking["estimate"]` is what leaving that one predictor out of the selected model would cost the criterion, so unlike §18's absolute Wald z it has a sign, and the sign is the verdict: positive for the five worth their parameters, negative for the four the model is better off without. `parameters["maximize"]` is `False` for the same reason, since a criterion is a cost and not a score, and `resampling` is `None` because nothing was resampled.

`profile` is a different table on this side. §18's is a ladder, one row per subset size with all nine of them scored; this one is a path, one row per step, and `step` names the move that reached it:

```python
step.profile
```

```
   n_vars        AIC         BIC   step  chosen
0       9  87.668739  120.858817          False
1       8  85.913957  116.086755  - x_5   False
2       7  84.257573  111.413091  - x_8   False
3       6  82.903204  107.041443  - x_4   False
4       5  81.469781  102.590740  - x_3    True
```

The first row is the model the search started from, which is why its `step` is empty, and `chosen` is `True` on the last row because a stepwise search stops where it chose. The four sizes below five are absent by construction: this is a record of where the walk went, not a survey of every size.

The path drops exactly the four planted zeros, in ascending order of what they were worth, and stops:

```python
step.selected
#> ['x_1', 'x_7', 'x_2', 'x_cat_1', 'x_6']
```

Both criteria sit on every row, so the same path is readable on either scale, and here they agree. BIC charges `log(151)` = 5.02 per parameter against AIC's 2, and the `BIC` column falls at every step of the path, so the heavier charge walks the same way and stops in the same place:

```python
sa.perform_stepwise(
    data=cls_train, outcome=cls_outcome, predictors=cls_predictors,
    outcome_lv=outcome_lv, model="logistic", criterion="BIC",
).selected
#> ['x_1', 'x_7', 'x_2', 'x_cat_1', 'x_6']
```

Two searches with nothing in common but the data arrived at the same five, and both arrived at the five that were planted. That is what a well-separated signal looks like, and not something to count on — the AIC path and the BIC path are the same object only until one predictor sits near the charge.

### 20–22. Penalized, forest and kernel models

The three remaining engines return the same `SaModel` as §16 and §17. They are shown here in one place because the interface is the point: what changes is what the model can honestly say about a term, not how it is called or read.

All of them are fitted on one set of predictors, chosen once by RFE on the training half, so that §23 compares engines rather than column sets:

```python
kept = sa.perform_rfe(
    data=train_data, outcome=outcome, predictors=predictors, seed=2026
).selected
kept
#> ['x_1', 'x_7', 'x_2', 'x_cat_1', 'x_6']

lin_kept = sa.fit_linear_regression(
    data=train_data, outcome=outcome, predictors=kept,
    cv=True, cv_method="repeated_kfold", n_fold=10, n_repeat=3, seed=2026,
)
```

**`fit_elastic_net()`** covers the three corners of one model: `penalty="lasso"` is alpha 1, `"ridge"` is alpha 0, and `"elastic_net"` tunes alpha as well. This is the first model where resampling **chooses** rather than scores, so `parameters["lambda"]` and `parameters["alpha"]` are the values that won and the grid is the rows of `performance`.

```python
lasso = sa.fit_elastic_net(
    data=train_data, outcome=outcome, predictors=kept, penalty="lasso",
    lambda_=(0.01, 0.1, 0.5, 1.0, 2.0),
    cv=True, cv_method="repeated_kfold", n_fold=10, n_repeat=3, seed=2026,
)
lasso.coef()
```

```
         terms  estimate  selected
0  (Intercept) -0.096895      True
1          x_1  1.844899      True
2          x_7 -1.431609      True
3          x_2  1.839450      True
4   x_cat_1mid  1.636070      True
5  x_cat_1high -1.125199      True
6          x_6 -0.756575      True
```

There are no `stderr`, `pval` or interval columns here, and they are **absent rather than empty**. A penalized estimate is deliberately biased and the usual standard error assumes an unbiased one, so there is no honest number to put in them; a column of missing values reads as a table whose values are missing, which is a different claim. `selected` takes their place, and `"pval" not in lasso.coef()` is how a consumer tells the two kinds of table apart. Every term keeps its row either way: a dropped term is `estimate` of exactly 0, not a missing row. At the winning lambda nothing is dropped here, because §23's search had already removed the four predictors a penalty would have taken.

**`fit_rf()`** is the first model with no coefficients. A forest holds hundreds of trees and their splits, not one effect per predictor, so `estimate` is **permutation importance** and the table is sorted by it. `impurity` carries the other measure the same fit reports, because the two disagree in a way worth seeing: permutation is measured on held-out rows, impurity on the splits themselves.

```python
rf = sa.fit_rf(data=train_data, outcome=outcome, predictors=kept, cv=True, seed=2026)
rf.coef()
```

```
     terms  estimate  impurity
0      x_1  1.223723  0.322557
1      x_2  1.011330  0.264230
2      x_7  0.743515  0.210432
3      x_6  0.406767  0.124209
4  x_cat_1  0.391993  0.078573
```

The two measures agree on the order here, and both put the two strongest planted coefficients on top. They do not always: `x_cat_1` is last on permutation by a hair and last on impurity by a wide margin, which is the gap the second column is carried for — impurity counts how often the trees split on a predictor, permutation measures what breaking it costs on rows the trees did not see. A forest splits factors by level directly, so `x_cat_1` is one term rather than two — unlike every other model in this part. `fit_stats` is out-of-bag rather than in-sample and says so in its names: `oob_r_squared`, `oob_rmse`, `oob_mae`, and for classification `oob_accuracy` and the rest.

**`fit_svm()`** is the second model with no coefficients, for the opposite reason. A forest has too many numbers per predictor to report one; a radial kernel machine has **none** — it holds support vectors and their weights, which are points in the data rather than directions in the predictor space. So `estimate` is permutation importance again, and `coef()` carries `terms` and `estimate` and nothing else:

```python
svm = sa.fit_svm(data=train_data, outcome=outcome, predictors=kept, cv=True, seed=2026)
list(svm.coef().columns)
#> ['terms', 'estimate']
```

`sigma=None` reads the kernel width from the data, and the predictors are centred and scaled before the kernel measures a distance, so `sigma` is a width on the standardised scale.

### 23. Score regression models on held-out rows

`evaluate_regression_models()` is the layer above `SaModel.predict()`: the same held-out rows for every model, one table of metrics and one of deltas against a baseline. Rows are the **intersection** of what every model could predict, not the union, because a delta only means something when the two numbers came from the same rows.

The four models of §16 and §20–§22, all fitted on the `kept` predictors, are scored against the linear one:

```python
eval_reg = sa.evaluate_regression_models(
    baseline_model=lin_kept,
    new_models={"lasso": lasso, "rf": rf, "svm": svm},
    newdata=test_data,
    answer=test_data[outcome],
    baseline_label="linear",
)
eval_reg
```

```
<SaPerformance> regression_performance
  outcome  : <vector>  (continuous)
  rows     : 48 scored
  models   : 4, baseline = linear

  metrics
    linear  cor = 0.83, r_squared = 0.671, rmse = 2.05, mae = 1.67
    lasso   cor = 0.83, r_squared = 0.672, rmse = 2.05, mae = 1.66
    rf      cor = 0.78, r_squared = 0.59, rmse = 2.29, mae = 1.78
    svm     cor = 0.733, r_squared = 0.492, rmse = 2.55, mae = 2.12

  comparisons  (against linear)
    lasso  delta_cor = -0.000146, delta_r_squared = 0.00063, delta_rmse = -0.00196, delta_mae = -0.00285
    rf     delta_cor = -0.0498, delta_r_squared = -0.0805, delta_rmse = 0.237, delta_mae = 0.117
    svm    delta_cor = -0.0964, delta_r_squared = -0.179, delta_rmse = 0.498, delta_mae = 0.453
```

The truth behind this data is linear, so the linear model wins and the two flexible engines pay for their flexibility on 152 rows. LASSO ties it to three decimal places, which is what a penalty does when the columns it would have shrunk are already gone.

```python
eval_reg.metrics
```

```
    model  n_used       cor  r_squared      rmse       mae      bias  calib_slope  calib_intercept
0  linear      48  0.829751   0.670991  2.050732  1.666903  0.405223     0.745065         0.576879
1   lasso      48  0.829605   0.671621  2.048769  1.664054  0.402255     0.740478         0.577000
2      rf      48  0.779939   0.590459  2.287989  1.783529  0.149703     0.509365         0.480065
3     svm      48  0.733319   0.491691  2.548996  2.119981  0.384392     0.673979         0.603914
```

There is no p-value beside the deltas: held-out error on rows the caller did not generate has no null the package can name. `cor` and `r_squared` sit together because their gap is what `calib_slope` and `calib_intercept` report — a slope under one is a model whose predictions are squeezed towards their own mean, which all four of these are.

`draw_prediction_plot()` reads those calibration numbers rather than refitting, so the line and the table cannot drift apart:

```python
sa.draw_prediction_plot(eval_reg, type="overlay", anno_corr=True, anno_rsq=True)
```

![Predicted against observed for four regression models on the same held-out rows](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-eval-regression.png)

### 24. Score classification models on held-out rows

The same intersection rule and the same RFE-first pipeline, with logistic regression as the baseline. Here the comparisons add three paired questions beside `delta_auc`: DeLong's test on the ranks, the IDI on the probabilities, and the NRI on how often each probability moved the right way.

The four classifiers are the same four engines, fitted on the predictors §18's search kept:

```python
cls_kept = rfe.selected
shared = {
    "data": cls_train, "outcome": cls_outcome, "predictors": cls_kept,
    "outcome_lv": outcome_lv,
    "cv": True, "cv_method": "repeated_kfold", "n_fold": 10, "n_repeat": 3, "seed": 2026,
}

log_kept = sa.fit_logistic_regression(**shared)
lasso_cls = sa.fit_elastic_net(**shared, penalty="lasso", lambda_=(0.01, 0.1, 0.5, 1.0, 2.0))
rf_cls = sa.fit_rf(**shared)
svm_cls = sa.fit_svm(**shared)

eval_cls = sa.evaluate_classification_models(
    baseline_model=log_kept,
    new_models={"lasso": lasso_cls, "rf": rf_cls, "svm": svm_cls},
    newdata=cls_test,
    answer=cls_test[cls_outcome],
    outcome_lv=outcome_lv,
    baseline_label="logistic",
)
eval_cls
```

```
<SaPerformance> classification_performance
  outcome  : <vector>  (two classes)
             scoring the probability of case against control, 15 of 49 row(s)
  rows     : 49 scored
  models   : 4, baseline = logistic
  threshold: 0.5  (accuracy, sensitivity and specificity only)

  metrics
    logistic  auc = 0.951  [0.898, 1], brier = 0.0895, accuracy = 0.857
    lasso     auc = 0.955  [0.905, 1.01], brier = 0.0857, accuracy = 0.878
    rf        auc = 0.937  [0.867, 1.01], brier = 0.101, accuracy = 0.857
    svm       auc = 0.931  [0.864, 0.999], brier = 0.0982, accuracy = 0.816

  comparisons  (against logistic)
    lasso
      delta_auc  = 0.00392  [-0.00689, 0.0147]  p = 0.477
      IDI        = -0.056  [-0.0857, -0.0264]  p = 0.000214
      NRI        = -1.5  [-1.9, -1.09]  p = 5.13e-13
    rf
      delta_auc  = -0.0137  [-0.074, 0.0466]  p = 0.656
      IDI        = -0.192  [-0.302, -0.0814]  p = 0.000656
      NRI        = -1.32  [-1.76, -0.883]  p = 3.48e-09
    svm
      delta_auc  = -0.0196  [-0.0582, 0.019]  p = 0.32
      IDI        = -0.128  [-0.219, -0.0371]  p = 0.00578
      NRI        = -1.07  [-1.57, -0.569]  p = 2.9e-05
```

Not one of the three `delta_auc` values clears 0.05, and all three IDIs are significantly negative. That is the reason three statistics are reported rather than one: the ranks barely change, so the AUC cannot separate these models on 49 rows, while the probabilities themselves moved the wrong way by an amount the IDI can see.

```python
sa.draw_roc_curve(eval_cls, anno_auc=True)
```

![ROC curves of four classifiers scored on the same held-out rows](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-eval-classification.png)

Every number is about the odds of `case`, because that is what the fitted models predict; `outcome_lv` and `control_label` are read as statements about what was already fit, not as knobs to turn after the fact.

### 25. Predict on held-out data

`predict()` is a method on the **result**, not on `fit`. The engine object knows only the array it was handed: it was given a design matrix and reads it by position, so a frame whose numeric columns are in a different order is multiplied by the wrong coefficients without any error, and a factor predictor is not coded at all. The result object is the only thing that knows which columns were predictors and what the levels of a factor were, so one line covers all five models:

```python
lin.predict(newdata=test_data)                       # numeric
log_fit.predict(newdata=cls_test, type="raw")        # labels from outcome_lv
log_fit.predict(newdata=cls_test, type="prob")       # one column per class
log_fit.predict(newdata=cls_test, type="response")   # P(outcome_lv[1])
```

```
array([ 0.94900505,  5.49366905,  3.71081139, -0.16217344,  4.72602662])

array(['control', 'control', 'control', 'case', 'control'], dtype=object)

    control      case
0  0.834922  0.165078
1  0.948047  0.051953
2  0.991615  0.008385
3  0.003519  0.996481
4  0.869999  0.130001

array([0.16507825, 0.05195299, 0.00838523, 0.99648101, 0.13000142])
```

`type="response"` is the second column of `type="prob"`, and `type="raw"` is that column cut at the threshold. Extra columns in `newdata` are ignored, a missing one is an error that names it, and a level the training data never saw is an error that names both the column and the level. A level that is missing from `newdata` is not an error: its dummy column is simply zero, since the levels come from `design["predictor_lv"]` rather than from the new rows. One prediction comes back per row, and a row with a missing value in a predictor is missing rather than dropped, so the answer stays aligned with `newdata`.

---

# Part 3 — Unsupervised learning

Everything above had an answer to score against. These sections have none for the coordinates, and §27 has one for the labels when a grouping was known before any algorithm ran. §26 asks where each point lands when many features are pressed into two dimensions; the three functions answer differently on purpose: PCA is a rotation, so it is reversible and says which feature moved a point, but it only finds straight structure; t-SNE and UMAP find curved structure but cannot say which feature made it. §27 asks which points belong together on the same coordinates, using the same `points` axis the reductions use.

### 26. `perform_pca()`, `perform_tsne()` and `perform_umap()`

Three functions rather than one call with a `methods` argument, because they answer in coordinates that share no scale — nothing but `points` could be joined between them — and because `perplexity`, `n_neighbors` and `metric` each belong to exactly one of them. What makes them comparable is the input: all three read `data` the same way, so the same rows drop for the same reason.

`embedding_scale` chooses which margin becomes the points. The input is one row per sample as everywhere else in the package, and `design["point_type"]` reports which axis was embedded.

```python
red_cor = sa.make_block_cor(
    n_features=8,
    blocks=[
        {"features": range(0, 2), "cor": 0.8},
        {"features": range(2, 5), "cor": 0.5},
        {"features": range(6, 8), "cor": 0.9},
    ],
)
red_data = sa.simulate_classification(cor_mat=red_cor, seed=2026).args["data"]
numeric = [f"x_{i}" for i in range(1, 9)]

pca = sa.perform_pca(
    data=red_data, feats=numeric, embedding_scale="features", center=True, scale=True
)
pca
```

```
<SaReduction> pca
  data     : 200 sample(s) x 8 feature(s)
  points   : 8 feature(s)
  scaling  : centred and scaled
  variance : PC1 27.2308%, PC2 24.2164%, PC3 21.4689%  (3 of 8 component(s), 72.916% cumulative)
```

```python
pca.scores.iloc[:, :3]
```

```
  points        PC1        PC2
0    x_1  -2.803914  10.207512
1    x_2  -2.246913  10.149690
2    x_3  10.639573  -1.125793
3    x_4   9.783704  -3.251261
4    x_5  10.749521  -2.634340
5    x_6   0.394345  -0.315373
6    x_7  -7.136983  -8.963304
7    x_8  -6.710990  -8.893636
```

The three blocks come out as three groups and `x_6`, which belongs to none, sits between them:

```python
sa.draw_dim_reduction_plot(pca, anno_points=True)
```

![PCA of the eight features](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-pca.png)

PCA is a singular value decomposition, so one fit answers both margins at once and the matrix is never turned around. `embedding_scale="features"` rescales the rotation from unit length to variance-weighted length and puts the sample scale in `loadings`; `variance` and `fit` are not touched, so the axis labels are the same either way.

Transposing the input by hand instead is a **third** analysis, not the same one: the decomposition always centres and scales the columns it is given, so `perform_pca(data.T)` standardises samples rather than features. The shapes match, the picture reads, and the answer is to a different question. This is the one mistake in these three functions that produces a plot instead of an error, which is why all three document it.

`perform_tsne()` sees literally the same matrix `perform_pca()` did. Eight points is a small neighbourhood, so `perplexity` has to come down with it rather than keeping the engine default:

```python
tsne = sa.perform_tsne(
    data=red_data, feats=numeric, embedding_scale="features",
    center=True, scale=True, perplexity=(len(numeric) - 1) / 3, seed=2026,
)
tsne
```

```
<SaReduction> tsne
  data     : 200 sample(s) x 8 feature(s)
  points   : 8 feature(s)
  scaling  : centred and scaled
  tsne     : 2 dimension(s), perplexity = 2.33333, theta = 0.5  (seed = 2026)
```

![t-SNE of the eight features](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-tsne.png)

`perform_umap()` is the one that standardises nothing by default, because `metric` is its own argument and `"cosine"` or `"correlation"` compares the shape of a row rather than its size, which already answers what standardising would. Both neighbourhood sizes are read from the engine's limits when they are `None`, and both are read from the number of **points** rather than the number of samples — with eight features that is small enough to be worth a message:

```python
umap_res = sa.perform_umap(
    data=red_data, feats=numeric, embedding_scale="features", n_neighbors=3, seed=2026
)
```

This is the one function in the package that needs an optional extra. Without it the call raises and names the install line rather than failing obscurely, which is why the figure above stops at t-SNE.

The coordinates are `scores` in all three and the engine object is `fit`. `variance` and `loadings` are the two slots only PCA carries, so `"variance" in res` is how a consumer tells a rotation from an embedding.

### 27. Cluster on an embedding

The four `cluster_*()` functions read their input through the same path as the reductions, so a clustering and an embedding of the same frame are about the same rows. `SaCluster` assigns every point a label — `0` is noise for the density methods — and `draw_dim_reduction_plot()` is where those labels meet the coordinates.

Colour and shape are two channels on purpose. A clustering is what the data was found to say; a `group` is what was known before either algorithm ran. One colour per shape is a clustering that recovered the groups; one shape split across colours is a group the data does not see as one thing.

```python
clust_sim = sa.simulate_two_groups(n_feats=50, deg_log2fc=(5, 10), seed=2026)

clust_pca = sa.perform_pca(
    data=clust_sim.args["data"],
    feats=clust_sim.args["feats"],
    embedding_scale="samples",
)

clust_km = sa.cluster_kmeans(
    data=clust_pca.scores,
    feats=["PC1", "PC2"],
    cluster_scale="samples",
    n_clust=len(clust_sim.args["group_lv"]),
    seed=2026,
)
clust_km
```

```
<SaCluster> kmeans
  data     : 100 sample(s) x 2 feature(s)
  points   : 100 sample(s)
  scaling  : centred and scaled
  clusters : 2
  sizes    : #1 n = 50, s = 0.666; #2 n = 50, s = 0.417
  silhouette: mean 0.542 over the 100 assigned sample(s), on the euclidean distance
  kmeans   : k = 2, 25 start(s), 99.9772 within-cluster ss  (seed = 2026)
```

```python
pd.crosstab(
    pd.Series(clust_sim.args["group"], name="group"),
    pd.Series(clust_km.assignments["cluster"].to_numpy(), name="cluster"),
)
```

```
cluster   1   2
group          
case      0  50
control  50   0
```

Fifty features were moved up or down on a log2 scale of five to ten, embedded in the first two principal components of the samples, and cut into two clusters. Every control lands in cluster 1 and every case in cluster 2, which is the grouping that was planted — read through shape on the first plot and through colour on the second:

```python
sa.draw_dim_reduction_plot(
    clust_pca, group=clust_sim.args["group"], group_lv=clust_sim.args["group_lv"]
)
sa.draw_dim_reduction_plot(clust_pca, cluster_result=clust_km)
```

| PCA, shape = known group | PCA, colour = k-means |
| --- | --- |
| ![PCA of fifty features shaped by the planted group](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-cluster-pca-group.png) | ![PCA of fifty features coloured by k-means clusters](https://raw.githubusercontent.com/hiows/STATassist-py/master/docs/figures/README-cluster-pca-cluster.png) |

Passing both at once puts the clustering in the colours and the known grouping in the markers, which is the read the plot exists for. `cluster_hclust()`, `cluster_dbscan()` and `cluster_snn()` return the same object and go in the same place. Noise from the two density methods is drawn grey rather than given a palette colour, because a point left out is the absence of a cluster rather than a cluster of its own.

---

## Result contracts

Every result is a `Mapping` whose keys are also attributes. `to_dict()` is a shallow copy of the slots, and the engine handle is deliberately outside it: `model.fit` works, `model["fit"]` raises, and nothing that has to survive being written out as JSON carries a fitted estimator.

| Class | Produced by | Axis | Stable slots |
| --- | --- | --- | --- |
| `SaSimulation` | the six `simulate_*` | varies | `args`, `truth`, and the extra `truth_*` tables the scenario can score |
| `SaSplit` | `split_data()` | rows | `full_data`, `datasets`, `train_idx`, `design`, `parameters`, `metadata` |
| `SaComparison` | `compare_two_groups()`, `compare_multiple_groups()`, `compare_one_sample()`, `compare_factorial_groups()` | `features` | `analysis`, `features`, `design`, `parameters`, `effect`, `tests`, `test_info`, `diagnostics`, `metadata` |
| `SaCategorical` | `compare_categorical_groups()` | cells | `analysis`, `variables`, `design`, `parameters`, `cells`, `tests`, `test_info`, `association`, `diagnostics`, `metadata` |
| `SaSignificance` | `estimate_significance()` | `features` | `analysis_type`, `significance` |
| `SaCategoricalSignificance` | `estimate_categorical_significance()` | cells or table | `analysis_type`, `significance` |
| `SaDiagnosis` | `diagnose_distribution()` | `features` | `analysis`, `features`, `design`, `parameters`, `normality`, `variance`, `outliers`, `summary`, `metadata` |
| `SaModel` | the five `fit_*` | `terms` | `analysis`, `terms`, `design`, `parameters`, `coefficients`, `fit_stats`, `performance`, `resampling`, `engine`, `metadata` |
| `SaSelection` | `perform_rfe()`, `perform_stepwise()` | `candidates` | `analysis`, `candidates`, `design`, `parameters`, `selected`, `ranking`, `profile`, `resampling`, `engine`, `metadata` |
| `SaPerformance` | `evaluate_regression_models()`, `evaluate_classification_models()` | `models` | `analysis`, `models`, `design`, `parameters`, `predictions`, `metrics`, `metadata` |
| `SaReduction` | `perform_pca()`, `perform_tsne()`, `perform_umap()` | `points` | `analysis`, `points`, `design`, `parameters`, `scores`, `engine`, `metadata` |
| `SaCluster` | the four `cluster_*` | `points` | `analysis`, `points`, `design`, `parameters`, `assignments`, `clusters`, `engine`, `metadata` |

A slot that only some scenarios can fill is **absent rather than empty**, so `in` is the way to ask for it. This matters because "there was nothing to report" and "there was something and it is missing" are different facts:

```python
"posthoc" in multi          # a post-hoc stage ran
"pairwise" in multi         # the same numbers one contrast at a time
"terms" in fact             # a factorial model, one row per feature and term
"cells" in fact             # the cell means of a crossed design
"variance" in pca           # the reduction was a PCA, so `loadings` is there too
"comparisons" in eval_cls   # the evaluation had a baseline to compare against
"curves" in eval_cls        # the evaluation was of a two-class outcome
```

`SaModel`, `SaSelection`, `SaReduction` and `SaCluster` each carry `fit`, the engine object, as a property beside their slots. `SaCategorical` carries `as_table()`, and `SaModel` carries `coef()` and `predict()`, which are the three places where a method rather than a slot is the right shape.

## Main functions

| Function | Purpose |
| --- | --- |
| `compare_two_groups()` | Welch / Wilcoxon / robust tests plus fold change for two groups |
| `compare_multiple_groups()` | Four omnibus tests for three or more groups, each with its matching post-hoc stage; independent or repeated |
| `compare_factorial_groups()` | One factorial ANOVA for crossed factors, with an answer per model term and Tukey contrasts on the marginal means and inside each stratum |
| `compare_categorical_groups()` | Chi-square beside Fisher on two categorical variables, or McNemar / Cochran's Q on repeated binary conditions |
| `compare_one_sample()` | One-sample t, signed-rank and proportion tests against a hypothesised value |
| `diagnose_distribution()` | Normality, homogeneity of variance and outliers for a set of features |
| `screen_outliers()` | Flag observations by IQR fences, robust z or Grubbs, without removing them |
| `estimate_significance()` | Filter features by log2FC and p-value from any comparison result, over the omnibus test, one pairwise contrast at a time, or one model term at a time |
| `estimate_categorical_significance()` | The same verdict for a contingency table, read one cell at a time or once for the table |
| `summarize_descriptive_stats()` | Feature-wise, and optionally group-wise, descriptive table |
| `summarize_association_stats()` | Pearson, Spearman and Kendall on every pair of features, each as a square matrix beside its p-values |
| `center_by_control()` | Remove the control group's centre from every feature |
| `draw_forest_plot()` | Forest plot of estimates, of pairwise contrasts, or of p-values |
| `draw_volcano_plot()` | Volcano plot from `estimate_significance()` output, or one panel per model term |
| `draw_grouped_boxplot()` | Boxplots for several features x group levels, or one panel per feature for a crossed design |
| `draw_grouped_barplot()` | One column of `summarize_descriptive_stats()` as clusters of bars, with an error bar read under `mainbar` |
| `draw_butterfly_hist()` | Back-to-back histogram, kernel density, or both, for exactly two groups |
| `draw_heatmap()` | Clustered heatmap of features x samples, with the sample groups annotated |
| `draw_corrplot()` | The correlation matrix as a heatmap, with one clustering shared by both axes |
| `draw_interaction_plot()` | Cell means of a crossed design joined across one factor, one line per level of another |
| `draw_mosaic_plot()` | Mosaic of a contingency table, shaded by the residual of the null it was tested against |
| `split_data()` | Train/test partition, stratified and leakage-aware through `id` |
| `fit_linear_regression()` | Linear model with coefficient inference and resampled performance |
| `fit_logistic_regression()` | Two-class logistic model with odds ratios and their intervals |
| `fit_elastic_net()` | LASSO, ridge or elastic net for either outcome type, with the selected terms |
| `fit_rf()` | Random forest with permutation and impurity importance and out-of-bag fit |
| `fit_svm()` | Radial-kernel support vector machine with permutation importance |
| `evaluate_regression_models()` | Score one or more fitted regressions on the same held-out rows, each against a baseline |
| `evaluate_classification_models()` | The same for a two-class outcome, with DeLong's test, the IDI and the NRI |
| `draw_prediction_plot()` | Observed against predicted, with the identity line and the calibration line the metrics table holds |
| `draw_roc_curve()` | ROC curves of the scored classifications, always overlaid |
| `perform_rfe()` | Recursive feature elimination inside the resampling, returning the predictors it kept, their ranking and the score at every subset size |
| `perform_stepwise()` | Stepwise search by AIC or BIC, returning the predictors it kept, what each is worth to the criterion, and the path it walked |
| `perform_pca()` | Principal components of the samples or of the features, with loadings and variance |
| `perform_tsne()` | t-SNE embedding of either margin |
| `perform_umap()` | UMAP embedding of either margin; needs the `umap` extra |
| `cluster_hclust()` | Hierarchical clustering of either margin, returning the tree as well as the cut |
| `cluster_kmeans()` | k-means clustering of either margin, best of 25 starts |
| `cluster_dbscan()` | Density-based clustering, deriving the number of clusters and leaving sparse points as noise |
| `cluster_snn()` | Shared nearest neighbour clustering, which groups by how many neighbours two points have in common |
| `draw_dim_reduction_plot()` | Two coordinates of a reduction as a scatter, coloured by a clustering and shaped by a known grouping |
| `simulate_two_groups()` | Two-group log2 expression data with the planted answer returned alongside it |
| `simulate_multiple_groups()` | One control and any number of treatment groups, scored per feature, per level and per contrast |
| `simulate_factorial_groups()` | Crossed factors, scored per model term as well as per cell and per contrast |
| `simulate_categorical_groups()` | A contingency table, or repeated binary conditions, with the planted association returned cell by cell |
| `simulate_regression()` | Continuous outcome from planted coefficients, with a correlation structure and the truth per predictor and per term |
| `simulate_classification()` | Two-class outcome at a chosen event rate from the same design |
| `make_block_cor()` | Block correlation matrix for the simulators, with `against` for the predictors of a block that move the other way |

## Differences from the R package

The port keeps the workflow and the result contracts. Three things read differently, and are worth knowing before translating a script across:

- **There is no `plot()` generic.** R dispatches `plot()` on a result to whichever `draw_*` fits it. Python names the function: `draw_forest_plot(comp_res)`, `draw_mosaic_plot(cat)`, `draw_dim_reduction_plot(pca)`. The `draw_*` functions render onto the current `matplotlib` figure and hand back the table they drew, so `matplotlib.pyplot.gcf().savefig(...)` is how a figure is written out.
- **`coef()` and `predict()` are methods.** R's `coef(model)` and `predict(model, newdata = )` are `model.coef()` and `model.predict(newdata=)`, and `as.table()` on a contingency result is `cat.as_table()`.
- **Repeated measures inside a factorial design are not implemented.** `compare_factorial_groups(within=...)` raises rather than guessing. Repeated conditions in a one-factor design are supported, through `compare_multiple_groups(paired=True)`.

Every figure and every printed block in this README is produced by [`tools/make_readme_figures.py`](tools/make_readme_figures.py), which runs the whole document end to end on `seed=2026` and writes both the images and [`docs/readme_outputs.txt`](docs/readme_outputs.txt). Nothing here was transcribed from the R package.

## Author

**Wonseok Oh** ([ORCID: 0009-0002-0687-8466](https://orcid.org/0009-0002-0687-8466))

## License

MIT. See [LICENSE](LICENSE).
