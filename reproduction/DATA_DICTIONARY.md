# Saved-output schema

- `results/confirmation/scores.csv`: one row per operation; balanced accuracy is on the 0–1 scale. Six conditions per participant/draw include base and pooled controls in addition to own and three single-donor conditions.
- `results/confirmation/trial_predictions.csv.gz`: evaluation trial identity, binary truth/predicted class and class probabilities for each operation. The script validates the exact column schema.
- `results/confirmation/person_effects.csv`: ten-draw participant means. `own_mean_ba` and `mean_single_ba` use 0–1; `effect_pp` uses percentage points.
- `results/confirmation/summary.json`: primary effect statistics in percentage points, frozen classification, and descriptive optional contrasts. The primary confidence interval is Student-t; bootstrap and sign-flip are secondary.
- `results/large_source/participant_effects.csv`: one row per cohort/participant, averaged over ten draws; accuracies **and differences** use the 0–1 BA scale. Multiply differences by 100 for percentage points.
- `results/large_source/summary.json`: the same 0–1 units; descriptive t and percentile-bootstrap intervals, signs, means and medians for each contrast and cohort.
- `results/source_context_values.json`: participant plot values and summary statistics, also on the 0–1 scale. The renderer alone converts to percent/percentage points.

The key `openbmi_confirmation` in the large-source records names the reused cohort's origin; it does not designate the large-source analysis as confirmatory. Participants are the resampling units after within-person draw averaging. Shared source data and overlapping BNCI folds limit generalization beyond the specified source corpus and construction.
