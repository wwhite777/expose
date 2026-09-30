# Reproduce the central saved-output results

This layer reconstructs the 24-person OpenBMI confirmation from saved predictions and supplies the aggregate results for the later source-size comparison. It does not train models or authenticate the historical freeze. The original fitting code remains in `../src` and `../scripts`.

## Confirmation: complete with this repository

From the repository root, using Python 3.10 (tested with 3.10.12):

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r reproduction/requirements.txt
make -C reproduction confirmation
```

Only NumPy and SciPy are needed. No EEG download, GPU, account or model weights are needed. The command checks 144,000 prediction rows, reconstructs 1,440 operation scores and averages ten draws within each of 24 participants. It checks trial identity, class balance, probability validity, paired C, saved participant effects and the primary summary.

Expected own-minus-mean-single-donor effect: **3.0277778 percentage points**, primary Student-t 95% interval **[1.1490478, 4.9065078]**, 18 positive and six negative participant effects. Mean own and donor accuracies are 61.85% and 58.82222%. The comparison averages three separately fitted donor models' accuracies; it does not ensemble predictions. Source base S=100, added block h=60, final fit count N=160.

The output is `reproduction/reproduced/run1/confirmation.json`. Replays refuse to overwrite an existing result. For another run, use `make -C reproduction confirmation OUT=reproduced/run2`. To use an existing Python environment, pass `PYTHON=/path/to/python`.

## Large-source comparison: public aggregates, local exact replay

`results/large_source/` contains participant-level effects and summary statistics. These are post-hoc results, including reuse of the 24 people; they are not a second held-out confirmation.

| Cohort | n | Matched large source S | Own-minus-donor mean (pp) | Descriptive t95 interval (pp) |
|---|---:|---:|---:|---:|
| OpenBMI development | 12 | 1,620 | 2.0139 | [0.6422, 3.3856] |
| BNCI2014-001 | 9 | 972 | 1.3683 | [-1.3245, 4.0611] |
| OpenBMI reused cohort | 24 | 1,620 | 0.8806 | [-0.2654, 2.0265] |

The large bases reserve three disjoint 60-trial donor blocks. C is retained from the small-source construction; representations are refitted across source sizes and fixed within each paired addition comparison. In the reused cohort, the paired large-minus-small effect is -2.1472 pp, t95 [-3.4918, -0.8026]. A zero-crossing interval is not equivalence, and this is not a retuned large-source benchmark.

The exact large-source replay additionally needs BCI Competition trial labels and membership records. Those inputs are **not distributed in this public repository** because an explicit redistribution license was not established. On the original research machine the ignored `private/large_source` link supplies the preserved inputs:

```bash
make -C reproduction large-source
```

For a separately authorized copy, use `PRIVATE_INPUTS=/absolute/path/to/large_source`. Required files are `predictions.csv.gz`, `plan/operations.json.gz`, `plan/membership_sets.json.gz`, `small/openbmi_operation_scores.csv` and `small/bnci_operation_scores.csv`. The command reconstructs 342,720 rows and 3,150 operation scores; checks the donor/base membership relations and retained C; and compares all 45 participant vectors and saved summary values. It does not reconstruct raw preprocessing, representation fitting or C selection.

## Redraw the source-size figure

```bash
python -m pip install -r reproduction/requirements-figures.txt
make -C reproduction figure
```

This uses the released participant values and saved means/Student-t intervals. It writes Figure 2 in PDF, SVG and PNG plus a checksum receipt. The absolute-accuracy panel starts at 40% and uses dots, not truncated bars. Figure 1's editable SVG/PPTX and embedded-font PDF are in `../figures/`.

## File integrity and provenance

`SOURCE_MANIFEST.json` records source and distributed SHA-256 hashes for this added scientific payload, including the two explicitly documented metadata-only transformations. It does not replace the historical frozen hashes. `CHECKSUMS.sha256` covers this reproduction layer and the referenced scripts/figures; run `sha256sum --check reproduction/CHECKSUMS.sha256` from the repository root.

The confirmation analysis is the completed r002 analysis; the large-source analysis is r001, used in the final manuscript's v10–v13 scientific content. Read [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md) before redistributing data. Saved-output agreement is a computational consistency check, not new independent scientific validation.
