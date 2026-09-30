# Personal history versus matched external data in motor-imagery decoding

![Figure 1: study design](figures/figure1_overview.png)

**Question.** How much does a person's earlier-session EEG archive add beyond an equal amount of data from other people, and does
that advantage last when the shared source database grows?

**Design (Figure 1).**
- A common source base fixes the tangent reference and scaler; logistic-regression C is selected using source people only.
- The base then receives either 60 trials from the target person's first session, or 60 trials from each of three random
  external donors. Logistic regression is fitted separately for each addition, with the same base, representation and classifier settings.
- Evaluation uses only the person's later session, with no fitting or tuning on it.
- The paired effect is the balanced accuracy with the person's own data minus the mean over the three donors.

**Results** (from the frozen result files):
- **Small source base (100 trials):** the mean advantage of personal archives was 3.8, 8.5 and 4.6 percentage points in three
  cross-day motor-imagery datasets.
- **Separately held 24-person cohort:** a 3.0-point advantage (primary 95% Student-t interval 1.1 to 4.9), at 61.9% versus
  58.8% absolute accuracy.
- **Larger common bases (1,620 and 972 trials):** the external additions were reserved first. Mean advantages were then 2.0
  and 1.4 points in the two development datasets, and 0.9 points in the reused 24-person cohort (95% interval -0.3 to 2.0).
  In that cohort the advantage shrank by 2.1 points (95% interval 0.8 to 3.5).

## Layout
| Path | Contents |
|---|---|
| `src/expose/` | Dataset loaders (MOABB BNCI 2014-001, BNCI 2014-004, Lee 2019 / OpenBMI), baselines, membership and operation construction, validation |
| `scripts/` | Download, preparation, run, analysis, verification and plotting scripts, by study stage |
| `research/` | Frozen study configurations with SHA-256 files, participant roles and the confirmation source packet |
| `tests/` | Unit and contract tests |
| `reproduction/` | Download-free OpenBMI confirmation replay, large-source aggregate results, exact figure inputs and data provenance |

## Reproduce

For the central 24-person result, no raw EEG or fitting libraries are needed:

```bash
python -m venv .venv && . .venv/bin/activate
python -m pip install -r reproduction/requirements.txt
make -C reproduction confirmation
```

This reconstructs all 144,000 saved confirmation predictions and the +3.0278 pp result. See [the reproduction guide](reproduction/README.md) for scope, expected values, figure rendering and the separate local large-source replay. BCI trial-level inputs are excluded from the public release; its aggregate source-size results are included.

For the preserved fitting code and its unit/contract tests:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q            # 372 tests, CPU only, no EEG download needed
```
The full pipeline downloads the public datasets through MOABB. Lee 2019 is GigaScience DB 10.5524/100542, about 25 GB.
The historical scripts cover download, preparation, fitting and analysis, but their frozen configurations and prerequisite artifacts must be supplied in the documented stage-specific contracts. A clean, complete raw-to-model grid rebuild has not been verified from this export. Raw EEG and prepared signal/covariance arrays are not in this repository.

## Limitations
- The effect sizes hold for the specified pipeline only: covariance, tangent space and logistic regression, with 60-trial
  additions and three donors.
- At larger source bases the advantage becomes smaller and its interval includes zero in the reused cohort.
- Two tests are deselected in `pytest.ini`: one needs the private adoption record, and the other needs the original prepared-source index. The scientific source-only packet and completed participant roles are included. Historical configuration statuses describe their creation time, not a currently unopened cohort.

## License
Project-owned software and documentation: MIT, as recorded in [LICENSE](LICENSE). Third-party data retain their own terms; see [data provenance and redistribution scope](reproduction/DATA_AND_LICENSES.md).

Research authors: Woncheol Jeong and Hayoung Oh, Sungkyunkwan University.
