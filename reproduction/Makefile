PYTHON ?= python
OUT ?= reproduced/run1
PRIVATE_INPUTS ?= ../private/large_source

.PHONY: confirmation figure large-source

confirmation:
	$(PYTHON) ../scripts/reproduce_final_confirmation.py --predictions results/confirmation/trial_predictions.csv.gz --scores results/confirmation/scores.csv --participant-effects results/confirmation/person_effects.csv --summary results/confirmation/summary.json --out $(OUT)/confirmation.json

figure:
	$(PYTHON) ../scripts/plot_source_context.py --data results/source_context_values.json --out $(OUT)/figure2

# This target requires the separately licensed local BCI trial-level inputs.
large-source:
	$(PYTHON) ../scripts/reproduce_revision_v10_large_source.py --predictions $(PRIVATE_INPUTS)/predictions.csv.gz --operations $(PRIVATE_INPUTS)/plan/operations.json.gz --memberships $(PRIVATE_INPUTS)/plan/membership_sets.json.gz --small-openbmi $(PRIVATE_INPUTS)/small/openbmi_operation_scores.csv --small-bnci $(PRIVATE_INPUTS)/small/bnci_operation_scores.csv --small-confirmation results/confirmation/scores.csv --summary results/large_source/summary.json --participant-effects results/large_source/participant_effects.csv --out $(OUT)/large_source.json
