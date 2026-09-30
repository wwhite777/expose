# Data, credit and redistribution scope

Research authors: Woncheol Jeong and Hayoung Oh, Sungkyunkwan University. The existing repository MIT license governs project-owned software and documentation; it does not relicense third-party datasets, software or publications.

## OpenBMI

The GigaDB record for Lee et al.'s OpenBMI dataset, DOI [10.5524/100542](https://doi.org/10.5524/100542), identifies CC0-1.0 in the [provider's DataCite metadata](https://api.datacite.org/dois/10.5524/100542), checked 2026-09-30. The released confirmation predictions use dataset participant/trial identifiers, binary task labels and model outputs; they contain no raw EEG or real participant names.

Cite the dataset and [Lee et al., “EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy,” GigaScience, 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6501944/). The OpenBMI toolbox's GPL terms are separate from the dataset's CC0 record. This repository does not bundle the toolbox.

## BCI Competition IV / BNCI

The [BCI Competition IV provider terms](https://www.bbci.de/competition/iv/) and [download page](https://www.bbci.de/competition/iv/download/) request credit to the recording group, citation of the dataset publication and reporting of resulting publications. An explicit redistribution license for the 2a/2b trial data was not established in the 2026-09-30 review.

Accordingly, this repository distributes derived participant-level effects and summary statistics, but excludes BCI raw EEG, source trial labels, trial-level membership maps and prediction rows carrying those labels/identifiers. The full local replay's inputs stay outside Git. Obtain datasets from the provider under its terms; publication of aggregate results does not grant permission to redistribute its trial data.

BCI Competition IV 2a is BNCI2014-001; 2b is BNCI2014-004. Their distinct recording/session/channel structures must remain explicit in any new analysis. Dataset acquisition is separate from the no-download confirmation replay.

## Scope of this release

There is no raw-to-model full-grid reproduction claim, clinical-performance claim, prospective public-preregistration claim, DOI deposit claim or journal-acceptance claim. The 24-person local confirmation was completed before the post-hoc large-source analysis; that cohort is already exposed. Private manuscripts, reviewer records, instructions and original workspace history are not included in the public export.
