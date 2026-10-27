# ascitwin

A field lexicon for a physics-informed digital twin of the colorectal tumour microenvironment.
The model inverts a five-equation reaction-diffusion-pharmacokinetic system from bulk RNA-seq to
recover patient-specific spatial dynamics and to predict immune-checkpoint-inhibitor response.

Entries are alphabetical. Each headword gives a one-line gloss; where a term names something you
can run, the command and the value the manuscript reports sit underneath. Commands in their full
form assume the A100 compute profile (see **Compute**); swap `experiment=main` for
`experiment=_smoke` to run the same path at unit-test scale on a laptop CPU.

---

### Ablation
Removal of one governing module to measure its contribution. Each `ablation_*` experiment sets the
matching residual weight to zero.

    ascitwin evaluate experiment=ablation_immune

Reported (Table II): removing the immune-dynamics equation moves AUC by -0.055, the largest single
drop; the PD-L1/PD-1 equation is second at -0.042.

### Acquisition
The data layer: a registry of the seven cohorts, non-negative deconvolution into 22 immune
fractions plus three pathway scores, CMS assignment, a deterministic synthetic cohort, and the
Visium HD spatial reference. Real bulk data is read through a JSON manifest.

### CIAI
The Composite Immune Activation Index, the learnable response target. Four established correlates
vote: MSI-H status, tumour mutational burden above 10 mut/Mb, CD8+ infiltration, and the 18-gene
T-cell inflamed score. A patient is a predicted responder when at least two of the four agree.

### Cohorts (Data)
| cohort | role | N | platform | access |
| --- | --- | --- | --- | --- |
| TCGA-COAD/READ | training | 623 | RNA-seq | GDC Data Portal |
| GSE39582 | external | 566 | Affymetrix U133 Plus 2.0 | GEO |
| GSE17536 | external | 177 | Affymetrix GPL570 | GEO |
| GSE236581 | external ICI | 22 | RNA-seq + RECIST | GEO |
| IMvigor210 | external cross-cancer | 298 | RNA-seq (atezolizumab) | R package, CC BY 3.0 |
| CPTAC-CRC | external multi-omics | 290 | proteomics + sequencing | NCI Proteomic Data Commons |
| Visium HD CRC | spatial pilot | 5 FFPE | 10x Genomics, 2x2 um | 10x Genomics |

To point the twin at real data instead of the synthetic cohort, supply a manifest of records with
`fractions`, `pathways`, `msi_high`, `tmb_per_mb`, `cd8_fraction`, `gep_score`, and `treated`.

### Compute
| resource | value |
| --- | --- |
| accelerator | 1x NVIDIA A100 80GB |
| training wall-clock | 4.2 GPU-hours over TCGA-COAD/READ |
| per-patient inversion | about 45 seconds |
| memory | 8.3 GB per patient |
| numerical-solver reference | about 180 seconds per patient |

### Console
The command surface, built on plac:

    ascitwin train experiment=main
    ascitwin evaluate experiment=main
    ascitwin invert experiment=main
    ascitwin spatial experiment=main
    ascitwin export experiment=main

### Equations (Governing)
Five coupled fields on a square domain with zero-flux Neumann boundaries: tumour growth
(Fisher-KPP with a Kuznetsov kill term), CD8+ immune dynamics, cytokine reaction-diffusion,
one-compartment drug pharmacokinetics, and PD-L1 surface dynamics. The residuals live in
`ascitwin/governing.py`.

### Ethics
Only de-identified, publicly available data is used. The synthetic cohort that ships with the
package carries no patient information.

### Evaluation
| endpoint | command | reported value |
| --- | --- | --- |
| internal response AUC | `ascitwin evaluate experiment=main` | 0.893 +/- 0.014 |
| cross-cohort AUC | `ascitwin evaluate experiment=main data.cohort=GSE39582` | 0.868 +/- 0.016 |
| immune-equation ablation | `ascitwin evaluate experiment=ablation_immune` | delta-AUC -0.055 |
| spatial concordance | `ascitwin spatial experiment=main` | Pearson r 0.78 |
| recovered parameters | `ascitwin invert experiment=main` | 15 per-patient values |

### Identifiability
Fisher Information condition number and profile-likelihood intervals. Twelve of fifteen parameters
have bounded 95% intervals; the carrying capacity, elimination rate, and basal PD-L1 production are
the least constrained.

### Installation
pip:

    pip install -e .

conda:

    conda env create -f environment.yml
    conda activate ascitwin

Docker:

    docker build -t ascitwin .
    docker run --rm ascitwin evaluate experiment=_smoke

### Inversion
The physics-informed network: four hidden layers of 512 units with tanh activations and 0.2
dropout, about 2.4M parameters, trained with Adam at 1e-3 for 8,000 epochs under cosine annealing.
The composite loss balances data fidelity, PDE residuals, initial and boundary terms by their
gradient norms. Spatial integrals are tied to bulk fractions through the constraint identity.

### Parameters
The fifteen recovered PDE parameters per patient, with bounded ranges drawn from the manuscript's
profile-likelihood table. CMS-stratified medians separate the immune-hot CMS1 regime from the
stromal-excluded CMS4 regime.

### Quickstart

    pip install -e .
    ascitwin evaluate experiment=_smoke

### Spatial
Recovered immune fields are compared against Visium HD cell-type density on a 50x50 grid. The
manuscript reports a Pearson correlation of 0.78, holding at 0.61 after the radial mean is removed.

### Tests
Varied checks span shape, physics invariants, manufactured-solution residuals, single-batch
overfitting, gradient flow, metric agreement against reference libraries, identifiability, config
integrity, and an end-to-end training step.

    make test
