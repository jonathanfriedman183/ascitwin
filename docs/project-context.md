# Project Context

    project_name       : ascitwin                                  [HIGH]
    domain             : computational oncology - physics-informed  [HIGH]
                         inverse modelling of the CRC tumour
                         microenvironment for ICI response
    framework          : PyTorch 2.x (hand-written PINN)            [HIGH]
    primary_datasets   : 6 RNA-seq cohorts + 1 spatial pilot        [HIGH]
    compute_target     : 1x NVIDIA A100 80GB, 4.2 GPU-h train       [HIGH]
    hparams_reference  : Methods V-A/V-B + Tables III, V, VI        [HIGH]
    supp_path          : none (appendix tables inline in main PDF)
    extra_signals      : 5 numbered PDEs, constraint-inversion Eq7,
                         composite loss Eq6, learnable CIAI target,
                         ablation Table II, no released checkpoints

    NEEDS_USER_DECISION: 0
    Status             : resolved; build executed Turn 0 through Turn 6.

## 1. project_name
`ascitwin` — the Aggregate-to-Spatiotemporal Constraint Inversion framework that drives the
patient-specific digital twin. Confirmed with the user. snake_case single token, distinct from
every prior package name in this code collection.

## 2. supp_path
No separate supplementary file ships with the manuscript. The appendix Tables V and VI are
typeset inside the main PDF (pages 10-11). Derived parameter regimes live in Tables III, V, VI.

## 3. domain
Computational oncology. Specifically: physics-informed inverse modelling of the colorectal
cancer (CRC) tumour microenvironment, recovering patient-specific reaction-diffusion-pharmacokinetic
dynamics from bulk transcriptomics to predict immune-checkpoint-inhibitor (ICI) response.
Source: Abstract; Introduction; Methods V-A.

## 4. framework
PyTorch 2.x with a hand-written physics-informed neural network. Spatial and temporal
derivatives of the network output are taken with `torch.autograd.grad`; the five PDE residuals
are assembled in `governing.py`. The manuscript's Code Availability line names a
"PyTorch/DeepXDE implementation"; DeepXDE is replaced by an explicit torch implementation for
transparency and to keep the dependency surface small (logged in deviations.md, D1).
Source: Code Availability; Methods V-B.

## 6. primary_datasets
| name | role | N | platform | license / access |
| --- | --- | --- | --- | --- |
| TCGA-COAD/READ | training | 623 (456+167) | RNA-seq (FPKM, log2) | open, GDC Data Portal |
| GSE39582 | external | 566 | Affymetrix U133 Plus 2.0 | open, GEO |
| GSE17536 | external | 177 | Affymetrix GPL570 | open, GEO |
| GSE236581 | external ICI | 22 | RNA-seq + RECIST | open, GEO |
| IMvigor210 | external cross-cancer ICI | 298 | RNA-seq (atezolizumab, urothelial) | CC BY 3.0, R package |
| CPTAC-CRC | external multi-omics | 290 | proteomics + sequencing | open, NCI Proteomic Data Commons |
| Visium HD CRC | spatial pilot | 5 FFPE | 10x Genomics, 2x2 um | open, 10x Genomics |

Totals: 1,976 patients across six RNA-seq cohorts plus five FFPE spatial samples. Immune
fractions are estimated with CIBERSORTx (22 cell types, LM22) and MCP-counter. CMS labels come
from CMScaller. Training uses an 80/20 split with stratified 5-fold cross-validation.
Source: Methods V-E; Data Availability.

## 7. compute_target
Single NVIDIA A100 80GB. Training over the full TCGA-COAD/READ cohort takes 4.2 GPU-hours.
Per-patient inference (PDE parameter estimation + spatial field recovery + response
classification) takes about 45 seconds and uses 8.3 GB of VRAM. A standard numerical ODE solver
on the same five-equation system takes about 180 seconds per patient (about four times slower).
Source: Methods E (Computational Efficiency).

## 8. hparams_reference
- Architecture and optimisation: Methods V-B. Backbone 4 fully-connected hidden layers x 512
  nodes, tanh, dropout 0.2, 2.4M parameters. Adam, initial lr 1e-3, 8,000 epochs, cosine
  annealing. Collocation points N_c = 10,000. Composite loss Eq6 with gradient-norm balancing.
- Parameter ranges: Methods V-A. D_T in [1e-4, 1e-3] cm^2 day^-1; rho in [0.01, 0.1] day^-1;
  D_C in [1e-3, 1e-1] cm^2 day^-1.
- CMS-stratified population medians: Table III (D_I, delta_TI, gamma, k_up by CMS1-4 / MSI-H / MSS).
- Profile-likelihood 95% CIs for all 15 parameters: Table V (12 identifiable; k_el, p_basal, K
  unbounded / least identifiable).
- Consolidated supplementary settings (architecture sweep, loss-weight sensitivity, training-set
  scaling, ensemble UQ): Table VI.

## 9. extra_signals
- Five numbered governing equations (Eq1-5) plus the constraint-inversion identity (Eq7) and the
  composite loss (Eq6).
- The Composite Immune Activation Index (CIAI): a learnable surrogate target combining four
  established ICI correlates (MSI-H vs MSS; TMB dichotomised at 10 mut/Mb; CD8+ infiltration from
  CIBERSORTx; 18-gene T-cell inflamed GEP). A patient is labelled a predicted responder when at
  least two of the four criteria are satisfied.
- Ablation study over the five PDE modules (Table II), with the immune-dynamics and PD-L1/PD-1
  equations the two largest single-component contributors and a super-additive immune x cytokine
  interaction (IR = 1.35).
- Parameter identifiability is assessed with a Fisher Information Matrix condition number and
  profile-likelihood intervals (Methods V-D, Table V).
- No released checkpoints; the manuscript states code goes open with a permanent DOI on
  publication. The verbatim Code Availability sentence is preserved in `implementation-map.md`
  for provenance and is intentionally kept out of the README.
- Ethics: only de-identified public data; authors declare no competing interests; funding and
  contributions withheld for double-anonymous review.
