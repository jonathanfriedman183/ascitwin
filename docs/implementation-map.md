# Implementation Map

All paper provenance for `ascitwin` is recorded here. Source files carry no comments and no
docstrings by design; this table is the single point that ties each module back to the
manuscript. The seismic-tomography module names are noted in parentheses.

## Governing equations and constraints

| paper location | equation / object | file | symbol | notes |
| --- | --- | --- | --- | --- |
| Methods V-A, Eq. (1) | tumour growth (Fisher-KPP + Kuznetsov kill) | governing.py | `tumour_residual` | dT/dt = D_T lap(T) + rho T (1 - T/K) - delta_TI I T |
| Methods V-A, Eq. (2) | CD8+ immune dynamics | governing.py | `immune_residual` | dI/dt = D_I lap(I) + s_I + alpha T/(beta+T) I - mu_I I - gamma P I |
| Methods V-A, Eq. (3) | cytokine reaction-diffusion | governing.py | `cytokine_residual` | dC/dt = D_C lap(C) + kappa_T T + kappa_I I - lambda_C C |
| Methods V-A, Eq. (4) | one-compartment drug PK | governing.py | `drug_residual` | dA/dt = -k_el A + f_dose(t) |
| Methods V-A, Eq. (5) | PD-L1 surface dynamics | governing.py | `pdl1_residual` | dP/dt = p_basal - k_deg P + k_up T/(K_P+T) - k_block A P |
| Methods V-A | zero-flux Neumann boundary | governing.py | `neumann_flux` | du/dn = 0 on boundary of Omega for all diffusing species |
| Methods V-A | domain Omega subset R^2 x [0, t_f] | collocation.py | `DomainSampler` | interior collocation + boundary + initial samples |
| Methods V-C, Eq. (7) | CIBERSORT volume-average identity | inversion.py | `fraction_constraint` | f_i = integral u_i / sum_j integral u_j at t_biopsy |
| Methods V-B, Eq. (6) | composite loss | inversion.py | `CompositeLoss` | w_data L_data + w_pde L_pde + w_ic L_ic + w_bc L_bc |
| Methods V-B | gradient-norm loss balancing | inversion.py | `GradNormBalancer` | adaptive w_* from per-term gradient norms |
| Methods V-B | PINN backbone phi_theta(x,t) | inversion.py | `FieldNetwork` | 4 x 512 tanh, dropout 0.2, 2.4M params, outputs T,I,C,A,P |
| Methods V-B | per-patient PDE parameters theta_PDE | medium.py | `PdeParameters` | 15 parameters with bounded ranges (Table V) |

## Surrogate target and observables

| paper location | object | file | symbol | notes |
| --- | --- | --- | --- | --- |
| Methods V-E (CIAI) | Composite Immune Activation Index | surrogate.py | `composite_immune_activation_index` | 4 ICI correlates, responder if >= 2 of 4 satisfied |
| Methods V-E | MSI-H vs MSS criterion | surrogate.py | `msi_criterion` | one of four CIAI components |
| Methods V-E | TMB dichotomised at 10 mut/Mb | surrogate.py | `tmb_criterion` | one of four CIAI components |
| Methods V-E | CD8+ infiltration (CIBERSORTx) | surrogate.py | `cd8_criterion` | one of four CIAI components |
| Methods V-E | 18-gene T-cell inflamed GEP | surrogate.py | `gep_criterion` | one of four CIAI components |
| Methods V-C | K ~= 25 aggregate observables | gathers.py | `AggregateObservation` | 22 CIBERSORTx fractions + 3 pathway-activity scores |
| Methods V-E | CIBERSORTx LM22 + MCP-counter | acquisition.py | `deconvolve_fractions` | immune fraction estimation on bulk expression |
| Methods V-E | spatial Visium HD ground truth | acquisition.py | `load_spatial_reference` | 50 x 50 grid, 100 x 100 um^2 cells, per cell type |

## Datasets and preprocessing

| paper location | item | file | symbol | notes |
| --- | --- | --- | --- | --- |
| Methods V-E; Data Availability | seven cohorts | acquisition.py | `CohortRegistry` | TCGA-COAD/READ, GSE39582, GSE17536, GSE236581, IMvigor210, CPTAC-CRC, Visium HD |
| Methods V-E | 80/20 split, stratified 5-fold CV | acquisition.py | `stratified_folds` | training partition for TCGA-COAD/READ |
| Methods V-E | CMS labels (CMScaller) | acquisition.py | `assign_cms` | CMS1-4 plus Unclassified |
| deviation D2 | deterministic synthetic cohort | acquisition.py | `SyntheticCohort` | recoverable spatial signal when real bulk data is unavailable |

## Tables and reported numbers

| paper location | content | file | symbol | notes |
| --- | --- | --- | --- | --- |
| Table I | response prediction across cohorts | interpretation.py | `response_auc_table` | PINN-DT 0.893 internal, 0.880 mean; baselines IRRS 0.851 etc. |
| Table II | PDE-component ablation | interpretation.py | `ablation_study` | per-module delta-AUC, immune x cytokine IR = 1.35 |
| Table III | CMS-stratified PDE parameters | interpretation.py | `cms_parameter_anova` | per-subtype AUC, one-way ANOVA + Tukey, 12/15 significant |
| Table IV | cross-cohort generalisation | interpretation.py | `cross_cohort_degradation` | degradation = 1 - AUC_ext/AUC_tcga, mean 3.3% |
| Table V | profile-likelihood 95% CIs | identifiability.py | `profile_likelihood_intervals` | 12 identifiable; k_el, p_basal, K unbounded |
| Methods V-D | FIM condition number kappa | identifiability.py | `fisher_condition_number` | median kappa ~ 3.8e3 for TCGA-COAD/READ |
| Table VI | consolidated supplementary experiments | interpretation.py | `supplementary_panels` | pairwise interactions, per-cell spatial r, ensemble UQ, scaling |

## Metrics and statistics

| paper location | item | file | symbol | notes |
| --- | --- | --- | --- | --- |
| Methods V-F | AUC, F1, accuracy, sensitivity, specificity | interpretation.py | `classification_metrics` | AUC primary |
| Methods V-F | spatial Pearson r + Spearman rho | interpretation.py | `spatial_concordance` | 50 x 50 grid per cell type, r = 0.78 |
| Methods V-F | radial-profile residual | interpretation.py | `radial_residual` | r_residual = 0.61 after removing radial mean |
| Methods V-F | DeLong test, Bonferroni | interpretation.py | `delong_test` | pairwise AUC comparison |
| Methods V-F | bootstrap 1000x, 95% CI | interpretation.py | `bootstrap_ci` | 20 random seeds |
| Methods V-F | Cohen's d effect size | interpretation.py | `cohens_d` | alpha = 0.05 |
| Methods V-F | ECE and PICP at 90% | interpretation.py | `calibration_metrics` | ensemble uncertainty over 20 seeds |
| Methods V-F | one-way ANOVA + Tukey | interpretation.py | `oneway_anova` | CMS parameter separation |

## Training and engineering

| paper location | item | file | symbol | notes |
| --- | --- | --- | --- | --- |
| Methods V-B | Adam + cosine annealing, 8000 epochs | campaign.py | `Campaign` | seed stored in checkpoint, restored on resume |
| Methods V-F | 20-seed ensemble | campaign.py | `ensemble_run` | mean +/- std across seeds |
| R4 | atomic checkpoint writes | rig.py | `atomic_save` | tmp file + os.replace |
| R4 | single seed utility | rig.py | `set_seed` | seeds python / numpy / torch |
| Methods E | per-patient inversion + export | console.py | `invert`, `export` | plac CLI verbs |

## Verbatim Code Availability statement (provenance; intentionally not in README)

"A PyTorch/DeepXDE implementation of the inverse-PINN constraint-inversion framework can be
obtained from the corresponding author on reasonable request, available to editors and reviewers
throughout the review process; the code will go open source with a permanent DOI once the paper
is published."

## Ethics statement (provenance)

"Only de-identified public data went into this work. The authors declare no competing interests.
Funding sources, author contributions, and acknowledgments are held back for double-anonymous
peer review and will be added once the paper is accepted."
