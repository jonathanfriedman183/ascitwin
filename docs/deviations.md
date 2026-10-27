# Deviations

Each entry records an intentional departure from a literal reading of the manuscript, the paper
location it touches, and the justification.

## D1 - DeepXDE replaced by a hand-written PyTorch PINN
- Paper location: Code Availability ("A PyTorch/DeepXDE implementation ..."); Methods V-B.
- Deviation: the physics-informed network, the autograd Laplacian, the collocation sampling, and
  the composite loss are implemented directly in `torch` rather than through DeepXDE.
- Justification: an explicit implementation keeps every governing-equation residual and loss term
  readable in `governing.py` / `inversion.py`, removes a heavyweight dependency, and passes
  `mypy --strict`. The network width/depth (4 x 512 tanh), dropout (0.2), parameter count
  (~2.4M), optimiser (Adam, lr 1e-3, cosine, 8,000 epochs), and collocation count (N_c = 10,000)
  match the paper exactly.

## D2 - Deterministic synthetic cohort stands in for restricted bulk data
- Paper location: Methods V-E; Data Availability.
- Deviation: CIBERSORTx outputs and several cohorts require controlled access or external tools.
  `acquisition.SyntheticCohort` generates a deterministic cohort whose hidden spatial fields carry
  a recoverable immune signal, so the full inversion-and-evaluation pipeline runs end to end
  offline. Real data is read through `acquisition.ManifestLoader` from a user-provided manifest.
- Justification: keeps the release runnable and testable without redistributing controlled data,
  while preserving the exact aggregate-observation interface (22 fractions + 3 pathway scores).

## D3 - Spatial integrals via fixed-grid quadrature
- Paper location: Methods V-C, Eq. (7).
- Deviation: the volume-average constraint integral is evaluated by midpoint quadrature on the
  same 50 x 50 grid used for spatial validation rather than by an adaptive quadrature rule.
- Justification: the grid matches the Visium HD ground-truth resolution (100 x 100 um^2 cells),
  so the constraint is evaluated on the same support as the spatial concordance metric; the
  normalisation in Eq. (7) makes the constraint invariant to the constant cell area.
