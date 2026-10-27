# Repository Plan

## Directory tree

    ascitwin/
      __init__.py          version + public re-exports
      rig.py               seeding, device, atomic io, logging, shared typing
      gathers.py           typed records: aggregate observations, samples, spatial grids
      medium.py            PdeParameters (15 bounded params) + state-field containers
      governing.py         five PDE residual operators (Eq 1-5) + Neumann flux
      collocation.py       interior / boundary / initial collocation samplers
      acquisition.py       cohort registry, deconvolution, CMS, synthetic cohort, spatial ref
      surrogate.py         CIAI learnable label (four ICI correlates)
      inversion.py         FieldNetwork (PINN), CompositeLoss, GradNormBalancer, fraction_constraint
      campaign.py          training loop, ensemble, checkpointing
      interpretation.py    metrics, DeLong, bootstrap, spatial concordance, ablation, ANOVA, calibration
      identifiability.py   Fisher condition number, profile-likelihood intervals
      config.py            OmegaConf structured schema + layered loader
      console.py           plac CLI: train / evaluate / invert / spatial / export
    configs/
      model/backbone.yaml
      data/cohort.yaml
      train/optim.yaml
      experiment/main.yaml
      experiment/ablation_immune.yaml ... (one per Table II row)
      experiment/supplementary_architecture.yaml ...
      experiment/_smoke.yaml
    tests/                 see test plan
    docs/                  project-context, implementation-map, repo-plan, deviations
    scripts/               launch_train.sh, launch_eval.sh, prepare_data.sh
    assets/                .gitkeep
    pyproject.toml requirements.txt environment.yml Dockerfile Makefile
    .gitignore .pre-commit-config.yaml .github/workflows/ci.yml README.md LICENSE

## Module responsibilities

- `rig`: deterministic seeding across python/numpy/torch; device resolution; atomic checkpoint
  writes; a small logging helper; shared `TensorDict`-style typed aliases.
- `gathers`: frozen dataclasses / TypedDicts for the K-dimensional aggregate observation vector,
  a per-patient `PatientSample`, and a `SpatialGrid` container. No tensor maths.
- `medium`: `PdeParameters` (the recovered theta_PDE) with bounded ranges and a `StateFields`
  bundle for (T, I, C, A, P).
- `governing`: the five residuals as pure functions of fields + derivatives + parameters; the
  autograd Laplacian helper; the Neumann boundary flux.
- `collocation`: samplers that draw interior points over Omega x [0, t_f], boundary points, and
  initial-condition points; reproducible via a seeded generator.
- `acquisition`: cohort metadata registry, bulk-expression deconvolution into fractions, CMS
  assignment, the deterministic synthetic cohort, and the spatial reference loader.
- `surrogate`: the four CIAI criteria and their >= 2-of-4 combination into a response label.
- `inversion`: the PINN module, the composite loss, gradient-norm weight balancing, and the
  CIBERSORT fraction constraint that links spatial integrals to aggregate observations.
- `campaign`: the optimisation loop (Adam + cosine), seed-bearing checkpoints, and the
  multi-seed ensemble driver.
- `interpretation`: every reported metric and statistical test.
- `identifiability`: Fisher information condition number and profile-likelihood intervals.
- `config`: the typed configuration schema and the YAML + dotlist loader.
- `console`: the command-line entry points.

## Pinned dependencies

    torch >= 2.2, < 3
    numpy >= 1.26, < 3
    scipy >= 1.11
    omegaconf >= 2.3
    plac >= 1.4

Dev: ruff, black, isort, mypy, pytest, onnx (optional, for export).

## Expected test coverage

| test file | kind | targets |
| --- | --- | --- |
| test_gathers.py | shape / contract | observation vector length, container round-trip |
| test_governing.py | manufactured-solution / finite-difference | residuals vanish on analytic fields; autograd Laplacian vs central difference |
| test_physics_invariants.py | physics-invariant | Neumann zero-flux, non-negativity, Fisher-KPP fixed point, PK exponential decay |
| test_collocation.py | shape / determinism | point counts, domain bounds, seed reproducibility |
| test_surrogate.py | correctness | CIAI >= 2-of-4 truth table |
| test_inversion.py | gradient-flow / overfit-single-batch | grads reach theta_PDE and backbone; loss collapses on one sample |
| test_interpretation.py | metric-vs-reference | AUC / DeLong / bootstrap vs sklearn / scipy; spatial Pearson |
| test_identifiability.py | numerical | finite Fisher condition number; interval ordering |
| test_config.py | config-integrity | main.yaml epochs / lr / N_c equal the paper values |
| test_house_style.py | guard | AST + tokenize no-comment / no-docstring; forbidden phrases + emoji |
| test_training_smoke.py | end-to-end | two optimisation steps on _smoke.yaml, loss decreases |
