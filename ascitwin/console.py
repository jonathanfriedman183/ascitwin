from __future__ import annotations

from pathlib import Path

import plac
import torch

from ascitwin.acquisition import SyntheticCohort
from ascitwin.campaign import (
    fit_patient,
    fit_response_model,
    free_parameter_names,
    invert_cohort,
    predict_proba,
)
from ascitwin.config import ExperimentConfig, load_config
from ascitwin.gathers import PatientSample
from ascitwin.interpretation import bootstrap_ci, classification_metrics, spatial_concordance
from ascitwin.inversion import FieldNetwork
from ascitwin.rig import atomic_save, get_logger

LOG = get_logger("ascitwin")


def _split(args: tuple[str, ...]) -> tuple[str, list[str]]:
    experiment = "_smoke"
    overrides: list[str] = []
    for token in args:
        if token.startswith("experiment="):
            experiment = token.split("=", 1)[1]
        elif "=" in token:
            overrides.append(token)
    return experiment, overrides


def _cohort(exp: ExperimentConfig) -> list[PatientSample]:
    builder = SyntheticCohort(exp.data.cohort, resolution=exp.data.resolution)
    return builder.sample(exp.data.n_patients, exp.data.seed)


def _run_dir(name: str) -> Path:
    path = Path("runs") / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def train(*args: str) -> None:
    experiment, overrides = _split(args)
    exp = load_config(experiment, overrides)
    samples = _cohort(exp)
    theta, labels = invert_cohort(samples, exp, exp.data.seed)
    model, standardiser = fit_response_model(theta, labels, exp.data.seed)
    destination = _run_dir(exp.name) / "twin.pt"
    atomic_save(
        {
            "config_name": exp.name,
            "theta": theta,
            "labels": labels,
            "state_dict": model.state_dict(),
            "mean": standardiser.mean,
            "scale": standardiser.scale,
        },
        destination,
    )
    LOG.info("trained %s on %d patients -> %s", exp.name, len(samples), destination)


def evaluate(*args: str) -> None:
    experiment, overrides = _split(args)
    exp = load_config(experiment, overrides)
    samples = _cohort(exp)
    half = max(1, len(samples) // 2)
    train_samples = samples[:half]
    val_samples = samples[half:] or samples[:half]
    theta_train, labels_train = invert_cohort(train_samples, exp, exp.data.seed)
    theta_val, labels_val = invert_cohort(val_samples, exp, exp.data.seed + 100)
    model, standardiser = fit_response_model(theta_train, labels_train, exp.data.seed)
    probs = predict_proba(model, standardiser, theta_val)
    metrics = classification_metrics(probs, labels_val)
    low, high = bootstrap_ci(probs, labels_val, n_resamples=200, seed=exp.data.seed)
    LOG.info(
        "evaluate %s AUC %.3f [%.3f, %.3f] F1 %.3f acc %.3f",
        exp.name,
        metrics["auc"],
        low,
        high,
        metrics["f1"],
        metrics["accuracy"],
    )


def invert(*args: str) -> None:
    experiment, overrides = _split(args)
    exp = load_config(experiment, overrides)
    sample = _cohort(exp)[0]
    result = fit_patient(sample, exp, exp.data.seed)
    LOG.info("inverted %s final_loss %.4e", sample.patient_id, result.final_loss)
    for name, value in zip(free_parameter_names(), result.parameters.free_vector(), strict=True):
        LOG.info("  %-9s %.5f", name, value)


def spatial(*args: str) -> None:
    experiment, overrides = _split(args)
    exp = load_config(experiment, overrides)
    builder = SyntheticCohort(exp.data.cohort, resolution=exp.data.resolution)
    sample = builder.sample(exp.data.n_patients, exp.data.seed)[0]
    reference = builder.spatial_reference(sample)
    result = fit_patient(sample, exp, exp.data.seed)
    from ascitwin.collocation import grid_coordinates
    from ascitwin.inversion import DigitalTwin

    twin = DigitalTwin(exp.model, treated=sample.treated, dose_rate=exp.train.dose_rate)
    coords = grid_coordinates(1.0, exp.data.resolution, 1.0)
    with torch.no_grad():
        fields = twin.forward(coords)
    predicted = fields.immune.reshape(exp.data.resolution, exp.data.resolution).numpy()
    truth = reference.map_for("cd8_t")
    pearson, spearman = spatial_concordance(predicted, truth)
    LOG.info(
        "spatial %s pearson %.3f spearman %.3f (final_loss %.4e)",
        sample.patient_id,
        pearson,
        spearman,
        result.final_loss,
    )


def export(*args: str) -> None:
    experiment, overrides = _split(args)
    exp = load_config(experiment, overrides)
    network = FieldNetwork(exp.model)
    network.eval()
    destination = _run_dir(exp.name) / "field.onnx"
    dummy = torch.zeros(1, 3)
    torch.onnx.export(
        network, (dummy,), str(destination), input_names=["coords"], output_names=["fields"]
    )
    LOG.info("exported field network -> %s", destination)


class AsciTwin:
    commands = ("train", "evaluate", "invert", "spatial", "export")

    def train(self, *args: str) -> None:
        train(*args)

    def evaluate(self, *args: str) -> None:
        evaluate(*args)

    def invert(self, *args: str) -> None:
        invert(*args)

    def spatial(self, *args: str) -> None:
        spatial(*args)

    def export(self, *args: str) -> None:
        export(*args)


def main() -> None:
    plac.call(AsciTwin())


if __name__ == "__main__":
    main()
