from __future__ import annotations

from ascitwin.config import load_config


def test_main_matches_paper_compute_profile() -> None:
    exp = load_config("main")
    assert exp.train.epochs == 8000
    assert exp.train.lr == 0.001
    assert exp.train.n_seeds == 20
    assert exp.domain.n_interior == 10000
    assert exp.model.width == 512
    assert exp.model.depth == 4
    assert exp.model.dropout == 0.2
    assert exp.data.n_patients == 623
    assert exp.data.resolution == 50


def test_smoke_is_small() -> None:
    exp = load_config("_smoke")
    assert exp.train.epochs == 2
    assert exp.data.n_patients == 12


def test_ablation_disables_one_equation() -> None:
    exp = load_config("ablation_immune")
    assert exp.train.disabled_equations == ["immune"]


def test_override_applies() -> None:
    exp = load_config("_smoke", ["train.epochs=5"])
    assert exp.train.epochs == 5
