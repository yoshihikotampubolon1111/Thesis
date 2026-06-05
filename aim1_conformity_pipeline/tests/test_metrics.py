from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aim1_conformity import (  # noqa: E402
    calculate_metrics,
    contact_area,
    contact_area_fraction,
    gap_volume_approx,
    weighted_mean_absolute_gap,
    weighted_percentile,
    weighted_rms_gap,
)


def test_total_area_calculation() -> None:
    distances = np.array([0.1, 0.2, 0.3])
    area_weights = np.array([1.5, 2.5, 3.0])

    metrics = calculate_metrics(distances, area_weights, tolerances=[0.5])

    assert metrics["cage_area_mm2"] == pytest.approx(7.0)


def test_mean_gap() -> None:
    distances = np.array([0.2, 0.4, 0.8, 1.2, 1.6])
    area_weights = np.array([2, 2, 2, 2, 2])

    assert weighted_mean_absolute_gap(distances, area_weights) == pytest.approx(0.84)


def test_rms_gap() -> None:
    distances = np.array([0.2, 0.4, 0.8, 1.2, 1.6])
    area_weights = np.array([2, 2, 2, 2, 2])
    expected = math.sqrt(np.mean(distances**2))

    assert weighted_rms_gap(distances, area_weights) == pytest.approx(expected)
    assert weighted_rms_gap(distances, area_weights) == pytest.approx(0.98387, rel=1e-5)


def test_contact_area_fraction() -> None:
    distances = np.array([0.2, 0.4, 0.8, 1.2, 1.6])
    area_weights = np.array([2, 2, 2, 2, 2])

    assert contact_area(distances, area_weights, 0.5) == pytest.approx(4.0)
    assert contact_area_fraction(distances, area_weights, 0.5) == pytest.approx(0.4)
    assert contact_area_fraction(distances, area_weights, 1.0) == pytest.approx(0.6)


def test_weighted_percentile_equal_weights() -> None:
    distances = np.array([0.2, 0.4, 0.8, 1.2, 1.6])
    area_weights = np.array([2, 2, 2, 2, 2])

    assert weighted_percentile(distances, area_weights, 0) == pytest.approx(0.2)
    assert weighted_percentile(distances, area_weights, 50) == pytest.approx(0.8)
    assert weighted_percentile(distances, area_weights, 100) == pytest.approx(1.6)


def test_weighted_percentile_unequal_weights() -> None:
    values = np.array([0.0, 10.0])
    weights = np.array([1.0, 3.0])

    assert weighted_percentile(values, weights, 50) == pytest.approx(8.333333333)


def test_calculate_metrics_tolerance_keys() -> None:
    distances = np.array([0.2, 0.4, 0.8, 1.2, 1.6])
    area_weights = np.array([2, 2, 2, 2, 2])

    metrics = calculate_metrics(distances, area_weights, tolerances=[0.5, 1.0])

    assert metrics["contact_area_0_5mm_mm2"] == pytest.approx(4.0)
    assert metrics["caf_0_5mm_fraction"] == pytest.approx(0.4)
    assert metrics["caf_0_5mm_percent"] == pytest.approx(40.0)
    assert metrics["contact_area_1_0mm_mm2"] == pytest.approx(6.0)
    assert metrics["caf_1_0mm_fraction"] == pytest.approx(0.6)
    assert metrics["caf_1_0mm_percent"] == pytest.approx(60.0)


def test_gap_volume_approx() -> None:
    distances = np.array([0.5, 1.0, 1.5])
    area_weights = np.array([2.0, 2.0, 2.0])

    assert gap_volume_approx(distances, area_weights) == pytest.approx(6.0)
