"""Aim 1 geometric conformity analysis for cage-to-endplate meshes.

This module implements proximity-based geometric conformity metrics between a
cage contact surface mesh and a patient-specific vertebral endplate surface.

The calculated "contact" metrics are tolerance-based geometric proximity
metrics. They are not true physical contact area, contact pressure, or finite
element contact results.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh


DEFAULT_TOLERANCES_MM = (0.5, 1.0)


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load and validate an STL/OBJ/PLY mesh with vertices and faces.

    Parameters
    ----------
    path:
        Path to the mesh file.

    Returns
    -------
    trimesh.Trimesh
        Loaded mesh.

    Raises
    ------
    FileNotFoundError
        If the input path does not exist.
    ValueError
        If the file cannot be loaded as a mesh with vertices and faces.
    """

    mesh_path = Path(path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file does not exist: {mesh_path}")
    if not mesh_path.is_file():
        raise ValueError(f"Mesh path is not a file: {mesh_path}")

    try:
        loaded = trimesh.load(mesh_path, force="mesh", process=False)
    except Exception as exc:  # pragma: no cover - exact loader errors vary
        raise ValueError(f"Failed to load mesh '{mesh_path}': {exc}") from exc

    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"Mesh file contains an empty scene: {mesh_path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))

    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(
            f"Expected a triangular surface mesh from '{mesh_path}', "
            f"got {type(loaded).__name__}."
        )
    if loaded.vertices is None or len(loaded.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")
    if loaded.faces is None or len(loaded.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")

    return loaded


def load_transform(path: str | Path | None) -> np.ndarray | None:
    """Load an optional 4 x 4 transformation matrix.

    The transform may be comma-delimited or whitespace-delimited. Empty or
    ``None`` paths return ``None``.
    """

    if path is None or str(path).strip() == "":
        return None

    transform_path = Path(path)
    if not transform_path.exists():
        raise FileNotFoundError(f"Transform file does not exist: {transform_path}")
    if not transform_path.is_file():
        raise ValueError(f"Transform path is not a file: {transform_path}")

    text = transform_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Transform file is empty: {transform_path}")

    normalized_text = text.replace(",", " ")
    values = np.fromstring(normalized_text, sep=" ", dtype=float)
    if values.size != 16:
        raise ValueError(
            f"Transform file '{transform_path}' must contain exactly 16 numeric "
            f"values for a 4 x 4 matrix; found {values.size}."
        )

    transform = values.reshape((4, 4))
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"Transform file contains non-finite values: {transform_path}")

    return transform


def apply_transform(
    mesh: trimesh.Trimesh, transform: np.ndarray | None
) -> trimesh.Trimesh:
    """Return a transformed copy of ``mesh`` if a transform is provided."""

    transformed = mesh.copy()
    if transform is not None:
        if transform.shape != (4, 4):
            raise ValueError(f"Transform must have shape (4, 4), got {transform.shape}")
        transformed.apply_transform(transform)
    return transformed


def repair_basic(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Perform conservative mesh repair without smoothing or remeshing.

    The goal is to remove clear topological bookkeeping problems while
    preserving anatomical curvature. This deliberately avoids smoothing,
    decimation, remeshing, or other operations that may alter endplate shape.
    """

    repaired = mesh.copy()

    if hasattr(repaired, "unique_faces"):
        unique_mask = repaired.unique_faces()
        repaired.update_faces(unique_mask)
    elif hasattr(repaired, "remove_duplicate_faces"):
        repaired.remove_duplicate_faces()

    if hasattr(repaired, "nondegenerate_faces"):
        nondegenerate_mask = repaired.nondegenerate_faces()
        repaired.update_faces(nondegenerate_mask)
    elif hasattr(repaired, "remove_degenerate_faces"):
        repaired.remove_degenerate_faces()

    repaired.remove_unreferenced_vertices()

    try:
        trimesh.repair.fix_normals(repaired, multibody=True)
    except TypeError:
        trimesh.repair.fix_normals(repaired)
    except Exception as exc:  # pragma: no cover - trimesh behavior can vary
        warnings.warn(f"Could not fix mesh normals: {exc}", RuntimeWarning)

    if len(repaired.vertices) == 0 or len(repaired.faces) == 0:
        raise ValueError("Mesh became empty during conservative repair.")

    return repaired


def sample_surface_equal_area(
    mesh: trimesh.Trimesh, n_samples: int, seed: int | None = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample a mesh surface by triangle area.

    Each sample receives the same area weight because the sampling probability
    is proportional to surface area:

    ``A_i = total cage contact area / n_samples``.
    """

    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    face_areas = np.asarray(mesh.area_faces, dtype=float)
    total_area = float(face_areas.sum())
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("Mesh surface area must be positive for sampling.")

    probabilities = face_areas / total_area
    rng = np.random.default_rng(seed)
    face_indices = rng.choice(len(mesh.faces), size=n_samples, p=probabilities)
    triangles = mesh.triangles[face_indices]

    # Uniform barycentric sampling over triangles.
    u = rng.random(n_samples)
    v = rng.random(n_samples)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    points = (
        triangles[:, 0]
        + u[:, None] * (triangles[:, 1] - triangles[:, 0])
        + v[:, None] * (triangles[:, 2] - triangles[:, 0])
    )

    area_weights = np.full(n_samples, total_area / n_samples, dtype=float)
    return points, area_weights


def closest_point_distances(
    points: np.ndarray, reference_mesh: trimesh.Trimesh
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Compute closest-point distances from points to a reference mesh.

    Returns distances, closest points, and triangle IDs when available.
    ``rtree`` is usually required by trimesh for efficient proximity queries.
    """

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if len(points) == 0:
        raise ValueError("points array is empty")

    try:
        closest_points, distances, triangle_ids = trimesh.proximity.closest_point(
            reference_mesh, points
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Closest-point queries require optional spatial-index dependencies. "
            "Install the requirements file, including rtree."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to compute closest-point distances: {exc}") from exc

    return (
        np.asarray(distances, dtype=float),
        np.asarray(closest_points, dtype=float),
        np.asarray(triangle_ids, dtype=int) if triangle_ids is not None else None,
    )


def _validate_values_and_weights(
    values: Iterable[float], weights: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite value and weight arrays with a positive total weight."""

    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)

    if values_array.shape != weights_array.shape:
        raise ValueError(
            f"values and weights must have the same shape, got "
            f"{values_array.shape} and {weights_array.shape}"
        )
    if values_array.size == 0:
        raise ValueError("values and weights must not be empty")
    if not np.all(np.isfinite(values_array)):
        raise ValueError("values contain non-finite entries")
    if not np.all(np.isfinite(weights_array)):
        raise ValueError("weights contain non-finite entries")
    if np.any(weights_array < 0):
        raise ValueError("weights must be non-negative")
    if float(weights_array.sum()) <= 0:
        raise ValueError("sum of weights must be positive")

    return values_array, weights_array


def weighted_mean_absolute_gap(
    distances: Iterable[float], area_weights: Iterable[float]
) -> float:
    """Area-weighted mean absolute cage-to-endplate gap in millimetres."""

    distances_array, weights_array = _validate_values_and_weights(
        distances, area_weights
    )
    return float(np.sum(weights_array * np.abs(distances_array)) / np.sum(weights_array))


def weighted_rms_gap(
    distances: Iterable[float], area_weights: Iterable[float]
) -> float:
    """Area-weighted root-mean-square cage-to-endplate gap in millimetres."""

    distances_array, weights_array = _validate_values_and_weights(
        distances, area_weights
    )
    return float(
        math.sqrt(np.sum(weights_array * distances_array**2) / np.sum(weights_array))
    )


def weighted_percentile(
    values: Iterable[float], weights: Iterable[float], percentile: float
) -> float:
    """Compute a weighted percentile using a centered cumulative distribution."""

    if percentile < 0 or percentile > 100:
        raise ValueError(f"percentile must be between 0 and 100, got {percentile}")

    values_array, weights_array = _validate_values_and_weights(values, weights)
    positive = weights_array > 0
    values_array = values_array[positive]
    weights_array = weights_array[positive]

    sorter = np.argsort(values_array)
    sorted_values = values_array[sorter]
    sorted_weights = weights_array[sorter]

    if percentile == 0:
        return float(sorted_values[0])
    if percentile == 100:
        return float(sorted_values[-1])

    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative /= np.sum(sorted_weights)
    return float(np.interp(percentile / 100.0, cumulative, sorted_values))


def contact_area(
    distances: Iterable[float], area_weights: Iterable[float], tolerance: float
) -> float:
    """Tolerance-based geometric proximity area for ``distance <= tolerance``."""

    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    distances_array, weights_array = _validate_values_and_weights(
        distances, area_weights
    )
    return float(np.sum(weights_array[distances_array <= tolerance]))


def contact_area_fraction(
    distances: Iterable[float], area_weights: Iterable[float], tolerance: float
) -> float:
    """Fraction of cage sampled area within a distance tolerance."""

    _, weights_array = _validate_values_and_weights(distances, area_weights)
    return float(contact_area(distances, area_weights, tolerance) / np.sum(weights_array))


def gap_volume_approx(
    distances: Iterable[float], area_weights: Iterable[float]
) -> float:
    """Approximate geometric gap volume in mm^3.

    This is a closest-point geometric approximation:

    ``V_gap ~= sum(A_i * max(d_i, 0))``.

    It is not a true axial projected volume and is not a finite element contact
    volume.
    """

    distances_array, weights_array = _validate_values_and_weights(
        distances, area_weights
    )
    return float(np.sum(weights_array * np.maximum(distances_array, 0.0)))


def _tolerance_label(tolerance: float) -> str:
    """Convert a numeric tolerance to a stable metric key label."""

    return str(float(tolerance)).replace(".", "_").replace("-", "neg_")


def calculate_metrics(
    distances: Iterable[float],
    area_weights: Iterable[float],
    tolerances: Iterable[float] = DEFAULT_TOLERANCES_MM,
) -> dict[str, float]:
    """Calculate area-weighted geometric conformity metrics."""

    distances_array, weights_array = _validate_values_and_weights(
        distances, area_weights
    )
    metrics: dict[str, float] = {
        "cage_area_mm2": float(np.sum(weights_array)),
        "mean_gap_mm": weighted_mean_absolute_gap(distances_array, weights_array),
        "rms_gap_mm": weighted_rms_gap(distances_array, weights_array),
        "median_gap_mm": weighted_percentile(distances_array, weights_array, 50),
        "p95_gap_mm": weighted_percentile(distances_array, weights_array, 95),
        "max_gap_mm": float(np.max(distances_array)),
        "gap_volume_approx_mm3": gap_volume_approx(distances_array, weights_array),
    }

    for tolerance in tolerances:
        tolerance_float = float(tolerance)
        label = _tolerance_label(tolerance_float)
        area = contact_area(distances_array, weights_array, tolerance_float)
        fraction = area / metrics["cage_area_mm2"]
        metrics[f"contact_area_{label}mm_mm2"] = float(area)
        metrics[f"caf_{label}mm_fraction"] = float(fraction)
        metrics[f"caf_{label}mm_percent"] = float(fraction * 100.0)

    return metrics


def rim_overlap_fraction(
    cage_points: np.ndarray,
    area_weights: Iterable[float],
    endplate_mesh: trimesh.Trimesh,
    rim_width_mm: float,
) -> float:
    """Approximate cage area fraction over the peripheral endplate rim.

    This first-pass support-region estimate fits a PCA plane to the endplate
    vertices, projects endplate and cage points to that plane, builds a convex
    hull endplate footprint, and defines the rim as the outer polygon minus an
    inward buffer. A manually verified anatomical rim or support mask is better
    for thesis-quality final analysis.
    """

    if rim_width_mm <= 0:
        raise ValueError(f"rim_width_mm must be positive, got {rim_width_mm}")

    try:
        from scipy.spatial import ConvexHull
        from shapely.geometry import MultiPoint, Point, Polygon
    except ImportError as exc:
        raise RuntimeError(
            "rim_overlap_fraction requires scipy and shapely. Install the "
            "requirements file or omit --rim_width_mm."
        ) from exc

    cage_points = np.asarray(cage_points, dtype=float)
    _, weights_array = _validate_values_and_weights(
        np.ones(len(cage_points)), area_weights
    )

    vertices = np.asarray(endplate_mesh.vertices, dtype=float)
    centroid = vertices.mean(axis=0)
    centered_vertices = vertices - centroid
    _, _, vh = np.linalg.svd(centered_vertices, full_matrices=False)
    basis = vh[:2].T

    endplate_2d = centered_vertices @ basis
    cage_2d = (cage_points - centroid) @ basis

    if len(endplate_2d) < 3:
        raise ValueError("At least three endplate vertices are required for rim overlap.")

    try:
        hull = ConvexHull(endplate_2d)
        boundary_points = endplate_2d[hull.vertices]
        endplate_polygon = Polygon(boundary_points)
    except Exception:
        endplate_polygon = MultiPoint(endplate_2d).convex_hull

    if endplate_polygon.is_empty or endplate_polygon.area <= 0:
        raise ValueError("Could not build a valid endplate footprint polygon.")

    inner_polygon = endplate_polygon.buffer(-rim_width_mm)
    if inner_polygon.is_empty:
        rim_region = endplate_polygon
    else:
        rim_region = endplate_polygon.difference(inner_polygon)

    in_rim = np.array(
        [rim_region.covers(Point(float(x), float(y))) for x, y in cage_2d],
        dtype=bool,
    )
    return float(np.sum(weights_array[in_rim]) / np.sum(weights_array))


def save_distance_histogram(
    distances: Iterable[float], output_png: str | Path, case_id: str
) -> None:
    """Save a histogram of closest-point distances."""

    distances_array = np.asarray(distances, dtype=float)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.hist(distances_array, bins=50, edgecolor="black", linewidth=0.4)
    plt.xlabel("Cage-to-endplate closest-point distance (mm)")
    plt.ylabel("Sample count")
    plt.title(f"{case_id}: geometric proximity distances")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def save_deviation_point_cloud(
    points: np.ndarray, distances: Iterable[float], output_ply: str | Path
) -> None:
    """Save sampled cage points as a coloured PLY point cloud.

    Colour scaling is clipped at the 95th percentile so a single outlier does
    not dominate the colour map.
    """

    points_array = np.asarray(points, dtype=float)
    distances_array = np.asarray(distances, dtype=float)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points_array.shape}")
    if len(points_array) != len(distances_array):
        raise ValueError("points and distances must have the same length")

    finite_distances = distances_array[np.isfinite(distances_array)]
    if finite_distances.size == 0:
        raise ValueError("distances contain no finite values")

    scale = float(np.percentile(finite_distances, 95))
    if scale <= 0:
        scale = float(np.max(finite_distances))
    if scale <= 0:
        scale = 1.0

    normalized = np.clip(distances_array / scale, 0.0, 1.0)
    rgba = plt.get_cmap("viridis")(normalized)
    colors = (rgba[:, :4] * 255).astype(np.uint8)

    output_ply = Path(output_ply)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    point_cloud = trimesh.points.PointCloud(points_array, colors=colors)
    point_cloud.export(output_ply)


def _warn_about_mesh_units(mesh: trimesh.Trimesh, label: str) -> None:
    """Warn about suspicious dimensions without changing mesh units."""

    extents = np.asarray(mesh.extents, dtype=float)
    max_extent = float(np.max(extents)) if extents.size else 0.0
    if max_extent > 1000.0:
        warnings.warn(
            f"{label} mesh has a maximum extent of {max_extent:.1f}. The pipeline "
            "assumes millimetres and does not apply automatic scaling.",
            RuntimeWarning,
        )
    elif 0.0 < max_extent < 1.0:
        warnings.warn(
            f"{label} mesh has a maximum extent of {max_extent:.3f}. The pipeline "
            "assumes millimetres and does not apply automatic scaling.",
            RuntimeWarning,
        )


def _warn_about_stl_metadata(path: str | Path, label: str) -> None:
    """Warn that STL does not preserve medical image metadata."""

    if Path(path).suffix.lower() == ".stl":
        warnings.warn(
            f"{label} is an STL file. STL does not store medical image metadata, "
            "patient coordinate-system metadata, or units; verify alignment and "
            "millimetre scaling externally.",
            RuntimeWarning,
        )


def run_conformity_analysis(
    case_id: str,
    cage_path: str | Path,
    endplate_path: str | Path,
    output_dir: str | Path,
    transform_path: str | Path | None = None,
    samples: int = 100_000,
    tolerances: Iterable[float] = DEFAULT_TOLERANCES_MM,
    rim_width_mm: float | None = None,
    seed: int | None = 42,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a complete single-case geometric conformity analysis.

    Outputs include per-case metrics CSV, sampled distances CSV, distance
    histogram, coloured deviation point cloud, and JSON metadata summary.
    """

    if not case_id:
        raise ValueError("case_id must be non-empty")

    tolerances = [float(tolerance) for tolerance in tolerances]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _warn_about_stl_metadata(cage_path, "Cage mesh")
    _warn_about_stl_metadata(endplate_path, "Endplate mesh")

    cage_mesh = repair_basic(load_mesh(cage_path))
    endplate_mesh = repair_basic(load_mesh(endplate_path))
    _warn_about_mesh_units(cage_mesh, "Cage")
    _warn_about_mesh_units(endplate_mesh, "Endplate")

    transform = load_transform(transform_path)
    cage_mesh = apply_transform(cage_mesh, transform)

    cage_points, area_weights = sample_surface_equal_area(cage_mesh, samples, seed)
    distances, closest_points, triangle_ids = closest_point_distances(
        cage_points, endplate_mesh
    )

    metrics: dict[str, Any] = {
        "case_id": case_id,
        **calculate_metrics(distances, area_weights, tolerances),
    }
    if metadata:
        metrics.update(metadata)

    if rim_width_mm is not None:
        rim_fraction = rim_overlap_fraction(
            cage_points, area_weights, endplate_mesh, rim_width_mm
        )
        metrics["rim_width_mm_approx"] = float(rim_width_mm)
        metrics["rim_overlap_fraction_approx"] = float(rim_fraction)
        metrics["rim_overlap_percent_approx"] = float(rim_fraction * 100.0)

    metrics_csv = output_path / f"{case_id}_metrics.csv"
    sampled_csv = output_path / f"{case_id}_sampled_distances.csv"
    histogram_png = output_path / f"{case_id}_distance_histogram.png"
    deviation_ply = output_path / f"{case_id}_deviation_points.ply"
    summary_json = output_path / f"{case_id}_summary.json"

    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)

    sampled_data: dict[str, Any] = {
        "x": cage_points[:, 0],
        "y": cage_points[:, 1],
        "z": cage_points[:, 2],
        "distance_mm": distances,
        "area_weight_mm2": area_weights,
        "closest_x": closest_points[:, 0],
        "closest_y": closest_points[:, 1],
        "closest_z": closest_points[:, 2],
    }
    if triangle_ids is not None:
        sampled_data["closest_triangle_id"] = triangle_ids
    pd.DataFrame(sampled_data).to_csv(sampled_csv, index=False)

    save_distance_histogram(distances, histogram_png, case_id)
    save_deviation_point_cloud(cage_points, distances, deviation_ply)

    summary = {
        "case_id": case_id,
        "inputs": {
            "cage_path": str(cage_path),
            "endplate_path": str(endplate_path),
            "transform_path": str(transform_path) if transform_path else None,
            "samples": samples,
            "seed": seed,
            "tolerances_mm": tolerances,
            "rim_width_mm": rim_width_mm,
        },
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "sampled_distances_csv": str(sampled_csv),
            "distance_histogram_png": str(histogram_png),
            "deviation_points_ply": str(deviation_ply),
        },
        "method_notes": [
            "Closest-point distances measure geometric proximity, not pressure.",
            "Tolerance-based contact area fraction is not true physical contact area.",
            "Gap volume is a closest-point geometric approximation.",
            "Cage/endplate alignment must be visually checked.",
        ],
        "metrics": metrics,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return metrics


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for a single-case analysis."""

    parser = argparse.ArgumentParser(
        description=(
            "Calculate geometric proximity-based cage-to-endplate conformity metrics."
        )
    )
    parser.add_argument("--case_id", required=True, help="Traceable case identifier.")
    parser.add_argument("--cage", required=True, help="Cage contact surface mesh path.")
    parser.add_argument(
        "--endplate", required=True, help="Patient-specific endplate mesh path."
    )
    parser.add_argument(
        "--transform",
        default=None,
        help="Optional 4 x 4 cage-to-patient transform matrix text file.",
    )
    parser.add_argument("--output", required=True, help="Per-case output directory.")
    parser.add_argument(
        "--samples",
        type=int,
        default=100_000,
        help="Number of area-weighted cage surface samples.",
    )
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=list(DEFAULT_TOLERANCES_MM),
        help="Distance tolerances in mm for proximity-based contact area fractions.",
    )
    parser.add_argument(
        "--rim_width_mm",
        type=float,
        default=None,
        help="Optional approximate rim width in mm for PCA/convex-hull rim overlap.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible surface sampling.",
    )
    return parser


def main() -> None:
    """Command-line entrypoint."""

    parser = _build_parser()
    args = parser.parse_args()

    metrics = run_conformity_analysis(
        case_id=args.case_id,
        cage_path=args.cage,
        endplate_path=args.endplate,
        output_dir=args.output,
        transform_path=args.transform,
        samples=args.samples,
        tolerances=args.tolerances,
        rim_width_mm=args.rim_width_mm,
        seed=args.seed,
    )

    print(f"Completed Aim 1 conformity analysis for {args.case_id}")
    print(f"Metrics written to: {Path(args.output) / f'{args.case_id}_metrics.csv'}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6g}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
