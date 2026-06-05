# Aim 1 Conformity Pipeline

Python tools for Aim 1 of a lumbar interbody fusion cage conformity thesis:

> To quantify the conformity of existing off-the-shelf cages with patient-specific endplates and supporting regions.

The pipeline compares:

1. a cage contact surface mesh, usually STL; and
2. a patient-specific vertebral endplate surface mesh, usually STL.

For each sampled point on the cage contact surface, the code finds the closest
point on the endplate surface and calculates the local cage-to-endplate
distance:

```text
d_i = min_(q in S_e) ||p_i - q||
```

where `p_i` is a sampled cage point, `q` is a point on the endplate surface, and
`S_e` is the endplate surface.

The resulting metrics are geometric proximity/conformity metrics. They are not
finite element contact mechanics results.

## What this pipeline does

- Loads cage and endplate surface meshes (`.stl`, `.obj`, `.ply`) with
  `trimesh`.
- Optionally applies a 4 x 4 cage-to-patient/endplate transform.
- Performs conservative mesh repair:
  - duplicate face removal;
  - degenerate face removal;
  - unreferenced vertex removal;
  - normal fixing where possible.
- Uniformly samples the cage contact surface by triangle area.
- Computes closest-point cage-to-endplate distances.
- Calculates area-weighted geometric conformity metrics.
- Writes per-sample audit data for re-plotting and QC.
- Optionally estimates approximate peripheral rim overlap using PCA projection
  and a convex hull footprint.

## What this pipeline does not do

- It does not calculate true physical contact area.
- It does not calculate contact pressure.
- It does not perform finite element contact analysis.
- It does not infer or correct medical image coordinate systems from STL files.
- It does not automatically scale units.
- It does not apply aggressive smoothing, remeshing, or anatomical shape changes.

Use terms such as **geometric proximity-based contact area**,
**tolerance-based contact area fraction**, **cage-to-endplate surface deviation**,
and **geometric conformity metrics** when reporting outputs.

## Installation

Python 3.10 or newer is recommended.

```bash
cd aim1_conformity_pipeline
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
cd aim1_conformity_pipeline
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`rtree` is included because `trimesh` uses spatial indexing for efficient
closest-point queries.

## Single-case usage

```bash
python aim1_conformity.py \
  --case_id Patient01_L5sup \
  --cage data/Patient01/cage_inferior_contact.stl \
  --endplate data/Patient01/L5_superior_endplate_clean.stl \
  --output results/Patient01_L5sup \
  --samples 100000 \
  --tolerances 0.5 1.0
```

With an optional 4 x 4 transform:

```bash
python aim1_conformity.py \
  --case_id Patient01_L5sup \
  --cage data/Patient01/cage_inferior_contact.stl \
  --endplate data/Patient01/L5_superior_endplate_clean.stl \
  --transform data/Patient01/cage_to_patient_transform.txt \
  --output results/Patient01_L5sup \
  --samples 100000
```

The transform text file may be comma-delimited or whitespace-delimited and must
contain 16 numeric values arranged as a 4 x 4 matrix.

Optional approximate rim overlap:

```bash
python aim1_conformity.py \
  --case_id Patient01_L5sup \
  --cage data/Patient01/cage_inferior_contact.stl \
  --endplate data/Patient01/L5_superior_endplate_clean.stl \
  --output results/Patient01_L5sup \
  --samples 100000 \
  --rim_width_mm 4.0
```

## Batch usage

Create or edit `batch_cases.csv` with columns:

```text
case_id,cage_stl,endplate_stl,transform_file,group,level,endplate_side,cage_model
```

The `transform_file` value may be empty. Relative paths are resolved relative to
the CSV file location.

Run:

```bash
python batch_run.py \
  --cases batch_cases.csv \
  --output results \
  --samples 100000 \
  --tolerances 0.5 1.0
```

For each case, the batch runner writes a case folder:

```text
results/Patient01_L5sup/
```

The batch output folder also contains:

```text
results/all_cases_summary_metrics.csv
```

## Expected inputs

- Cage contact surface mesh: `.stl`, `.obj`, or `.ply`.
- Patient-specific vertebral endplate surface mesh: `.stl`, `.obj`, or `.ply`.
- Optional cage transform: 4 x 4 text matrix, comma-delimited or
  whitespace-delimited.
- Meshes are assumed to be in millimetres.

STL files do not store medical image metadata, patient coordinate system
metadata, or units. Cage-endplate alignment and scale must be checked visually
before trusting metrics.

## Per-case outputs

For `case_id = Patient01_L5sup`, the output folder contains:

```text
Patient01_L5sup_metrics.csv
Patient01_L5sup_sampled_distances.csv
Patient01_L5sup_distance_histogram.png
Patient01_L5sup_deviation_points.ply
Patient01_L5sup_summary.json
```

The sampled distances CSV contains:

```text
x
y
z
distance_mm
area_weight_mm2
closest_x
closest_y
closest_z
closest_triangle_id
```

This file is intended for audit, re-plotting, and debugging.

## Metric interpretation

- `cage_area_mm2`: sampled cage contact surface area.
- `mean_gap_mm`: lower values indicate better average geometric conformity.
- `rms_gap_mm`: lower values indicate better global conformity and fewer large
  mismatches.
- `median_gap_mm`: area-weighted median closest-point gap.
- `p95_gap_mm`: area-weighted 95th percentile gap; useful for detecting large
  mismatch regions.
- `max_gap_mm`: maximum sampled closest-point gap.
- `contact_area_0_5mm_mm2`: cage area within 0.5 mm of the endplate.
- `caf_0_5mm_percent`: higher values indicate more cage area lies within 0.5 mm
  of the endplate.
- `contact_area_1_0mm_mm2`: cage area within 1.0 mm of the endplate.
- `caf_1_0mm_percent`: higher values indicate more cage area lies within 1.0 mm
  of the endplate.
- `rim_overlap_percent_approx`: higher values indicate more cage footprint lies
  over the approximate peripheral support region.
- `gap_volume_approx_mm3`: approximate geometric void measure, not finite
  element contact volume.

Tolerance outputs are generated for all values passed to `--tolerances`. For
example, `--tolerances 0.25 0.5 1.0` adds corresponding `0_25mm`, `0_5mm`, and
`1_0mm` metric columns.

## Approximate rim overlap method

When `--rim_width_mm` is supplied, the code:

1. fits a PCA plane to the endplate vertices;
2. projects endplate vertices and cage sample points to that plane;
3. builds a convex hull of the projected endplate vertices;
4. defines the rim as the outer polygon minus an inward buffer; and
5. reports the area-weighted fraction of cage points in that rim region.

This is only a first-pass support-region estimate. A manually verified
anatomical rim mask or validated support-region segmentation is preferable for
final thesis-quality reporting.

## Scientific limitations

1. Closest-point distance measures geometric proximity, not physical contact
   pressure.
2. Tolerance-based contact area fraction is not true contact area unless
   validated by finite element contact mechanics or experimental pressure film.
3. The approximate rim overlap based on convex hull/PCA is only a first-pass
   support-region estimate. Final anatomical rim/support masks should be
   manually verified.
4. Aggressive mesh smoothing should not be applied because it may destroy
   endplate concavity and curvature.
5. Cage-endplate alignment must be checked visually. A wrong transform or
   inconsistent STL coordinate system will invalidate all metrics.

## Running tests

```bash
cd aim1_conformity_pipeline
pytest
```

The included tests focus on the metric calculations with small arrays where the
expected answers are obvious.
