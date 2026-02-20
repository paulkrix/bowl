# Stepped Bowl From SVG

Build a shallow stepped bowl STL from concentric contour lines in an SVG file.

## Requirements

- Python 3.10+
- `cadquery`

Example install:

```bash
pip install cadquery
```

## Basic Usage

```bash
python3 build_stepped_bowl_from_svg.py
```

Default behavior:

- input contours: `contours-fewer.svg`
- profile SVG: `profile.svg`
- bezier flattening: `24` segments per curve
- contour layers: all contours found in SVG
- output STL: `bowl_stepped_from_svg.stl`

## Command Line Arguments

### `--svg`

Input SVG with contours ordered inner-to-outer.

```bash
python3 build_stepped_bowl_from_svg.py --svg contours.svg
```

### `--output`

Output STL path.

```bash
python3 build_stepped_bowl_from_svg.py --output out/custom_bowl.stl
```

### `--profile-svg`

Profile definition SVG (supports Bezier paths).

```bash
python3 build_stepped_bowl_from_svg.py --profile-svg profile1.svg
```

### `--curve-segments`

Number of line segments used to flatten each Bezier curve segment.
Higher values preserve curve detail more closely.

```bash
python3 build_stepped_bowl_from_svg.py --curve-segments 48
```

### `--contour-layers`

Number of contour layers to use from the input contours.
If omitted, all contours are used.

```bash
python3 build_stepped_bowl_from_svg.py --contour-layers 8
```

## Combined Example

```bash
python3 build_stepped_bowl_from_svg.py \
  --svg contours.svg \
  --profile-svg profile.svg \
  --curve-segments 36 \
  --contour-layers 10 \
  --output bowl_v2.stl
```

