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
- bezier flattening: `36` segments per curve
- contour layers: all contours found in SVG
- mapping span: `0.7`
- mapping span anchor: `mid-rim`
- rim clearance: `5.0`
- inner clearance: `0.05`
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

### `--mapping-span`

Fraction of the mapping profile used from rim inward (`0-1`].
Lower values pack contour layers closer together and usually preserve more waviness while keeping a single global waviness scale.
Compression is anchored to `mid-rim` by default.

```bash
python3 build_stepped_bowl_from_svg.py --mapping-span 0.6
```

### `--mapping-span-anchor`

Controls which end of the mapping profile receives span compression.
- `center`: clumps layers closer to the bowl center.
- `mid-rim` (default): anchors the span between `rim` and `center` (useful to reduce inner clamping while still shifting inward).
- `rim`: clumps layers closer to the rim.

```bash
python3 build_stepped_bowl_from_svg.py --mapping-span 0.6 --mapping-span-anchor mid-rim
```

### `--rim-clearance`

Minimum radial clearance from the bowl rim used when clamping wave amplitude.
Increase this to keep the outer ring farther from the rim.

```bash
python3 build_stepped_bowl_from_svg.py --rim-clearance 0.35
```

### `--inner-clearance`

Minimum radial clearance from the inner pattern boundary used when clamping wave amplitude.
Lower values allow more waviness near the center.

```bash
python3 build_stepped_bowl_from_svg.py --inner-clearance 0.05
```

## Combined Example

```bash
python3 build_stepped_bowl_from_svg.py \
  --svg contours.svg \
  --profile-svg profile.svg \
  --curve-segments 36 \
  --contour-layers 5 \
  --mapping-span 0.7 \
  --mapping-span-anchor mid-rim \
  --rim-clearance 5 \
  --inner-clearance 0.05 \
  --output bowl_v2.stl
```

