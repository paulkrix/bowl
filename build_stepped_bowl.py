#!/usr/bin/env python3
"""
Build a 3D-printable stepped bowl from concentric contour lines in an SVG.

Core algorithm implemented from the prompt:
- Map contour loops to equal distances along a bowl profile arc-length.
- Reserve a central undecorated foot ring area.
- For each contour n>=1:
  - contour-n-up: mapped contour at its profile level
  - contour-n-down: same contour shifted by -ridge_height in Z
  - bridge contour-(n-1)-up -> contour-n-down
  - bridge contour-n-down -> contour-n-up

Output:
- OBJ triangle mesh (default)
- STL triangle mesh (optional)
- Optional CadQuery export of the smooth base bowl profile revolve if CadQuery is installed.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Reuse the repository's robust SVG flattening/parsing helpers.
from map_contours_profile_steps import (  # type: ignore
    cumulative_lengths,
    dedupe_consecutive,
    ensure_closed,
    interp_on_polyline,
    load_svg_polylines,
    polygon_centroid,
    polyline_length,
)

try:
    import cadquery as cq  # type: ignore
except Exception:
    cq = None

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]
Face = Tuple[int, int, int]


@dataclass
class BowlParams:
    outer_radius: float = 68.0
    height: float = 24.0
    wall_thickness: float = 3.0
    bottom_thickness: float = 4.0
    foot_radius: float = 18.0
    top_margin: float = 2.5
    ridge_height: float = 3.0
    start_fraction: Optional[float] = None
    end_fraction: Optional[float] = None


@dataclass
class Mesh:
    vertices: List[Point3]
    faces: List[Face]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def vsub(a: Point3, b: Point3) -> Point3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vcross(a: Point3, b: Point3) -> Point3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vnorm(a: Point3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def vunorm(a: Point3) -> Point3:
    n = vnorm(a)
    if n <= 1e-12:
        return (0.0, 0.0, 1.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def polygon_area(points: Sequence[Point2]) -> float:
    poly = ensure_closed(points)
    if len(poly) < 4:
        return 0.0
    area2 = 0.0
    for i in range(len(poly) - 1):
        x0, y0 = poly[i]
        x1, y1 = poly[i + 1]
        area2 += x0 * y1 - x1 * y0
    return 0.5 * area2


def point_in_polygon(pt: Point2, poly: Sequence[Point2]) -> bool:
    x, y = pt
    p = ensure_closed(poly)
    inside = False
    for i in range(len(p) - 1):
        x0, y0 = p[i]
        x1, y1 = p[i + 1]
        cond = (y0 > y) != (y1 > y)
        if not cond:
            continue
        den = y1 - y0
        if abs(den) < 1e-12:
            continue
        x_cross = x0 + (y - y0) * (x1 - x0) / den
        if x < x_cross:
            inside = not inside
    return inside


def load_contours(svg_path: Path, curve_samples: int, min_length: float) -> List[List[Point2]]:
    items = load_svg_polylines(svg_path, curve_samples=curve_samples)
    contours: List[List[Point2]] = []
    for item in items:
        if not item.stroked:
            continue
        pts = ensure_closed(dedupe_consecutive(item.points))
        if len(pts) < 4:
            continue
        if polyline_length(pts) < min_length:
            continue
        contours.append(pts)
    if not contours:
        raise ValueError("No usable stroked contour loops found in SVG.")
    return contours


def detect_shared_center(contours: Sequence[Sequence[Point2]]) -> Point2:
    centroids = [polygon_centroid(c) for c in contours]
    cx = statistics.median(c[0] for c in centroids)
    cy = statistics.median(c[1] for c in centroids)
    center = (cx, cy)

    inside_count = sum(1 for c in contours if point_in_polygon(center, c))
    if inside_count >= max(1, len(contours) // 2):
        return center

    # Fallback: center of the smallest contour if median centroid does not land inside enough loops.
    smallest = min(contours, key=lambda c: abs(polygon_area(c)))
    return polygon_centroid(smallest)


def resample_closed_polyline(points: Sequence[Point2], samples: int) -> List[Point2]:
    if samples < 3:
        raise ValueError("Contour resampling requires at least 3 samples.")

    poly = ensure_closed(dedupe_consecutive(points))
    if len(poly) < 4:
        raise ValueError("Contour polyline is too short to resample.")

    seg_lengths: List[float] = []
    for i in range(len(poly) - 1):
        x0, y0 = poly[i]
        x1, y1 = poly[i + 1]
        seg_lengths.append(math.hypot(x1 - x0, y1 - y0))

    total = sum(seg_lengths)
    if total <= 1e-12:
        x, y = poly[0]
        return [(x, y) for _ in range(samples)]

    out: List[Point2] = []
    seg_i = 0
    seg_start = 0.0
    seg_len = seg_lengths[0]

    for k in range(samples):
        d = (total * k) / float(samples)
        while seg_i < len(seg_lengths) - 1 and d > seg_start + seg_len:
            seg_start += seg_len
            seg_i += 1
            seg_len = seg_lengths[seg_i]

        if seg_len <= 1e-12:
            x, y = poly[seg_i]
            out.append((x, y))
            continue

        u = (d - seg_start) / seg_len
        u = clamp(u, 0.0, 1.0)
        x0, y0 = poly[seg_i]
        x1, y1 = poly[seg_i + 1]
        out.append((x0 + (x1 - x0) * u, y0 + (y1 - y0) * u))

    return out


def contour_mean_radius(points: Sequence[Point2], center: Point2) -> float:
    if not points:
        return 0.0
    return sum(math.hypot(x - center[0], y - center[1]) for x, y in points) / float(len(points))


def contour_max_radius_local(points_local: Sequence[Point2]) -> float:
    if not points_local:
        return 0.0
    return max(math.hypot(x, y) for x, y in points_local)


def reorder_contours_by_radius(contours: List[List[Point2]], center: Point2) -> Tuple[List[List[Point2]], bool]:
    means = [contour_mean_radius(c, center) for c in contours]
    is_increasing = all(means[i] <= means[i + 1] + 1e-9 for i in range(len(means) - 1))
    if is_increasing:
        return contours, False

    ordering = sorted(range(len(contours)), key=lambda i: means[i])
    return [contours[i] for i in ordering], True


def align_loop_to_reference(reference: Sequence[Point2], loop: Sequence[Point2]) -> List[Point2]:
    if len(reference) != len(loop):
        raise ValueError("Loop alignment requires equal point counts.")
    n = len(loop)
    if n == 0:
        return []

    ref_scale = max(1e-9, contour_max_radius_local(reference))
    ref_norm = [(x / ref_scale, y / ref_scale) for x, y in reference]

    def score_shift(cand_norm: Sequence[Point2], shift: int) -> float:
        acc = 0.0
        for i in range(n):
            cx, cy = cand_norm[(i + shift) % n]
            rx, ry = ref_norm[i]
            dx = rx - cx
            dy = ry - cy
            acc += dx * dx + dy * dy
        return acc

    best_score = float("inf")
    best_shift = 0
    best_rev = False

    for rev in (False, True):
        src = list(reversed(loop)) if rev else list(loop)
        src_scale = max(1e-9, contour_max_radius_local(src))
        src_norm = [(x / src_scale, y / src_scale) for x, y in src]
        for shift in range(n):
            s = score_shift(src_norm, shift)
            if s < best_score:
                best_score = s
                best_shift = shift
                best_rev = rev

    src_final = list(reversed(loop)) if best_rev else list(loop)
    return [src_final[(i + best_shift) % n] for i in range(n)]


def align_contour_loops(loops: Sequence[Sequence[Point2]]) -> List[List[Point2]]:
    if not loops:
        return []
    aligned: List[List[Point2]] = [list(loops[0])]
    for i in range(1, len(loops)):
        aligned.append(align_loop_to_reference(aligned[-1], loops[i]))
    return aligned


def make_outer_profile(params: BowlParams) -> List[Point2]:
    r = params.outer_radius
    h = params.height
    fr = params.foot_radius
    # Mostly flat walls with a gentle transition out of the foot.
    return [
        (0.0, 0.0),
        (fr, 0.0),
        (fr + 0.18 * r, 0.10 * h),
        (0.78 * r, 0.58 * h),
        (0.93 * r, 0.88 * h),
        (r, h),
    ]


def make_inner_profile(params: BowlParams) -> List[Point2]:
    r = params.outer_radius
    h = params.height
    t = params.wall_thickness
    bt = params.bottom_thickness

    inner_r_top = max(2.0, r - t)
    return [
        (0.0, bt),
        (max(2.0, 0.22 * r), bt),
        (max(2.0, 0.78 * r - t), 0.58 * h),
        (max(2.0, 0.93 * r - t), 0.88 * h),
        (inner_r_top, h),
    ]


def sample_profile_equal(profile: Sequence[Point2], count: int) -> List[Point2]:
    if count < 2:
        return [profile[0], profile[-1]]
    cum = cumulative_lengths(profile)
    total = cum[-1]
    out: List[Point2] = []
    for i in range(count):
        s = total * i / float(count - 1)
        out.append(interp_on_polyline(profile, cum, s))
    return out


def map_contours_to_up_loops(
    contours_local_xy: Sequence[Sequence[Point2]],
    outer_profile: Sequence[Point2],
    foot_radius: float,
    top_margin: float,
    start_fraction: Optional[float],
    end_fraction: Optional[float],
) -> Tuple[List[List[Point3]], List[float], float, float, float]:
    n = len(contours_local_xy)
    if n == 0:
        return [], [], 0.0, 0.0, 0.0

    cum = cumulative_lengths(outer_profile)
    total_s = cum[-1]

    # Foot ring: start decoration at the profile location whose radius reaches foot_radius.
    foot_s = 0.0
    for i in range(len(outer_profile) - 1):
        r0, _ = outer_profile[i]
        r1, _ = outer_profile[i + 1]
        if (r0 <= foot_radius <= r1) or (r1 <= foot_radius <= r0):
            seg_len = math.hypot(r1 - r0, outer_profile[i + 1][1] - outer_profile[i][1])
            if seg_len <= 1e-12:
                foot_s = cum[i]
                break
            u = 0.0 if abs(r1 - r0) <= 1e-12 else (foot_radius - r0) / (r1 - r0)
            u = clamp(u, 0.0, 1.0)
            foot_s = cum[i] + u * seg_len
            break
    else:
        foot_s = min(cum[-1], max(0.0, cum[1] if len(cum) > 1 else 0.0))

    default_start_s = foot_s
    default_end_s = max(default_start_s + 1e-6, total_s - max(0.0, top_margin))

    if start_fraction is None:
        start_s = default_start_s
    else:
        start_s = clamp(start_fraction, 0.0, 1.0) * total_s

    if end_fraction is None:
        end_s = default_end_s
    else:
        end_s = clamp(end_fraction, 0.0, 1.0) * total_s

    if end_s <= start_s:
        raise ValueError("Contour mapping range is invalid: end must be > start.")

    step = 0.0 if n <= 1 else (end_s - start_s) / float(n - 1)

    up_loops: List[List[Point3]] = []
    contour_s: List[float] = []

    for idx, contour_local in enumerate(contours_local_xy):
        base_s = start_s + idx * step if n > 1 else start_s
        contour_s.append(base_s)

        base_r, _ = interp_on_polyline(outer_profile, cum, base_s)
        ref_r = contour_max_radius_local(contour_local)
        scale = 1.0 if ref_r <= 1e-9 else (base_r / ref_r)
        _, z_target = interp_on_polyline(outer_profile, cum, base_s)

        loop: List[Point3] = []
        for x, y in contour_local:
            loop.append((x * scale, y * scale, z_target))
        up_loops.append(loop)

    return up_loops, contour_s, start_s, end_s, step


def make_circle_loop(radius: float, z: float, samples: int) -> List[Point3]:
    return [
        (
            radius * math.cos(2.0 * math.pi * i / float(samples)),
            radius * math.sin(2.0 * math.pi * i / float(samples)),
            z,
        )
        for i in range(samples)
    ]


def add_vertices(mesh: Mesh, pts: Sequence[Point3]) -> List[int]:
    start = len(mesh.vertices)
    mesh.vertices.extend(pts)
    return list(range(start, start + len(pts)))


def add_face(mesh: Mesh, a: int, b: int, c: int) -> None:
    if a == b or b == c or c == a:
        return
    mesh.faces.append((a, b, c))


def bridge_loops(mesh: Mesh, loop_a: Sequence[int], loop_b: Sequence[int], flip: bool = False) -> None:
    if len(loop_a) != len(loop_b):
        raise ValueError("Cannot bridge loops with different vertex counts.")
    n = len(loop_a)
    for i in range(n):
        j = (i + 1) % n
        a0 = loop_a[i]
        a1 = loop_a[j]
        b0 = loop_b[i]
        b1 = loop_b[j]
        if not flip:
            add_face(mesh, a0, a1, b0)
            add_face(mesh, a1, b1, b0)
        else:
            add_face(mesh, a0, b0, a1)
            add_face(mesh, a1, b0, b1)


def fan_from_center(mesh: Mesh, center_idx: int, ring: Sequence[int], flip: bool) -> None:
    n = len(ring)
    for i in range(n):
        j = (i + 1) % n
        if not flip:
            add_face(mesh, center_idx, ring[i], ring[j])
        else:
            add_face(mesh, center_idx, ring[j], ring[i])


def bridge_top_annulus(mesh: Mesh, outer_loop: Sequence[int], inner_loop: Sequence[int]) -> None:
    if len(outer_loop) != len(inner_loop):
        raise ValueError("Top annulus loops must have the same vertex count.")
    n = len(outer_loop)
    for i in range(n):
        j = (i + 1) % n
        o0 = outer_loop[i]
        o1 = outer_loop[j]
        i0 = inner_loop[i]
        i1 = inner_loop[j]
        # Oriented for +Z on the top rim annulus.
        add_face(mesh, o0, o1, i0)
        add_face(mesh, o1, i1, i0)


def translate_mesh_z(mesh: Mesh, dz: float) -> None:
    if abs(dz) <= 1e-12:
        return
    mesh.vertices = [(x, y, z + dz) for (x, y, z) in mesh.vertices]


def write_obj(path: Path, mesh: Mesh) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Stepped bowl mesh\n")
        for x, y, z in mesh.vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in mesh.faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def write_stl_ascii(path: Path, mesh: Mesh, solid_name: str = "stepped_bowl") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"solid {solid_name}\n")
        for a, b, c in mesh.faces:
            p0 = mesh.vertices[a]
            p1 = mesh.vertices[b]
            p2 = mesh.vertices[c]
            n = vunorm(vcross(vsub(p1, p0), vsub(p2, p0)))
            f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {p0[0]:.6e} {p0[1]:.6e} {p0[2]:.6e}\n")
            f.write(f"      vertex {p1[0]:.6e} {p1[1]:.6e} {p1[2]:.6e}\n")
            f.write(f"      vertex {p2[0]:.6e} {p2[1]:.6e} {p2[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {solid_name}\n")


def export_cadquery_base_bowl(path: Path, outer_profile: Sequence[Point2], inner_profile: Sequence[Point2]) -> bool:
    if cq is None:
        return False

    # Build one closed profile in the XZ plane (local x=radius, local y=z) and revolve about z-axis.
    wp = cq.Workplane("XZ")
    wp = wp.moveTo(outer_profile[0][0], outer_profile[0][1])
    for r, z in outer_profile[1:]:
        wp = wp.lineTo(r, z)
    for r, z in reversed(inner_profile):
        wp = wp.lineTo(r, z)
    wp = wp.close()

    solid = wp.revolve(360.0, (0.0, 0.0), (0.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(solid, str(path))
    return True


def build_mesh_from_contours(
    contours_svg: Path,
    params: BowlParams,
    contour_samples: int,
    curve_samples: int,
    min_contour_length: float,
    inner_sections: int,
) -> Tuple[Mesh, dict]:
    raw_contours = load_contours(contours_svg, curve_samples=curve_samples, min_length=min_contour_length)
    center = detect_shared_center(raw_contours)
    ordered_contours, repaired_order = reorder_contours_by_radius(raw_contours, center)
    local_contours: List[List[Point2]] = []
    for contour in ordered_contours:
        sampled_xy = resample_closed_polyline(contour, contour_samples)
        local_contours.append([(x - center[0], y - center[1]) for x, y in sampled_xy])
    local_contours = align_contour_loops(local_contours)

    outer_profile = make_outer_profile(params)
    inner_profile = make_inner_profile(params)

    up_loops, contour_s, foot_s, end_s, step = map_contours_to_up_loops(
        contours_local_xy=local_contours,
        outer_profile=outer_profile,
        foot_radius=params.foot_radius,
        top_margin=params.top_margin,
        start_fraction=params.start_fraction,
        end_fraction=params.end_fraction,
    )

    if not up_loops:
        raise ValueError("No mapped contour loops were generated.")

    mesh = Mesh(vertices=[], faces=[])

    # Outer base: center disk up to reserved foot ring.
    outer_center = (0.0, 0.0, outer_profile[0][1])
    outer_center_idx = add_vertices(mesh, [outer_center])[0]

    outer_cum = cumulative_lengths(outer_profile)
    outer_total_s = outer_cum[-1]
    foot_r, foot_z = interp_on_polyline(outer_profile, outer_cum, foot_s)
    outer_foot_loop_idx = add_vertices(mesh, make_circle_loop(foot_r, foot_z, contour_samples))
    fan_from_center(mesh, outer_center_idx, outer_foot_loop_idx, flip=True)

    # Outer stepped surfaces.
    up_loop_indices = [add_vertices(mesh, loop) for loop in up_loops]

    bridge_loops(mesh, outer_foot_loop_idx, up_loop_indices[0], flip=False)

    for n in range(1, len(up_loop_indices)):
        down_pts = [(x, y, z - params.ridge_height) for (x, y, z) in up_loops[n]]
        down_idx = add_vertices(mesh, down_pts)

        # contour-(n-1)-up -> contour-n-down
        bridge_loops(mesh, up_loop_indices[n - 1], down_idx, flip=False)
        # contour-n-down -> contour-n-up
        bridge_loops(mesh, down_idx, up_loop_indices[n], flip=False)

    outer_rim = outer_profile[-1]
    outer_rim_loop_idx = add_vertices(mesh, make_circle_loop(outer_rim[0], outer_rim[1], contour_samples))
    bridge_loops(mesh, up_loop_indices[-1], outer_rim_loop_idx, flip=False)

    # Inner smooth bowl surface.
    inner_profile_samples = sample_profile_equal(inner_profile, max(3, inner_sections + 1))
    inner_bottom_center = (0.0, 0.0, inner_profile[0][1])
    inner_bottom_center_idx = add_vertices(mesh, [inner_bottom_center])[0]

    # Start inner loops away from exact center radius.
    inner_loop_points = [p for p in inner_profile_samples[1:] if p[0] > 1e-6]
    if len(inner_loop_points) < 2:
        raise ValueError("Inner profile collapsed; adjust bowl dimensions.")

    inner_loop_indices: List[List[int]] = []
    for r, z in inner_loop_points:
        inner_loop_indices.append(add_vertices(mesh, make_circle_loop(r, z, contour_samples)))

    # Inside floor faces must point upward (+Z) toward the bowl cavity.
    fan_from_center(mesh, inner_bottom_center_idx, inner_loop_indices[0], flip=False)

    for i in range(len(inner_loop_indices) - 1):
        # Reverse orientation for the inner wall so normals point into cavity.
        bridge_loops(mesh, inner_loop_indices[i], inner_loop_indices[i + 1], flip=True)

    # Rim closure (top annulus) between outer and inner loops.
    bridge_top_annulus(mesh, outer_rim_loop_idx, inner_loop_indices[-1])

    # Keep model printable from z=0 upward.
    min_z = min(v[2] for v in mesh.vertices)
    if min_z < 0.0:
        translate_mesh_z(mesh, -min_z)

    stats = {
        "contours_loaded": len(raw_contours),
        "contours_mapped": len(up_loops),
        "center": center,
        "repaired_order": repaired_order,
        "foot_s": foot_s,
        "end_s": end_s,
        "step_s": step,
        "start_fraction_effective": (foot_s / outer_total_s) if outer_total_s > 1e-12 else 0.0,
        "end_fraction_effective": (end_s / outer_total_s) if outer_total_s > 1e-12 else 1.0,
        "outer_profile": outer_profile,
        "inner_profile": inner_profile,
        "contour_s": contour_s,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
    }
    return mesh, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a stepped decorative bowl from contour SVG lines.")
    parser.add_argument("--contours", type=Path, default=Path("contours.svg"), help="Input contour SVG.")
    parser.add_argument("--output-obj", type=Path, default=Path("bowl_stepped.obj"), help="Output OBJ path.")
    parser.add_argument("--output-stl", type=Path, default=Path("bowl_stepped.stl"), help="Output STL path.")
    parser.add_argument(
        "--no-stl",
        action="store_true",
        help="Disable STL export and only write OBJ.",
    )
    parser.add_argument(
        "--output-base-cadquery",
        type=Path,
        default=None,
        help="Optional: export smooth base bowl using CadQuery revolve (STL/STEP supported by extension).",
    )

    parser.add_argument("--ridge-height", type=float, default=1.0, help="contour-n-down Z offset in mm.")
    parser.add_argument("--outer-radius", type=float, default=68.0, help="Bowl outer rim radius in mm.")
    parser.add_argument("--height", type=float, default=16.0, help="Bowl height in mm.")
    parser.add_argument("--wall-thickness", type=float, default=3.0, help="Wall thickness in mm.")
    parser.add_argument("--bottom-thickness", type=float, default=4.0, help="Bottom thickness in mm.")
    parser.add_argument("--foot-radius", type=float, default=18.0, help="Reserved undecorated foot radius in mm.")
    parser.add_argument("--top-margin", type=float, default=2.5, help="Keep decoration this far below rim in mm.")
    parser.add_argument(
        "--start-fraction",
        type=float,
        default=None,
        help="Contour mapping start position on outer profile arc-length (0=axis side, 1=rim side).",
    )
    parser.add_argument(
        "--end-fraction",
        type=float,
        default=None,
        help="Contour mapping end position on outer profile arc-length (0=axis side, 1=rim side).",
    )

    parser.add_argument("--contour-samples", type=int, default=360, help="Angular resampling count per contour.")
    parser.add_argument("--curve-samples", type=int, default=24, help="SVG curve flattening samples.")
    parser.add_argument("--inner-sections", type=int, default=28, help="Number of sections for inner bowl smooth surface.")
    parser.add_argument("--min-contour-length", type=float, default=2.0, help="Drop contour loops shorter than this SVG length.")

    args = parser.parse_args(argv)

    if args.contour_samples < 24:
        raise ValueError("--contour-samples must be >= 24.")
    if args.ridge_height < 0.0:
        raise ValueError("--ridge-height must be >= 0.")
    if args.outer_radius <= 5.0 or args.height <= 5.0:
        raise ValueError("Bowl dimensions are too small; increase --outer-radius and --height.")
    if args.start_fraction is not None and not (0.0 <= args.start_fraction <= 1.0):
        raise ValueError("--start-fraction must be within [0, 1].")
    if args.end_fraction is not None and not (0.0 <= args.end_fraction <= 1.0):
        raise ValueError("--end-fraction must be within [0, 1].")
    if args.start_fraction is not None and args.end_fraction is not None and args.end_fraction <= args.start_fraction:
        raise ValueError("--end-fraction must be greater than --start-fraction.")

    params = BowlParams(
        outer_radius=args.outer_radius,
        height=args.height,
        wall_thickness=args.wall_thickness,
        bottom_thickness=args.bottom_thickness,
        foot_radius=args.foot_radius,
        top_margin=args.top_margin,
        ridge_height=args.ridge_height,
        start_fraction=args.start_fraction,
        end_fraction=args.end_fraction,
    )

    if params.wall_thickness >= params.outer_radius:
        raise ValueError("--wall-thickness must be smaller than --outer-radius.")
    if params.bottom_thickness >= params.height:
        raise ValueError("--bottom-thickness must be smaller than --height.")

    mesh, stats = build_mesh_from_contours(
        contours_svg=args.contours,
        params=params,
        contour_samples=args.contour_samples,
        curve_samples=args.curve_samples,
        min_contour_length=args.min_contour_length,
        inner_sections=args.inner_sections,
    )

    write_obj(args.output_obj, mesh)
    if not args.no_stl:
        write_stl_ascii(args.output_stl, mesh)

    cadquery_exported = False
    if args.output_base_cadquery is not None:
        cadquery_exported = export_cadquery_base_bowl(
            path=args.output_base_cadquery,
            outer_profile=stats["outer_profile"],
            inner_profile=stats["inner_profile"],
        )

    print(f"Contours loaded: {stats['contours_loaded']}")
    print(f"Contours mapped: {stats['contours_mapped']}")
    print(f"Estimated contour center: ({stats['center'][0]:.4f}, {stats['center'][1]:.4f})")
    print(f"Contour order repaired: {stats['repaired_order']}")
    print(f"Profile mapping s_start: {stats['foot_s']:.4f}")
    print(f"Profile mapping s_end: {stats['end_s']:.4f}")
    print(f"Equal contour step: {stats['step_s']:.4f}")
    print(f"Effective fractions: start={stats['start_fraction_effective']:.4f}, end={stats['end_fraction_effective']:.4f}")
    print(f"Mesh vertices: {stats['vertices']}")
    print(f"Mesh faces: {stats['faces']}")
    print(f"Wrote OBJ: {args.output_obj}")
    if not args.no_stl:
        print(f"Wrote STL: {args.output_stl}")

    if args.output_base_cadquery is not None:
        if cadquery_exported:
            print(f"Wrote CadQuery base bowl: {args.output_base_cadquery}")
        else:
            print("CadQuery not available; skipped --output-base-cadquery export.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
