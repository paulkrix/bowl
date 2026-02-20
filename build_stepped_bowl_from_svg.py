#!/usr/bin/env python3
"""Build a shallow stepped bowl STL from concentric contours in an SVG file."""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cadquery as cq

Point2D = Tuple[float, float]

TOKEN_RE = re.compile(
    r"[A-Za-z]|[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"
)
ALLOWED_COMMANDS = set("MmLlHhVvZz")


@dataclass(frozen=True)
class BowlConfig:
    bowl_diameter: float = 120.0
    bowl_height: float = 24.0
    wall_thickness: float = 4.0
    foot_diameter: float = 40.0
    foot_ring_width: float = 4.0
    foot_depth: float = 5.0
    step_thickness: float = 1.0
    rim_clearance: float = 0.6
    pattern_inner_clearance: float = 0.4
    mask_z_min: float = -80.0
    mask_z_max: float = 80.0

    @property
    def rim_radius(self) -> float:
        return self.bowl_diameter * 0.5

    @property
    def foot_outer_radius(self) -> float:
        return self.foot_diameter * 0.5

    @property
    def foot_inner_radius(self) -> float:
        return self.foot_outer_radius - self.foot_ring_width


def dedupe_polyline(points: Sequence[Point2D], tol: float = 1e-9) -> List[Point2D]:
    out: List[Point2D] = []
    for x, y in points:
        if not out or abs(x - out[-1][0]) > tol or abs(y - out[-1][1]) > tol:
            out.append((x, y))
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) <= tol and abs(out[0][1] - out[-1][1]) <= tol:
        out.pop()
    return out


def parse_svg_path_to_polyline(d: str) -> List[Point2D]:
    tokens = TOKEN_RE.findall(d)
    if not tokens:
        return []

    points: List[Point2D] = []
    x = 0.0
    y = 0.0
    start: Point2D | None = None
    command: str | None = None
    i = 0

    def is_command(tok: str) -> bool:
        return len(tok) == 1 and tok.isalpha()

    def read_float(tok: str) -> float:
        return float(tok)

    while i < len(tokens):
        tok = tokens[i]
        if is_command(tok):
            if tok not in ALLOWED_COMMANDS:
                raise ValueError(f"Unsupported SVG path command: {tok}")
            command = tok
            i += 1

        if command is None:
            raise ValueError("Malformed path: numeric token before command")

        if command in ("Z", "z"):
            if start is not None:
                points.append(start)
                x, y = start
            command = None
            continue

        if command in ("M", "m"):
            first = True
            while i + 1 < len(tokens) and not is_command(tokens[i]) and not is_command(tokens[i + 1]):
                vx = read_float(tokens[i])
                vy = read_float(tokens[i + 1])
                i += 2
                if command == "m":
                    x += vx
                    y += vy
                else:
                    x = vx
                    y = vy
                if first:
                    start = (x, y)
                    first = False
                points.append((x, y))
            command = "l" if command == "m" else "L"
            continue

        if command in ("L", "l"):
            while i + 1 < len(tokens) and not is_command(tokens[i]) and not is_command(tokens[i + 1]):
                vx = read_float(tokens[i])
                vy = read_float(tokens[i + 1])
                i += 2
                if command == "l":
                    x += vx
                    y += vy
                else:
                    x = vx
                    y = vy
                points.append((x, y))
            continue

        if command in ("H", "h"):
            while i < len(tokens) and not is_command(tokens[i]):
                vx = read_float(tokens[i])
                i += 1
                x = x + vx if command == "h" else vx
                points.append((x, y))
            continue

        if command in ("V", "v"):
            while i < len(tokens) and not is_command(tokens[i]):
                vy = read_float(tokens[i])
                i += 1
                y = y + vy if command == "v" else vy
                points.append((x, y))
            continue

        raise ValueError(f"Unhandled SVG command: {command}")

    return dedupe_polyline(points)


def polygon_area(points: Sequence[Point2D]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def polygon_centroid(points: Sequence[Point2D]) -> Point2D:
    area = polygon_area(points)
    if abs(area) < 1e-12:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return ((min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5)
    cx = 0.0
    cy = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    scale = 1.0 / (6.0 * area)
    return (cx * scale, cy * scale)


def load_contours_from_svg(svg_path: Path) -> List[List[Point2D]]:
    root = ET.parse(svg_path).getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    contours: List[List[Point2D]] = []

    for path_el in root.findall(".//svg:path", ns):
        d = (path_el.get("d") or "").strip()
        if not d:
            continue
        style = (path_el.get("style") or "").replace(" ", "").lower()
        if "fill:none" not in style:
            continue
        points = parse_svg_path_to_polyline(d)
        if len(points) >= 3 and abs(polygon_area(points)) > 1e-6:
            contours.append(points)

    if not contours:
        raise ValueError(f"No usable contour paths found in {svg_path}")

    return contours


def center_contours(contours: Sequence[Sequence[Point2D]]) -> Tuple[List[List[Point2D]], Point2D]:
    center = polygon_centroid(contours[-1])  # outer-most contour
    cx, cy = center
    centered: List[List[Point2D]] = []
    for contour in contours:
        centered.append([(x - cx, y - cy) for x, y in contour])
    return centered, center


def polyline_arc_lengths(polyline: Sequence[Point2D]) -> List[float]:
    lengths = [0.0]
    for (x1, y1), (x2, y2) in zip(polyline, polyline[1:]):
        lengths.append(lengths[-1] + math.hypot(x2 - x1, y2 - y1))
    return lengths


def interpolate_along_polyline(polyline: Sequence[Point2D], distance: float) -> Point2D:
    if len(polyline) < 2:
        return polyline[0]
    lengths = polyline_arc_lengths(polyline)
    total = lengths[-1]
    if total <= 0.0:
        return polyline[0]
    target = min(max(distance, 0.0), total)
    for i in range(len(polyline) - 1):
        if lengths[i + 1] >= target:
            seg_len = lengths[i + 1] - lengths[i]
            if seg_len <= 0.0:
                return polyline[i]
            t = (target - lengths[i]) / seg_len
            x1, y1 = polyline[i]
            x2, y2 = polyline[i + 1]
            return (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
    return polyline[-1]


def decorative_profile_for_mapping(cfg: BowlConfig) -> List[Point2D]:
    rim = cfg.rim_radius
    foot_outer = cfg.foot_outer_radius
    inner_limit = foot_outer + cfg.foot_ring_width + 2.0
    return [
        (rim - 8.0, cfg.bowl_height - 1.0),
        (rim - 12.0, 16.0),
        (rim - 18.0, 9.0),
        (foot_outer + 16.0, 4.5),
        (foot_outer + 10.0, 2.0),
        (inner_limit, 0.8),
    ]


def equal_distance_target_radii(count: int, profile: Sequence[Point2D]) -> List[float]:
    if count <= 0:
        return []
    if count == 1:
        return [profile[0][0]]
    total = polyline_arc_lengths(profile)[-1]
    return [
        interpolate_along_polyline(profile, (i / (count - 1)) * total)[0]
        for i in range(count)
    ]


def build_outer_bowl(cfg: BowlConfig) -> cq.Workplane:
    rim = cfg.rim_radius
    foot_outer = cfg.foot_outer_radius
    foot_inner = cfg.foot_inner_radius

    profile_points = [
        (0.0, cfg.foot_depth),
        (foot_inner, cfg.foot_depth),
        (foot_inner, 0.0),
        (foot_outer, 0.0),
        (foot_outer + cfg.foot_ring_width, 0.6),
        (foot_outer + 10.0, 1.8),
        (foot_outer + 18.0, 4.2),
        (rim - 12.0, 10.5),
        (rim - 4.0, 18.0),
        (rim, cfg.bowl_height),
        (0.0, cfg.bowl_height),
    ]

    profile = cq.Workplane("XZ").polyline(profile_points).close()
    return profile.revolve(360.0)


def make_contour_mask(points: Sequence[Point2D], cfg: BowlConfig) -> cq.Shape:
    if len(points) < 3:
        raise ValueError("Contour must have at least 3 points")
    height = cfg.mask_z_max - cfg.mask_z_min
    prism = (
        cq.Workplane("XY")
        .polyline(list(points))
        .close()
        .extrude(height)
        .translate((0.0, 0.0, cfg.mask_z_min))
    )
    return prism.val()


def contour_radial_stats(contour: Sequence[Point2D]) -> Tuple[List[float], float, float, float]:
    radii = [math.hypot(x, y) for x, y in contour]
    mean_radius = sum(radii) / len(radii)
    dev_max = max(r - mean_radius for r in radii)
    dev_min = min(r - mean_radius for r in radii)
    return radii, mean_radius, dev_max, dev_min


def compute_global_wave_scale(
    contours_outer_to_inner: Sequence[Sequence[Point2D]],
    target_radii: Sequence[float],
    cfg: BowlConfig,
) -> float:
    rim_limit = cfg.rim_radius - 0.2
    inner_limit = cfg.foot_outer_radius + cfg.foot_ring_width + cfg.pattern_inner_clearance
    scale_limit = math.inf

    for contour, target_r in zip(contours_outer_to_inner, target_radii):
        _, mean_radius, dev_max, dev_min = contour_radial_stats(contour)
        if dev_max > 1e-9:
            scale_limit = min(scale_limit, (rim_limit - target_r) / dev_max)
        if dev_min < -1e-9:
            scale_limit = min(scale_limit, (target_r - inner_limit) / (-dev_min))
        if mean_radius <= 1e-9:
            continue

    if not math.isfinite(scale_limit):
        return 1.0
    return max(scale_limit * 0.98, 1e-4)


def wrap_contour_with_preserved_waves(
    contour: Sequence[Point2D], target_radius: float, wave_scale: float
) -> List[Point2D]:
    radii, mean_radius, _, _ = contour_radial_stats(contour)
    mapped: List[Point2D] = []
    for (x, y), radius in zip(contour, radii):
        theta = math.atan2(y, x)
        mapped_radius = target_radius + (radius - mean_radius) * wave_scale
        mapped.append((math.cos(theta) * mapped_radius, math.sin(theta) * mapped_radius))
    return dedupe_polyline(mapped)


def build_step_decoration(
    outer_shape: cq.Shape,
    contours_inner_to_outer: Sequence[Sequence[Point2D]],
    cfg: BowlConfig,
) -> cq.Shape | None:
    contours_outer_to_inner = list(reversed(contours_inner_to_outer))
    profile = decorative_profile_for_mapping(cfg)
    target_radii = equal_distance_target_radii(len(contours_outer_to_inner), profile)
    wave_scale = compute_global_wave_scale(contours_outer_to_inner, target_radii, cfg)

    decoration: cq.Shape | None = None
    for idx, (contour, target_r) in enumerate(zip(contours_outer_to_inner, target_radii)):
        mapped = wrap_contour_with_preserved_waves(contour, target_r, wave_scale)
        mask = make_contour_mask(mapped, cfg)

        offset = idx * cfg.step_thickness
        shell_hi = outer_shape.translate((0.0, 0.0, -offset))
        shell_lo = outer_shape.translate((0.0, 0.0, -(offset + cfg.step_thickness)))
        # Build an external layer: lower shifted shell minus upper shifted shell.
        layer = shell_lo.cut(shell_hi).intersect(mask)

        if layer.Volume() <= 1e-5:
            continue
        decoration = layer if decoration is None else decoration.fuse(layer)

    return decoration


def build_stepped_bowl(svg_path: Path, cfg: BowlConfig) -> cq.Workplane:
    contours = load_contours_from_svg(svg_path)
    centered_contours, _ = center_contours(contours)

    outer = build_outer_bowl(cfg)
    outer_shape = outer.val()
    hollow_shape = outer.faces(">Z").shell(-cfg.wall_thickness).val()
    inner_void = outer_shape.cut(hollow_shape)
    decoration = build_step_decoration(outer_shape, centered_contours, cfg)
    if decoration is None:
        final_shape = hollow_shape
    else:
        decorated_outer = outer_shape.fuse(decoration)
        final_shape = decorated_outer.cut(inner_void)
    return cq.Workplane("XY").newObject([final_shape])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--svg",
        type=Path,
        default=Path("contours-fewer.svg"),
        help="Input SVG with contours ordered inner-to-outer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bowl_stepped_from_svg.stl"),
        help="Output STL file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BowlConfig()
    bowl = build_stepped_bowl(args.svg, cfg)
    cq.exporters.export(bowl, str(args.output))
    print(f"Wrote STL: {args.output}")


if __name__ == "__main__":
    main()
