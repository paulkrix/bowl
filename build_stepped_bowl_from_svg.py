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
POINT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")
ALLOWED_COMMANDS = set("MmLlHhVvCcSsQqTtZz")


@dataclass(frozen=True)
class BowlConfig:
    bowl_diameter: float = 170.0
    bowl_height: float = 20.0
    wall_thickness: float = 3.0
    foot_diameter: float = 60.0
    foot_ring_width: float = 4.0
    foot_depth: float = 5.0
    step_thickness: float = 2.0
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


@dataclass
class ProfileGeometry:
    outer_profile: List[Point2D]
    interior_cavity_profile: List[Point2D]
    contour_mapping_profile: List[Point2D]
    foot_ridge_profile: List[Point2D] | None = None


def dedupe_polyline(points: Sequence[Point2D], tol: float = 1e-9) -> List[Point2D]:
    out: List[Point2D] = []
    for x, y in points:
        if not out or abs(x - out[-1][0]) > tol or abs(y - out[-1][1]) > tol:
            out.append((x, y))
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) <= tol and abs(out[0][1] - out[-1][1]) <= tol:
        out.pop()
    return out


def ensure_closed(points: Sequence[Point2D], tol: float = 1e-9) -> List[Point2D]:
    out = dedupe_polyline(points, tol=tol)
    if out and (abs(out[0][0] - out[-1][0]) > tol or abs(out[0][1] - out[-1][1]) > tol):
        out.append(out[0])
    return out


def ensure_open(points: Sequence[Point2D], tol: float = 1e-9) -> List[Point2D]:
    out = dedupe_polyline(points, tol=tol)
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) <= tol and abs(out[0][1] - out[-1][1]) <= tol:
        out.pop()
    return out


def cubic_bezier(p0: Point2D, p1: Point2D, p2: Point2D, p3: Point2D, t: float) -> Point2D:
    u = 1.0 - t
    x = (u * u * u * p0[0]) + (3.0 * u * u * t * p1[0]) + (3.0 * u * t * t * p2[0]) + (t * t * t * p3[0])
    y = (u * u * u * p0[1]) + (3.0 * u * u * t * p1[1]) + (3.0 * u * t * t * p2[1]) + (t * t * t * p3[1])
    return (x, y)


def quadratic_bezier(p0: Point2D, p1: Point2D, p2: Point2D, t: float) -> Point2D:
    u = 1.0 - t
    x = (u * u * p0[0]) + (2.0 * u * t * p1[0]) + (t * t * p2[0])
    y = (u * u * p0[1]) + (2.0 * u * t * p1[1]) + (t * t * p2[1])
    return (x, y)


def parse_svg_path_to_polyline(d: str, curve_segments: int = 24) -> List[Point2D]:
    tokens = TOKEN_RE.findall(d)
    if not tokens:
        return []

    curve_segments = max(2, curve_segments)
    points: List[Point2D] = []
    x = 0.0
    y = 0.0
    start: Point2D | None = None
    command: str | None = None
    prev_command: str | None = None
    last_cubic_ctrl: Point2D | None = None
    last_quad_ctrl: Point2D | None = None
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
            last_cubic_ctrl = None
            last_quad_ctrl = None
            prev_command = command
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
            last_cubic_ctrl = None
            last_quad_ctrl = None
            prev_command = "M"
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
            last_cubic_ctrl = None
            last_quad_ctrl = None
            prev_command = "L"
            continue

        if command in ("H", "h"):
            while i < len(tokens) and not is_command(tokens[i]):
                vx = read_float(tokens[i])
                i += 1
                x = x + vx if command == "h" else vx
                points.append((x, y))
            last_cubic_ctrl = None
            last_quad_ctrl = None
            prev_command = "H"
            continue

        if command in ("V", "v"):
            while i < len(tokens) and not is_command(tokens[i]):
                vy = read_float(tokens[i])
                i += 1
                y = y + vy if command == "v" else vy
                points.append((x, y))
            last_cubic_ctrl = None
            last_quad_ctrl = None
            prev_command = "V"
            continue

        if command in ("C", "c"):
            while (
                i + 5 < len(tokens)
                and not is_command(tokens[i])
                and not is_command(tokens[i + 1])
                and not is_command(tokens[i + 2])
                and not is_command(tokens[i + 3])
                and not is_command(tokens[i + 4])
                and not is_command(tokens[i + 5])
            ):
                x1, y1 = read_float(tokens[i]), read_float(tokens[i + 1])
                x2, y2 = read_float(tokens[i + 2]), read_float(tokens[i + 3])
                x3, y3 = read_float(tokens[i + 4]), read_float(tokens[i + 5])
                i += 6
                if command == "c":
                    c1 = (x + x1, y + y1)
                    c2 = (x + x2, y + y2)
                    p3 = (x + x3, y + y3)
                else:
                    c1 = (x1, y1)
                    c2 = (x2, y2)
                    p3 = (x3, y3)
                p0 = (x, y)
                for j in range(1, curve_segments + 1):
                    t = j / curve_segments
                    points.append(cubic_bezier(p0, c1, c2, p3, t))
                x, y = p3
                last_cubic_ctrl = c2
                last_quad_ctrl = None
            prev_command = "C"
            continue

        if command in ("S", "s"):
            while (
                i + 3 < len(tokens)
                and not is_command(tokens[i])
                and not is_command(tokens[i + 1])
                and not is_command(tokens[i + 2])
                and not is_command(tokens[i + 3])
            ):
                x2, y2 = read_float(tokens[i]), read_float(tokens[i + 1])
                x3, y3 = read_float(tokens[i + 2]), read_float(tokens[i + 3])
                i += 4
                if prev_command in ("C", "S") and last_cubic_ctrl is not None:
                    c1 = (2.0 * x - last_cubic_ctrl[0], 2.0 * y - last_cubic_ctrl[1])
                else:
                    c1 = (x, y)
                if command == "s":
                    c2 = (x + x2, y + y2)
                    p3 = (x + x3, y + y3)
                else:
                    c2 = (x2, y2)
                    p3 = (x3, y3)
                p0 = (x, y)
                for j in range(1, curve_segments + 1):
                    t = j / curve_segments
                    points.append(cubic_bezier(p0, c1, c2, p3, t))
                x, y = p3
                last_cubic_ctrl = c2
                last_quad_ctrl = None
            prev_command = "S"
            continue

        if command in ("Q", "q"):
            while (
                i + 3 < len(tokens)
                and not is_command(tokens[i])
                and not is_command(tokens[i + 1])
                and not is_command(tokens[i + 2])
                and not is_command(tokens[i + 3])
            ):
                x1, y1 = read_float(tokens[i]), read_float(tokens[i + 1])
                x2, y2 = read_float(tokens[i + 2]), read_float(tokens[i + 3])
                i += 4
                if command == "q":
                    c1 = (x + x1, y + y1)
                    p2 = (x + x2, y + y2)
                else:
                    c1 = (x1, y1)
                    p2 = (x2, y2)
                p0 = (x, y)
                for j in range(1, curve_segments + 1):
                    t = j / curve_segments
                    points.append(quadratic_bezier(p0, c1, p2, t))
                x, y = p2
                last_quad_ctrl = c1
                last_cubic_ctrl = None
            prev_command = "Q"
            continue

        if command in ("T", "t"):
            while i + 1 < len(tokens) and not is_command(tokens[i]) and not is_command(tokens[i + 1]):
                x2, y2 = read_float(tokens[i]), read_float(tokens[i + 1])
                i += 2
                if prev_command in ("Q", "T") and last_quad_ctrl is not None:
                    c1 = (2.0 * x - last_quad_ctrl[0], 2.0 * y - last_quad_ctrl[1])
                else:
                    c1 = (x, y)
                if command == "t":
                    p2 = (x + x2, y + y2)
                else:
                    p2 = (x2, y2)
                p0 = (x, y)
                for j in range(1, curve_segments + 1):
                    t = j / curve_segments
                    points.append(quadratic_bezier(p0, c1, p2, t))
                x, y = p2
                last_quad_ctrl = c1
                last_cubic_ctrl = None
            prev_command = "T"
            continue

        raise ValueError(f"Unhandled SVG command: {command}")

    return dedupe_polyline(points)


def parse_svg_points(points_attr: str) -> List[Point2D]:
    nums = [float(v) for v in POINT_RE.findall(points_attr)]
    if len(nums) < 4 or len(nums) % 2 != 0:
        return []
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


def parse_svg_profile_element(el: ET.Element, curve_segments: int = 24) -> List[Point2D]:
    tag = el.tag.split("}")[-1]
    if tag == "path":
        d = (el.get("d") or "").strip()
        if not d:
            return []
        return parse_svg_path_to_polyline(d, curve_segments=curve_segments)
    if tag in ("polyline", "polygon"):
        points_attr = (el.get("points") or "").strip()
        return parse_svg_points(points_attr)
    return []


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


def default_profile_geometry(cfg: BowlConfig) -> ProfileGeometry:
    rim = cfg.rim_radius
    foot_outer = cfg.foot_outer_radius
    inner_limit = foot_outer + cfg.foot_ring_width + 2.0

    return ProfileGeometry(
        outer_profile=[
            (0.0, 0.0),
            (foot_outer, 0.0),
            (foot_outer + cfg.foot_ring_width, 0.6),
            (foot_outer + 10.0, 1.8),
            (foot_outer + 18.0, 4.2),
            (rim - 12.0, 10.5),
            (rim - 4.0, 18.0),
            (rim, cfg.bowl_height),
            (0.0, cfg.bowl_height),
            (0.0, 0.0),
        ],
        interior_cavity_profile=[
            (0.0, cfg.wall_thickness),
            (18.0, cfg.wall_thickness + 0.5),
            (30.0, cfg.wall_thickness + 2.2),
            (42.0, cfg.wall_thickness + 6.2),
            (52.0, cfg.wall_thickness + 10.8),
            (cfg.rim_radius - cfg.wall_thickness + 0.3, cfg.bowl_height + 2.0),
            (0.0, cfg.bowl_height + 2.0),
            (0.0, cfg.wall_thickness),
        ],
        contour_mapping_profile=[
            (rim - 8.0, cfg.bowl_height - 1.0),
            (rim - 12.0, 16.0),
            (rim - 18.0, 9.0),
            (foot_outer + 16.0, 4.5),
            (foot_outer + 10.0, 2.0),
            (inner_limit, 0.8),
        ],
        foot_ridge_profile=[
            (cfg.foot_inner_radius, 0.0),
            (cfg.foot_inner_radius, -cfg.foot_depth),
            (cfg.foot_outer_radius, -cfg.foot_depth),
            (cfg.foot_outer_radius, 0.0),
        ],
    )


def load_profile_geometry(profile_svg: Path, cfg: BowlConfig, curve_segments: int = 24) -> ProfileGeometry:
    defaults = default_profile_geometry(cfg)
    if not profile_svg.exists():
        print(f"Profile SVG not found ({profile_svg}), using defaults.")
        return defaults

    root = ET.parse(profile_svg).getroot()

    def get_profile_points(element_id: str, required: bool, closed: bool) -> List[Point2D] | None:
        el = root.find(f".//*[@id='{element_id}']")
        if el is None:
            if required:
                raise ValueError(f"Missing '{element_id}' in {profile_svg}")
            return None
        pts = parse_svg_profile_element(el, curve_segments=curve_segments)
        if len(pts) < 2:
            if required:
                raise ValueError(f"'{element_id}' in {profile_svg} has insufficient points")
            return None
        return ensure_closed(pts) if closed else ensure_open(pts)

    outer = get_profile_points("outer_profile", required=True, closed=True) or defaults.outer_profile
    interior = (
        get_profile_points("interior_cavity_profile", required=True, closed=True)
        or defaults.interior_cavity_profile
    )
    mapping = (
        get_profile_points("contour_mapping_profile", required=True, closed=False)
        or defaults.contour_mapping_profile
    )
    if mapping[0][0] < mapping[-1][0]:
        mapping = list(reversed(mapping))

    foot = get_profile_points("foot_ridge_profile", required=False, closed=False)
    if foot is None:
        foot = defaults.foot_ridge_profile

    return ProfileGeometry(
        outer_profile=outer,
        interior_cavity_profile=interior,
        contour_mapping_profile=mapping,
        foot_ridge_profile=foot,
    )


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


def select_contour_layers(
    contours_inner_to_outer: Sequence[Sequence[Point2D]],
    layer_count: int | None,
) -> List[Sequence[Point2D]]:
    total = len(contours_inner_to_outer)
    if layer_count is None or layer_count == total:
        return list(contours_inner_to_outer)
    if layer_count <= 0:
        raise ValueError("Contour layer count must be at least 1")
    if layer_count > total:
        raise ValueError(
            f"Requested {layer_count} contour layers, but SVG only has {total} contours"
        )
    if layer_count == 1:
        # Keep the outer-most contour for a single visible step.
        return [contours_inner_to_outer[-1]]

    # Evenly sample contours so layers still span inner-to-outer coverage.
    step = (total - 1) / (layer_count - 1)
    indices = [math.floor(i * step) for i in range(layer_count - 1)] + [total - 1]
    return [contours_inner_to_outer[idx] for idx in indices]


def build_outer_bowl(profile_points: Sequence[Point2D]) -> cq.Workplane:
    profile = cq.Workplane("XZ").polyline(ensure_closed(profile_points)).close()
    return profile.revolve(360.0)


def build_foot_ridge(cfg: BowlConfig, foot_ridge_profile: Sequence[Point2D] | None = None) -> cq.Shape:
    """Annular ridge protruding downward from the bowl base."""
    outer = cfg.foot_outer_radius
    inner = cfg.foot_inner_radius
    depth = cfg.foot_depth
    if foot_ridge_profile is not None and len(foot_ridge_profile) >= 2:
        xs = [x for x, _ in foot_ridge_profile if x > 1e-6]
        ys = [y for _, y in foot_ridge_profile]
        if len(xs) >= 2:
            inner = min(xs)
            outer = max(xs)
        if ys:
            depth = max(depth, abs(min(ys)))

    ridge = (
        cq.Workplane("XY")
        .circle(outer)
        .circle(inner)
        .extrude(-depth)
    )
    return ridge.val()


def build_interior_cavity(profile_points: Sequence[Point2D]) -> cq.Shape:
    """Build a smooth interior cavity to guarantee a clean inside surface."""
    cavity = cq.Workplane("XZ").polyline(ensure_closed(profile_points)).close().revolve(360.0)
    return cavity.val()


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
    mapping_profile: Sequence[Point2D],
    cfg: BowlConfig,
) -> cq.Shape | None:
    contours_outer_to_inner = list(reversed(contours_inner_to_outer))
    target_radii = equal_distance_target_radii(len(contours_outer_to_inner), ensure_open(mapping_profile))
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


def build_stepped_bowl(
    svg_path: Path,
    cfg: BowlConfig,
    profile_svg: Path = Path("profile.svg"),
    curve_segments: int = 24,
    contour_layers: int | None = None,
) -> cq.Workplane:
    contours = load_contours_from_svg(svg_path)
    centered_contours, _ = center_contours(contours)
    selected_contours = select_contour_layers(centered_contours, contour_layers)
    profiles = load_profile_geometry(profile_svg, cfg, curve_segments=curve_segments)

    outer = build_outer_bowl(profiles.outer_profile)
    outer_shape = outer.val().fuse(build_foot_ridge(cfg, profiles.foot_ridge_profile))
    decoration = build_step_decoration(
        outer_shape,
        selected_contours,
        profiles.contour_mapping_profile,
        cfg,
    )
    decorated_outer = outer_shape if decoration is None else outer_shape.fuse(decoration)
    interior_cavity = build_interior_cavity(profiles.interior_cavity_profile)
    final_shape = decorated_outer.cut(interior_cavity)
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
    parser.add_argument(
        "--profile-svg",
        type=Path,
        default=Path("profile.svg"),
        help="Editable profile definition SVG (supports Bezier paths).",
    )
    parser.add_argument(
        "--curve-segments",
        type=int,
        default=24,
        help="Number of line segments used to flatten each Bezier segment.",
    )
    parser.add_argument(
        "--contour-layers",
        type=int,
        default=None,
        help="Number of contour layers to use (defaults to all contours in the SVG).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BowlConfig()
    bowl = build_stepped_bowl(
        args.svg,
        cfg,
        profile_svg=args.profile_svg,
        curve_segments=args.curve_segments,
        contour_layers=args.contour_layers,
    )
    cq.exporters.export(bowl, str(args.output))
    print(f"Wrote STL: {args.output}")


if __name__ == "__main__":
    main()
