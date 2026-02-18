#!/usr/bin/env python3
"""
Map 2D contour polylines from an SVG onto the outer surface of a cup.

The cup surface is defined by revolving a profile polyline in the YZ plane
around the Z axis. The script writes an OBJ containing:
1) a triangulated cup outer surface mesh
2) mapped contour lines as either OBJ lines and/or raised ribbon geometry
"""

from __future__ import annotations

import argparse
import bisect
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]
Segment2 = Tuple[float, float, float, float, float, float, float, float]


@dataclass
class SvgCanvas:
    min_x: float
    min_y: float
    width: float
    height: float

    @property
    def center(self) -> Point2:
        return (self.min_x + 0.5 * self.width, self.min_y + 0.5 * self.height)


@dataclass
class SvgPolyline:
    points: List[Point2]
    stroked: bool
    filled: bool


@dataclass
class DipPolygon:
    points: List[Point2]
    min_x: float
    min_y: float
    max_x: float
    max_y: float


def parse_number(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text.strip())
    if not match:
        return None
    return float(match.group(0))


def parse_viewbox(root: ET.Element) -> SvgCanvas:
    view_box = root.attrib.get("viewBox")
    if view_box:
        parts = [parse_number(p) for p in re.split(r"[,\s]+", view_box.strip()) if p]
        if len(parts) == 4 and all(v is not None for v in parts):
            return SvgCanvas(parts[0], parts[1], parts[2], parts[3])

    width = parse_number(root.attrib.get("width")) or 100.0
    height = parse_number(root.attrib.get("height")) or 100.0
    return SvgCanvas(0.0, 0.0, width, height)


def parse_style(style: Optional[str]) -> Dict[str, str]:
    if not style:
        return {}
    out: Dict[str, str] = {}
    for part in style.split(";"):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def element_paint_flags(elem: ET.Element) -> Tuple[bool, bool]:
    style = parse_style(elem.attrib.get("style"))
    stroke = elem.attrib.get("stroke", style.get("stroke", "none")).strip().lower()
    fill = elem.attrib.get("fill", style.get("fill", "black")).strip().lower()
    stroked = stroke not in ("none", "transparent", "")
    filled = fill not in ("none", "transparent", "")
    return stroked, filled


def parse_points_attr(points_text: str) -> List[Point2]:
    nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", points_text)
    vals = [float(n) for n in nums]
    pts: List[Point2] = []
    for i in range(0, len(vals) - 1, 2):
        pts.append((vals[i], vals[i + 1]))
    return dedupe_consecutive(pts)


def tokenize_path_d(d: str) -> List[str]:
    token_re = re.compile(r"[MmLlHhVvZz]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
    return token_re.findall(d)


def dedupe_consecutive(points: Sequence[Point2], eps: float = 1e-12) -> List[Point2]:
    if not points:
        return []
    out = [points[0]]
    for p in points[1:]:
        if (p[0] - out[-1][0]) ** 2 + (p[1] - out[-1][1]) ** 2 > eps:
            out.append(p)
    return out


def parse_path_d(d: str) -> List[List[Point2]]:
    tokens = tokenize_path_d(d)
    i = 0
    cmd: Optional[str] = None
    current: Point2 = (0.0, 0.0)
    start: Optional[Point2] = None
    curr_poly: List[Point2] = []
    polys: List[List[Point2]] = []

    def is_cmd(tok: str) -> bool:
        return len(tok) == 1 and tok.isalpha()

    def flush_poly() -> None:
        nonlocal curr_poly
        if len(curr_poly) >= 2:
            polys.append(dedupe_consecutive(curr_poly))
        curr_poly = []

    while i < len(tokens):
        tok = tokens[i]
        if is_cmd(tok):
            cmd = tok
            i += 1
        elif cmd is None:
            i += 1
            continue

        if cmd in ("M", "m"):
            first = True
            while i + 1 < len(tokens) and not is_cmd(tokens[i]) and not is_cmd(tokens[i + 1]):
                x = float(tokens[i])
                y = float(tokens[i + 1])
                i += 2
                if cmd == "m":
                    nxt = (current[0] + x, current[1] + y)
                else:
                    nxt = (x, y)
                current = nxt
                if first:
                    flush_poly()
                    curr_poly = [current]
                    start = current
                    first = False
                else:
                    curr_poly.append(current)
            cmd = "L" if cmd == "M" else "l"
            continue

        if cmd in ("L", "l"):
            while i + 1 < len(tokens) and not is_cmd(tokens[i]) and not is_cmd(tokens[i + 1]):
                x = float(tokens[i])
                y = float(tokens[i + 1])
                i += 2
                if cmd == "l":
                    current = (current[0] + x, current[1] + y)
                else:
                    current = (x, y)
                if not curr_poly:
                    curr_poly = [current]
                    start = current
                else:
                    curr_poly.append(current)
            continue

        if cmd in ("H", "h"):
            while i < len(tokens) and not is_cmd(tokens[i]):
                x = float(tokens[i])
                i += 1
                if cmd == "h":
                    current = (current[0] + x, current[1])
                else:
                    current = (x, current[1])
                if not curr_poly:
                    curr_poly = [current]
                    start = current
                else:
                    curr_poly.append(current)
            continue

        if cmd in ("V", "v"):
            while i < len(tokens) and not is_cmd(tokens[i]):
                y = float(tokens[i])
                i += 1
                if cmd == "v":
                    current = (current[0], current[1] + y)
                else:
                    current = (current[0], y)
                if not curr_poly:
                    curr_poly = [current]
                    start = current
                else:
                    curr_poly.append(current)
            continue

        if cmd in ("Z", "z"):
            if curr_poly and start is not None:
                if curr_poly[-1] != start:
                    curr_poly.append(start)
            flush_poly()
            start = None
            cmd = None
            continue

        i += 1

    flush_poly()
    return polys


def load_svg_polylines(svg_path: Path) -> Tuple[SvgCanvas, List[SvgPolyline]]:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    canvas = parse_viewbox(root)
    items: List[SvgPolyline] = []

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag not in ("path", "polyline", "polygon"):
            continue

        stroked, filled = element_paint_flags(elem)
        polys: List[List[Point2]] = []
        if tag == "path":
            d = elem.attrib.get("d")
            if d:
                polys = parse_path_d(d)
        else:
            pts_text = elem.attrib.get("points")
            if pts_text:
                pts = parse_points_attr(pts_text)
                if tag == "polygon" and pts and pts[0] != pts[-1]:
                    pts.append(pts[0])
                if len(pts) >= 2:
                    polys = [pts]

        for p in polys:
            if len(p) >= 2:
                items.append(SvgPolyline(points=p, stroked=stroked, filled=filled))

    return canvas, items


def polyline_length(points: Sequence[Point2]) -> float:
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total


def cumulative_lengths(points: Sequence[Point2]) -> List[float]:
    out = [0.0]
    for a, b in zip(points, points[1:]):
        out.append(out[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    return out


def build_mapping_coordinate(profile_rz: Sequence[Point2], steepness_compensation: float) -> List[float]:
    """
    Build a monotonic coordinate used for contour->profile mapping.
    0.0 uses full meridian arclength; 1.0 uses vertical travel only.
    """
    k = max(0.0, min(1.0, steepness_compensation))
    out = [0.0]
    for (r0, z0), (r1, z1) in zip(profile_rz, profile_rz[1:]):
        dr = abs(r1 - r0)
        dz = abs(z1 - z0)
        ds = math.hypot(dr, dz)
        du = (1.0 - k) * ds + k * dz
        out.append(out[-1] + max(1e-9, du))
    return out


def interp_on_polyline(points: Sequence[Point2], cumlen: Sequence[float], s: float) -> Point2:
    if s <= 0.0:
        return points[0]
    if s >= cumlen[-1]:
        return points[-1]
    i = bisect.bisect_right(cumlen, s) - 1
    i = max(0, min(i, len(points) - 2))
    s0 = cumlen[i]
    s1 = cumlen[i + 1]
    if s1 <= s0:
        return points[i]
    t = (s - s0) / (s1 - s0)
    x = points[i][0] * (1.0 - t) + points[i + 1][0] * t
    y = points[i][1] * (1.0 - t) + points[i + 1][1] * t
    return (x, y)


def resample_polyline(points: Sequence[Point2], samples: int) -> List[Point2]:
    if samples <= 2 or len(points) < 2:
        return list(points)
    cum = cumulative_lengths(points)
    total = cum[-1]
    if total <= 0.0:
        return [points[0]] * samples
    out: List[Point2] = []
    for k in range(samples):
        s = total * (k / (samples - 1))
        out.append(interp_on_polyline(points, cum, s))
    return out


def choose_profile_polyline(polys: Sequence[SvgPolyline], profile_index: int) -> List[Point2]:
    stroked = [p.points for p in polys if p.stroked and len(p.points) >= 2]
    candidates = stroked if stroked else [p.points for p in polys if len(p.points) >= 2]
    if not candidates:
        raise ValueError("No usable polylines found in profile SVG.")
    candidates = sorted(candidates, key=polyline_length, reverse=True)
    idx = max(0, min(profile_index, len(candidates) - 1))
    return dedupe_consecutive(candidates[idx])


def choose_contour_polylines(polys: Sequence[SvgPolyline], min_len: float) -> List[List[Point2]]:
    stroked = [p.points for p in polys if p.stroked and len(p.points) >= 2]
    candidates = stroked if stroked else [p.points for p in polys if len(p.points) >= 2]
    out = [dedupe_consecutive(p) for p in candidates if polyline_length(p) >= min_len]
    if not out:
        raise ValueError("No usable contour polylines found in contour SVG.")
    return out


def polyline_is_closed(points: Sequence[Point2], eps: float = 1e-12) -> bool:
    if len(points) < 3:
        return False
    dx = points[0][0] - points[-1][0]
    dy = points[0][1] - points[-1][1]
    return (dx * dx + dy * dy) <= eps


def smooth_contour_polyline(points: Sequence[Point2], iterations: int, strength: float) -> List[Point2]:
    if iterations <= 0 or strength <= 0.0 or len(points) < 3:
        return list(points)

    alpha = max(0.0, min(1.0, strength))
    closed = polyline_is_closed(points)
    core = list(points[:-1]) if closed else list(points)
    n = len(core)
    if n < 3:
        return list(points)

    out = core
    for _ in range(iterations):
        prev = out
        nxt = list(prev)
        if closed:
            for i in range(n):
                ax, ay = prev[(i - 1) % n]
                bx, by = prev[(i + 1) % n]
                mx = 0.5 * (ax + bx)
                my = 0.5 * (ay + by)
                px, py = prev[i]
                nxt[i] = (px + alpha * (mx - px), py + alpha * (my - py))
        else:
            for i in range(1, n - 1):
                ax, ay = prev[i - 1]
                bx, by = prev[i + 1]
                mx = 0.5 * (ax + bx)
                my = 0.5 * (ay + by)
                px, py = prev[i]
                nxt[i] = (px + alpha * (mx - px), py + alpha * (my - py))
        out = nxt

    if closed:
        out = out + [out[0]]
    return dedupe_consecutive(out)


def smooth_contour_polylines(
    polylines: Sequence[Sequence[Point2]], iterations: int, strength: float
) -> List[List[Point2]]:
    return [smooth_contour_polyline(poly, iterations=iterations, strength=strength) for poly in polylines]


def build_profile_rz(
    profile_points_svg: Sequence[Point2],
    axis_x: Optional[float],
    y_flip: bool,
    scale: float,
    samples: int,
) -> List[Point2]:
    pts = dedupe_consecutive(profile_points_svg)
    if len(pts) < 2:
        raise ValueError("Profile polyline requires at least 2 points.")

    if axis_x is None:
        axis_x = min(p[0] for p in pts)

    z_sign = -1.0 if y_flip else 1.0
    rz = [(abs((x - axis_x) * scale), (z_sign * y * scale)) for x, y in pts]

    # Prefer orientation that starts closest to the rotation axis.
    if rz[0][0] > rz[-1][0]:
        rz.reverse()

    # Re-zero Z at start so base point is Z=0.
    z0 = rz[0][1]
    rz = [(r, z - z0) for r, z in rz]
    rz = resample_polyline(rz, samples)
    return rz


def interp_1d(x: Sequence[float], y: Sequence[float], xq: float) -> float:
    if xq <= x[0]:
        return y[0]
    if xq >= x[-1]:
        return y[-1]
    i = bisect.bisect_right(x, xq) - 1
    i = max(0, min(i, len(x) - 2))
    x0, x1 = x[i], x[i + 1]
    if x1 <= x0:
        return y[i]
    t = (xq - x0) / (x1 - x0)
    return y[i] * (1.0 - t) + y[i + 1] * t


def v_add(a: Point3, b: Point3) -> Point3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a: Point3, b: Point3) -> Point3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_mul(a: Point3, k: float) -> Point3:
    return (a[0] * k, a[1] * k, a[2] * k)


def v_cross(a: Point3, b: Point3) -> Point3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v_norm(a: Point3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def v_unit(a: Point3, fallback: Point3 = (1.0, 0.0, 0.0)) -> Point3:
    n = v_norm(a)
    if n <= 1e-12:
        return fallback
    return (a[0] / n, a[1] / n, a[2] / n)


def build_area_cumulative(s: Sequence[float], r: Sequence[float]) -> List[float]:
    # A(s) = integral(2*pi*r ds), trapezoidal integration.
    area = [0.0]
    for i in range(len(s) - 1):
        ds = s[i + 1] - s[i]
        ring_avg = 0.5 * (r[i] + r[i + 1])
        area.append(area[-1] + 2.0 * math.pi * ring_avg * ds)
    return area


def contour_polys_to_local(
    contours: Sequence[Sequence[Point2]],
    contour_center: Point2,
    contour_scale: float,
) -> Tuple[List[List[Point2]], float]:
    local_contours: List[List[Point2]] = []
    rho_max = 0.0
    for poly in contours:
        local_poly: List[Point2] = []
        for p in poly:
            x, y = svg_point_to_local(p, contour_center, contour_scale, y_flip=True)
            local_poly.append((x, y))
            rho_max = max(rho_max, math.hypot(x, y))
        if len(local_poly) >= 2:
            local_contours.append(local_poly)
    return local_contours, rho_max


def map_rho_to_s(
    rho: float,
    rho_max: float,
    s_max: float,
    mapping: str,
    s_profile: Sequence[float],
    area_profile: Optional[Sequence[float]],
    base_gap_frac: float,
    top_gap_frac: float,
    height_exponent: float,
) -> float:
    if rho_max <= 0.0:
        return 0.0
    t = max(0.0, min(1.0, rho / rho_max))
    t = t**height_exponent
    span = max(1e-12, 1.0 - base_gap_frac - top_gap_frac)
    t = base_gap_frac + span * t
    t = max(0.0, min(1.0, t))
    if mapping == "linear":
        return t * s_max
    if mapping == "area":
        if area_profile is None or area_profile[-1] <= 0.0:
            return t * s_max
        target = (t * t) * area_profile[-1]
        return interp_1d(area_profile, s_profile, target)
    raise ValueError(f"Unsupported mapping mode: {mapping}")


def map_s_to_rho(
    s: float,
    rho_max: float,
    s_max: float,
    mapping: str,
    s_profile: Sequence[float],
    area_profile: Optional[Sequence[float]],
    base_gap_frac: float,
    top_gap_frac: float,
    height_exponent: float,
) -> float:
    if rho_max <= 0.0 or s_max <= 0.0:
        return 0.0

    s_clamped = max(0.0, min(s_max, s))
    if mapping == "linear":
        t = s_clamped / s_max
    elif mapping == "area":
        if area_profile is None or area_profile[-1] <= 0.0:
            t = s_clamped / s_max
        else:
            area_s = interp_1d(s_profile, area_profile, s_clamped)
            t = math.sqrt(max(0.0, min(1.0, area_s / area_profile[-1])))
    else:
        raise ValueError(f"Unsupported mapping mode: {mapping}")

    denom = max(1e-12, 1.0 - base_gap_frac - top_gap_frac)
    u = (t - base_gap_frac) / denom
    u = max(0.0, min(1.0, u))
    t0 = u ** (1.0 / height_exponent)
    return t0 * rho_max


def close_polyline_for_polygon(poly: Sequence[Point2], eps: float = 1e-9) -> List[Point2]:
    if len(poly) < 3:
        return []
    out = list(poly)
    if (out[0][0] - out[-1][0]) ** 2 + (out[0][1] - out[-1][1]) ** 2 > eps:
        out.append(out[0])
    return out


def make_dip_polygons(local_contours: Sequence[Sequence[Point2]]) -> List[DipPolygon]:
    polygons: List[DipPolygon] = []
    for poly in local_contours:
        closed = close_polyline_for_polygon(poly)
        if len(closed) < 4:
            continue
        xs = [p[0] for p in closed]
        ys = [p[1] for p in closed]
        polygons.append(
            DipPolygon(
                points=closed,
                min_x=min(xs),
                min_y=min(ys),
                max_x=max(xs),
                max_y=max(ys),
            )
        )
    return polygons


def point_in_polygon_xy(x: float, y: float, poly: Sequence[Point2]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n - 1):
        x0, y0 = poly[i]
        x1, y1 = poly[i + 1]
        cond = ((y0 > y) != (y1 > y))
        if not cond:
            continue
        x_int = x0 + ((y - y0) * (x1 - x0)) / (y1 - y0)
        if x_int >= x:
            inside = not inside
    return inside


def count_dip_layers_at_point(x: float, y: float, polygons: Sequence[DipPolygon]) -> int:
    count = 0
    for poly in polygons:
        if x < poly.min_x or x > poly.max_x or y < poly.min_y or y > poly.max_y:
            continue
        if point_in_polygon_xy(x, y, poly.points):
            count += 1
    return count


def build_contour_segments(local_contours: Sequence[Sequence[Point2]]) -> List[Segment2]:
    segments: List[Segment2] = []
    for poly in local_contours:
        for (ax, ay), (bx, by) in zip(poly, poly[1:]):
            min_x = min(ax, bx)
            max_x = max(ax, bx)
            min_y = min(ay, by)
            max_y = max(ay, by)
            segments.append((ax, ay, bx, by, min_x, max_x, min_y, max_y))
    return segments


def point_to_segment_distance(x: float, y: float, seg: Segment2) -> float:
    ax, ay, bx, by, _, _, _, _ = seg
    vx = bx - ax
    vy = by - ay
    wx = x - ax
    wy = y - ay
    vv = vx * vx + vy * vy
    if vv <= 1e-18:
        return math.hypot(x - ax, y - ay)
    t = (wx * vx + wy * vy) / vv
    t = max(0.0, min(1.0, t))
    qx = ax + t * vx
    qy = ay + t * vy
    return math.hypot(x - qx, y - qy)


def apply_dip_layers_to_surface(
    surface_vertices: Sequence[Point3],
    profile_rz: Sequence[Point2],
    theta_segments: int,
    mapping: str,
    profile_s: Sequence[float],
    profile_r: Sequence[float],
    rho_max: float,
    base_gap_frac: float,
    top_gap_frac: float,
    height_exponent: float,
    contour_segments: Sequence[Segment2],
    dip_layer_thickness: float,
    dip_taper_width: float,
) -> List[Point3]:
    if dip_layer_thickness <= 0.0 or dip_taper_width <= 0.0 or not contour_segments:
        return list(surface_vertices)

    s_max = profile_s[-1] if profile_s else 0.0
    area_profile = build_area_cumulative(profile_s, profile_r) if mapping == "area" else None
    taper_inv = 1.0 / dip_taper_width
    dipped: List[Point3] = []
    n_meridian = len(profile_rz)

    for i in range(n_meridian):
        s = profile_s[i]
        rho = map_s_to_rho(
            s,
            rho_max=rho_max,
            s_max=s_max,
            mapping=mapping,
            s_profile=profile_s,
            area_profile=area_profile,
            base_gap_frac=base_gap_frac,
            top_gap_frac=top_gap_frac,
            height_exponent=height_exponent,
        )
        for j in range(theta_segments):
            idx = i * theta_segments + j
            x, y, z = surface_vertices[idx]
            theta = (2.0 * math.pi * j) / theta_segments
            x2 = rho * math.cos(theta)
            y2 = rho * math.sin(theta)
            max_thickness = 0.0
            for seg in contour_segments:
                _, _, _, _, min_x, max_x, min_y, max_y = seg
                if (
                    x2 < (min_x - dip_taper_width)
                    or x2 > (max_x + dip_taper_width)
                    or y2 < (min_y - dip_taper_width)
                    or y2 > (max_y + dip_taper_width)
                ):
                    continue
                dist = point_to_segment_distance(x2, y2, seg)
                if dist >= dip_taper_width:
                    continue
                thickness = dip_layer_thickness * (1.0 - dist * taper_inv)
                if thickness > max_thickness:
                    max_thickness = thickness
                    if max_thickness >= dip_layer_thickness:
                        break

            if max_thickness <= 0.0:
                dipped.append((x, y, z))
                continue
            r = math.hypot(x, y)
            if r <= 1e-12:
                dipped.append((x, y, z + max_thickness))
                continue
            r2 = r + max_thickness
            scale = r2 / r
            dipped.append((x * scale, y * scale, z))

    return dipped


def make_surface_mesh(profile_rz: Sequence[Point2], theta_segments: int) -> Tuple[List[Point3], List[Tuple[int, int, int]]]:
    if theta_segments < 3:
        raise ValueError("theta_segments must be >= 3.")

    n_meridian = len(profile_rz)
    verts: List[Point3] = []
    faces: List[Tuple[int, int, int]] = []

    for i in range(n_meridian):
        r, z = profile_rz[i]
        for j in range(theta_segments):
            theta = (2.0 * math.pi * j) / theta_segments
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            verts.append((x, y, z))

    def vid(i: int, j: int) -> int:
        return i * theta_segments + j

    for i in range(n_meridian - 1):
        r0 = profile_rz[i][0]
        r1 = profile_rz[i + 1][0]
        for j in range(theta_segments):
            jn = (j + 1) % theta_segments
            a = vid(i, j)
            b = vid(i + 1, j)
            c = vid(i + 1, jn)
            d = vid(i, jn)

            # Skip triangles collapsing to a single point at the axis.
            if r0 <= 1e-12 and r1 <= 1e-12:
                continue
            if r0 <= 1e-12:
                faces.append((a + 1, b + 1, c + 1))
            elif r1 <= 1e-12:
                faces.append((a + 1, b + 1, d + 1))
            else:
                faces.append((a + 1, b + 1, c + 1))
                faces.append((a + 1, c + 1, d + 1))

    return verts, faces


def svg_point_to_local(
    p: Point2,
    center: Point2,
    scale: float,
    y_flip: bool = True,
) -> Point2:
    x = (p[0] - center[0]) * scale
    y = (center[1] - p[1]) * scale if y_flip else (p[1] - center[1]) * scale
    return (x, y)


def map_contours_to_surface(
    contours: Sequence[Sequence[Point2]],
    contour_center: Point2,
    contour_scale: float,
    profile_s: Sequence[float],
    profile_r: Sequence[float],
    profile_z: Sequence[float],
    mapping: str,
    base_gap_frac: float,
    top_gap_frac: float,
    height_exponent: float,
) -> List[List[Point3]]:
    local_contours, rho_max = contour_polys_to_local(contours, contour_center, contour_scale)

    s_max = profile_s[-1] if profile_s else 0.0
    area_profile = build_area_cumulative(profile_s, profile_r) if mapping == "area" else None
    mapped: List[List[Point3]] = []

    for poly in local_contours:
        out: List[Point3] = []
        prev_theta = 0.0
        for x, y in poly:
            rho = math.hypot(x, y)
            if rho <= 1e-12:
                theta = prev_theta
            else:
                theta = math.atan2(y, x)
                prev_theta = theta
            s = map_rho_to_s(
                rho,
                rho_max,
                s_max,
                mapping,
                profile_s,
                area_profile,
                base_gap_frac=base_gap_frac,
                top_gap_frac=top_gap_frac,
                height_exponent=height_exponent,
            )
            r = interp_1d(profile_s, profile_r, s)
            z = interp_1d(profile_s, profile_z, s)
            out.append((r * math.cos(theta), r * math.sin(theta), z))
        if len(out) >= 2:
            mapped.append(out)

    return mapped


def build_contour_ribbons(
    contour_lines: Sequence[Sequence[Point3]],
    width: float,
    lift: float,
) -> Tuple[List[Point3], List[Tuple[int, int, int]]]:
    ribbon_vertices: List[Point3] = []
    ribbon_faces: List[Tuple[int, int, int]] = []
    if width <= 0.0:
        return ribbon_vertices, ribbon_faces

    half = 0.5 * width
    z_axis: Point3 = (0.0, 0.0, 1.0)
    y_axis: Point3 = (0.0, 1.0, 0.0)

    for line in contour_lines:
        if len(line) < 2:
            continue

        n = len(line)
        tangents: List[Point3] = []
        outwards: List[Point3] = []
        sides: List[Point3] = []

        prev_outward: Point3 = (1.0, 0.0, 0.0)
        for i, p in enumerate(line):
            rx, ry, _ = p
            radial = (rx, ry, 0.0)
            if v_norm(radial) > 1e-12:
                prev_outward = v_unit(radial, prev_outward)
            outwards.append(prev_outward)

            if i == 0:
                t = v_sub(line[1], line[0])
            elif i == n - 1:
                t = v_sub(line[-1], line[-2])
            else:
                t = v_sub(line[i + 1], line[i - 1])
            tangents.append(v_unit(t, (0.0, 0.0, 1.0)))

        for i in range(n):
            side = v_cross(tangents[i], outwards[i])
            if v_norm(side) <= 1e-12:
                side = v_cross(tangents[i], z_axis)
            if v_norm(side) <= 1e-12:
                side = v_cross(tangents[i], y_axis)
            sides.append(v_unit(side, (1.0, 0.0, 0.0)))

        base_index = len(ribbon_vertices) + 1
        for i in range(n):
            p = line[i]
            offset = v_add(v_mul(outwards[i], lift), v_mul(sides[i], half))
            left = v_add(p, offset)
            right = v_add(p, v_add(v_mul(outwards[i], lift), v_mul(sides[i], -half)))
            ribbon_vertices.append(left)
            ribbon_vertices.append(right)

        for i in range(n - 1):
            a = base_index + 2 * i
            b = base_index + 2 * i + 1
            c = base_index + 2 * (i + 1)
            d = base_index + 2 * (i + 1) + 1
            ribbon_faces.append((a, b, c))
            ribbon_faces.append((b, d, c))

    return ribbon_vertices, ribbon_faces


def write_obj(
    output_path: Path,
    surface_vertices: Sequence[Point3],
    faces: Sequence[Tuple[int, int, int]],
    contour_lines: Sequence[Sequence[Point3]],
    contour_render: str,
    contour_width: float,
    contour_lift: float,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Generated by map_contours_to_cup.py\n")
        f.write("g cup_surface\n")
        for x, y, z in surface_vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a} {b} {c}\n")

        v_offset = len(surface_vertices)
        if contour_render in ("lines", "both"):
            for i, line in enumerate(contour_lines, start=1):
                f.write(f"g contour_line_{i:03d}\n")
                start = v_offset + 1
                for x, y, z in line:
                    f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                    v_offset += 1
                indices = " ".join(str(k) for k in range(start, v_offset + 1))
                f.write(f"l {indices}\n")

        if contour_render in ("ribbon", "both"):
            ribbon_vertices, ribbon_faces = build_contour_ribbons(
                contour_lines, width=contour_width, lift=contour_lift
            )
            if ribbon_vertices:
                f.write("g contour_ribbons\n")
                start = v_offset
                for x, y, z in ribbon_vertices:
                    f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                    v_offset += 1
                for a, b, c in ribbon_faces:
                    f.write(f"f {start + a} {start + b} {start + c}\n")


def parse_center_arg(value: Optional[str], fallback: Point2) -> Point2:
    if not value:
        return fallback
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise ValueError("Center must be in 'x,y' format.")
    return (float(parts[0]), float(parts[1]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Map contour SVG polylines onto a cup surface generated by revolving a profile SVG polyline."
    )
    parser.add_argument("--contours", required=True, type=Path, help="Input contours SVG file.")
    parser.add_argument("--profile", required=True, type=Path, help="Input cup profile SVG file.")
    parser.add_argument("--output", required=True, type=Path, help="Output OBJ path.")
    parser.add_argument(
        "--mapping",
        choices=("linear", "area"),
        default="linear",
        help="rho->meridian mapping mode (default: linear).",
    )
    parser.add_argument(
        "--contour-center",
        default=None,
        help="Center of contour image as 'x,y'. Defaults to SVG viewBox center.",
    )
    parser.add_argument(
        "--contour-scale",
        type=float,
        default=1.0,
        help="Scale factor from contour SVG units to model units.",
    )
    parser.add_argument(
        "--profile-axis-x",
        type=float,
        default=None,
        help="X coordinate of rotation axis in profile SVG. Defaults to left-most profile point.",
    )
    parser.add_argument(
        "--profile-scale",
        type=float,
        default=1.0,
        help="Scale factor from profile SVG units to model units.",
    )
    parser.add_argument(
        "--profile-index",
        type=int,
        default=0,
        help="Polyline index among sorted candidate profile polylines (0=longest).",
    )
    parser.add_argument(
        "--profile-samples",
        type=int,
        default=300,
        help="Number of samples along profile for interpolation and meshing.",
    )
    parser.add_argument(
        "--theta-segments",
        type=int,
        default=128,
        help="Angular segments for revolved mesh.",
    )
    parser.add_argument(
        "--min-contour-length",
        type=float,
        default=1.0,
        help="Drop contour polylines shorter than this SVG-unit length.",
    )
    parser.add_argument(
        "--contour-smooth-iterations",
        type=int,
        default=2,
        help="Laplacian smoothing passes for contour polylines (0 disables).",
    )
    parser.add_argument(
        "--contour-smooth-strength",
        type=float,
        default=0.5,
        help="Blend [0..1] per smoothing pass for contour polylines.",
    )
    parser.add_argument(
        "--contour-render",
        choices=("ribbon", "lines", "both", "none"),
        default="ribbon",
        help="How to export mapped contours (default: ribbon).",
    )
    parser.add_argument(
        "--contour-width",
        type=float,
        default=0.0,
        help="Ribbon width in model units (<=0 means auto).",
    )
    parser.add_argument(
        "--contour-lift",
        type=float,
        default=0.0,
        help="Outward offset above cup surface in model units (<=0 means auto).",
    )
    parser.add_argument(
        "--base-gap-frac",
        type=float,
        default=0.0,
        help="Fraction [0..0.95] of cup side height reserved as blank gap from the base.",
    )
    parser.add_argument(
        "--top-gap-frac",
        type=float,
        default=0.12,
        help="Fraction [0..0.95] of cup side height reserved as blank gap below the rim.",
    )
    parser.add_argument(
        "--height-exponent",
        type=float,
        default=1.0,
        help="Radial-to-height bias (>0). <1 pushes lines higher; >1 pulls lines lower.",
    )
    parser.add_argument(
        "--steepness-compensation",
        type=float,
        default=0.85,
        help="Blend [0..1] from arclength mapping (0) to vertical-height mapping (1) to reduce spacing on steeper profile regions.",
    )
    parser.add_argument(
        "--dip-layer-thickness",
        type=float,
        default=0.0,
        help="Peak thickness at each contour centerline (model units).",
    )
    parser.add_argument(
        "--dip-taper-width",
        type=float,
        default=0.0,
        help="Distance from contour centerline where dip thickness tapers to zero (model units, <=0 means auto).",
    )
    args = parser.parse_args(argv)
    args.base_gap_frac = max(0.0, min(0.95, args.base_gap_frac))
    args.top_gap_frac = max(0.0, min(0.95, args.top_gap_frac))
    args.steepness_compensation = max(0.0, min(1.0, args.steepness_compensation))
    args.contour_smooth_iterations = max(0, args.contour_smooth_iterations)
    args.contour_smooth_strength = max(0.0, min(1.0, args.contour_smooth_strength))
    if args.base_gap_frac + args.top_gap_frac >= 0.98:
        raise ValueError("--base-gap-frac + --top-gap-frac must be < 0.98.")
    if args.height_exponent <= 0.0:
        raise ValueError("--height-exponent must be > 0.")

    contour_canvas, contour_items = load_svg_polylines(args.contours)
    profile_canvas, profile_items = load_svg_polylines(args.profile)

    contour_center = parse_center_arg(args.contour_center, contour_canvas.center)
    contour_polys = choose_contour_polylines(contour_items, args.min_contour_length)
    contour_polys = smooth_contour_polylines(
        contour_polys,
        iterations=args.contour_smooth_iterations,
        strength=args.contour_smooth_strength,
    )
    local_contours, rho_max = contour_polys_to_local(contour_polys, contour_center, args.contour_scale)
    contour_segments = build_contour_segments(local_contours)
    profile_poly = choose_profile_polyline(profile_items, args.profile_index)

    profile_rz = build_profile_rz(
        profile_poly,
        axis_x=args.profile_axis_x,
        y_flip=True,
        scale=args.profile_scale,
        samples=args.profile_samples,
    )
    profile_s = build_mapping_coordinate(profile_rz, args.steepness_compensation)
    profile_r = [p[0] for p in profile_rz]
    profile_z = [p[1] for p in profile_rz]

    surface_vertices, faces = make_surface_mesh(profile_rz, args.theta_segments)
    if args.dip_layer_thickness > 0.0:
        if args.dip_taper_width <= 0.0:
            args.dip_taper_width = max(1e-6, 4.0 * args.dip_layer_thickness)
        surface_vertices = apply_dip_layers_to_surface(
            surface_vertices,
            profile_rz=profile_rz,
            theta_segments=args.theta_segments,
            mapping=args.mapping,
            profile_s=profile_s,
            profile_r=profile_r,
            rho_max=rho_max,
            base_gap_frac=args.base_gap_frac,
            top_gap_frac=args.top_gap_frac,
            height_exponent=args.height_exponent,
            contour_segments=contour_segments,
            dip_layer_thickness=args.dip_layer_thickness,
            dip_taper_width=args.dip_taper_width,
        )
    mapped_lines = map_contours_to_surface(
        contour_polys,
        contour_center=contour_center,
        contour_scale=args.contour_scale,
        profile_s=profile_s,
        profile_r=profile_r,
        profile_z=profile_z,
        mapping=args.mapping,
        base_gap_frac=args.base_gap_frac,
        top_gap_frac=args.top_gap_frac,
        height_exponent=args.height_exponent,
    )

    if not mapped_lines:
        raise ValueError("No mapped contour lines were produced.")

    model_radius = max(profile_r) if profile_r else 1.0
    contour_width = args.contour_width if args.contour_width > 0.0 else max(1e-4, 0.01 * model_radius)
    contour_lift = args.contour_lift if args.contour_lift > 0.0 else 0.35 * contour_width

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_obj(
        args.output,
        surface_vertices,
        faces,
        mapped_lines,
        contour_render=args.contour_render,
        contour_width=contour_width,
        contour_lift=contour_lift,
    )

    print(f"Contours loaded: {len(contour_polys)}")
    print(f"Profile canvas center (unused unless needed): {profile_canvas.center}")
    print(f"Surface vertices: {len(surface_vertices)}")
    print(f"Surface faces: {len(faces)}")
    print(f"Mapped contour lines: {len(mapped_lines)}")
    print(f"Base gap fraction: {args.base_gap_frac}")
    print(f"Top gap fraction: {args.top_gap_frac}")
    print(f"Height exponent: {args.height_exponent}")
    print(f"Steepness compensation: {args.steepness_compensation}")
    print(f"Contour smooth iterations: {args.contour_smooth_iterations}")
    print(f"Contour smooth strength: {args.contour_smooth_strength}")
    print(f"Dip contour segments: {len(contour_segments)}")
    print(f"Dip layer thickness: {args.dip_layer_thickness}")
    print(f"Dip taper width: {args.dip_taper_width}")
    print(f"Contour render mode: {args.contour_render}")
    if args.contour_render in ("ribbon", "both"):
        print(f"Contour ribbon width: {contour_width}")
        print(f"Contour ribbon lift: {contour_lift}")
    print(f"Wrote OBJ: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
