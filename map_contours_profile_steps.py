#!/usr/bin/env python3
"""
Map ordered SVG contour lines to stepped positions along an SVG profile line.

Behavior:
- The largest (outermost) contour is mapped to the top-most point of the profile.
- Each next contour is mapped farther down the profile by a fixed arc-length step.
- Contours are placed in 3D as XY polylines at the mapped Z height.
- Radius is taken from the profile (interpreted as YZ profile from SVG X/Y).
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
from typing import Dict, List, Optional, Sequence, Tuple


Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]


@dataclass
class SvgPolyline:
    points: List[Point2]
    stroked: bool
    filled: bool


def parse_number(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text.strip())
    if not match:
        return None
    return float(match.group(0))


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


def dedupe_consecutive(points: Sequence[Point2], eps: float = 1e-12) -> List[Point2]:
    if not points:
        return []
    out = [points[0]]
    for p in points[1:]:
        dx = p[0] - out[-1][0]
        dy = p[1] - out[-1][1]
        if dx * dx + dy * dy > eps:
            out.append(p)
    return out


def parse_points_attr(points_text: str) -> List[Point2]:
    nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", points_text)
    vals = [float(n) for n in nums]
    pts: List[Point2] = []
    for i in range(0, len(vals) - 1, 2):
        pts.append((vals[i], vals[i + 1]))
    return dedupe_consecutive(pts)


def tokenize_path_d(d: str) -> List[str]:
    token_re = re.compile(r"[A-Za-z]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
    return token_re.findall(d)


def bezier_cubic(p0: Point2, p1: Point2, p2: Point2, p3: Point2, t: float) -> Point2:
    u = 1.0 - t
    b0 = u * u * u
    b1 = 3.0 * u * u * t
    b2 = 3.0 * u * t * t
    b3 = t * t * t
    return (
        b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0],
        b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1],
    )


def bezier_quad(p0: Point2, p1: Point2, p2: Point2, t: float) -> Point2:
    u = 1.0 - t
    b0 = u * u
    b1 = 2.0 * u * t
    b2 = t * t
    return (b0 * p0[0] + b1 * p1[0] + b2 * p2[0], b0 * p0[1] + b1 * p1[1] + b2 * p2[1])


def parse_path_d(d: str, curve_samples: int) -> List[List[Point2]]:
    tokens = tokenize_path_d(d)
    i = 0
    cmd: Optional[str] = None
    current: Point2 = (0.0, 0.0)
    start: Optional[Point2] = None
    curr_poly: List[Point2] = []
    polys: List[List[Point2]] = []
    prev_cmd: Optional[str] = None
    prev_cubic_ctrl: Optional[Point2] = None
    prev_quad_ctrl: Optional[Point2] = None

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
                nxt = (current[0] + x, current[1] + y) if cmd == "m" else (x, y)
                current = nxt
                if first:
                    flush_poly()
                    curr_poly = [current]
                    start = current
                    first = False
                else:
                    curr_poly.append(current)
            cmd = "L" if cmd == "M" else "l"
            prev_cmd = "M"
            prev_cubic_ctrl = None
            prev_quad_ctrl = None
            continue

        if cmd in ("L", "l"):
            while i + 1 < len(tokens) and not is_cmd(tokens[i]) and not is_cmd(tokens[i + 1]):
                x = float(tokens[i])
                y = float(tokens[i + 1])
                i += 2
                current = (current[0] + x, current[1] + y) if cmd == "l" else (x, y)
                if not curr_poly:
                    curr_poly = [current]
                    start = current
                else:
                    curr_poly.append(current)
            prev_cmd = "L"
            prev_cubic_ctrl = None
            prev_quad_ctrl = None
            continue

        if cmd in ("H", "h"):
            while i < len(tokens) and not is_cmd(tokens[i]):
                x = float(tokens[i])
                i += 1
                current = (current[0] + x, current[1]) if cmd == "h" else (x, current[1])
                if not curr_poly:
                    curr_poly = [current]
                    start = current
                else:
                    curr_poly.append(current)
            prev_cmd = "H"
            prev_cubic_ctrl = None
            prev_quad_ctrl = None
            continue

        if cmd in ("V", "v"):
            while i < len(tokens) and not is_cmd(tokens[i]):
                y = float(tokens[i])
                i += 1
                current = (current[0], current[1] + y) if cmd == "v" else (current[0], y)
                if not curr_poly:
                    curr_poly = [current]
                    start = current
                else:
                    curr_poly.append(current)
            prev_cmd = "V"
            prev_cubic_ctrl = None
            prev_quad_ctrl = None
            continue

        if cmd in ("C", "c"):
            while i + 5 < len(tokens) and not any(is_cmd(tokens[i + k]) for k in range(6)):
                x1, y1, x2, y2, x3, y3 = (float(tokens[i + k]) for k in range(6))
                i += 6
                p1 = (current[0] + x1, current[1] + y1) if cmd == "c" else (x1, y1)
                p2 = (current[0] + x2, current[1] + y2) if cmd == "c" else (x2, y2)
                p3 = (current[0] + x3, current[1] + y3) if cmd == "c" else (x3, y3)
                if not curr_poly:
                    curr_poly = [current]
                    start = current
                for step in range(1, max(2, curve_samples) + 1):
                    t = step / max(2, curve_samples)
                    curr_poly.append(bezier_cubic(current, p1, p2, p3, t))
                current = p3
                prev_cubic_ctrl = p2
            prev_cmd = "C"
            prev_quad_ctrl = None
            continue

        if cmd in ("S", "s"):
            while i + 3 < len(tokens) and not any(is_cmd(tokens[i + k]) for k in range(4)):
                x2, y2, x3, y3 = (float(tokens[i + k]) for k in range(4))
                i += 4
                if prev_cmd in ("C", "S") and prev_cubic_ctrl is not None:
                    p1 = (2.0 * current[0] - prev_cubic_ctrl[0], 2.0 * current[1] - prev_cubic_ctrl[1])
                else:
                    p1 = current
                p2 = (current[0] + x2, current[1] + y2) if cmd == "s" else (x2, y2)
                p3 = (current[0] + x3, current[1] + y3) if cmd == "s" else (x3, y3)
                if not curr_poly:
                    curr_poly = [current]
                    start = current
                for step in range(1, max(2, curve_samples) + 1):
                    t = step / max(2, curve_samples)
                    curr_poly.append(bezier_cubic(current, p1, p2, p3, t))
                current = p3
                prev_cubic_ctrl = p2
            prev_cmd = "S"
            prev_quad_ctrl = None
            continue

        if cmd in ("Q", "q"):
            while i + 3 < len(tokens) and not any(is_cmd(tokens[i + k]) for k in range(4)):
                x1, y1, x2, y2 = (float(tokens[i + k]) for k in range(4))
                i += 4
                p1 = (current[0] + x1, current[1] + y1) if cmd == "q" else (x1, y1)
                p2 = (current[0] + x2, current[1] + y2) if cmd == "q" else (x2, y2)
                if not curr_poly:
                    curr_poly = [current]
                    start = current
                for step in range(1, max(2, curve_samples) + 1):
                    t = step / max(2, curve_samples)
                    curr_poly.append(bezier_quad(current, p1, p2, t))
                current = p2
                prev_quad_ctrl = p1
            prev_cmd = "Q"
            prev_cubic_ctrl = None
            continue

        if cmd in ("T", "t"):
            while i + 1 < len(tokens) and not any(is_cmd(tokens[i + k]) for k in range(2)):
                x2, y2 = (float(tokens[i + k]) for k in range(2))
                i += 2
                if prev_cmd in ("Q", "T") and prev_quad_ctrl is not None:
                    p1 = (2.0 * current[0] - prev_quad_ctrl[0], 2.0 * current[1] - prev_quad_ctrl[1])
                else:
                    p1 = current
                p2 = (current[0] + x2, current[1] + y2) if cmd == "t" else (x2, y2)
                if not curr_poly:
                    curr_poly = [current]
                    start = current
                for step in range(1, max(2, curve_samples) + 1):
                    t = step / max(2, curve_samples)
                    curr_poly.append(bezier_quad(current, p1, p2, t))
                current = p2
                prev_quad_ctrl = p1
            prev_cmd = "T"
            prev_cubic_ctrl = None
            continue

        if cmd in ("Z", "z"):
            if curr_poly and start is not None:
                if curr_poly[-1] != start:
                    curr_poly.append(start)
            flush_poly()
            start = None
            prev_cmd = "Z"
            prev_cubic_ctrl = None
            prev_quad_ctrl = None
            continue

        i += 1

    flush_poly()
    return [dedupe_consecutive(poly) for poly in polys if len(poly) >= 2]


def load_svg_polylines(svg_path: Path, curve_samples: int) -> List[SvgPolyline]:
    tree = ET.parse(svg_path)
    root = tree.getroot()

    items: List[SvgPolyline] = []

    for elem in root.iter():
        tag = elem.tag.split("}", 1)[-1]
        stroked, filled = element_paint_flags(elem)
        if tag == "path":
            d = elem.attrib.get("d")
            if not d:
                continue
            for poly in parse_path_d(d, curve_samples=curve_samples):
                if len(poly) >= 2:
                    items.append(SvgPolyline(points=poly, stroked=stroked, filled=filled))
        elif tag in ("polyline", "polygon"):
            pts = parse_points_attr(elem.attrib.get("points", ""))
            if tag == "polygon" and pts and pts[0] != pts[-1]:
                pts = pts + [pts[0]]
            if len(pts) >= 2:
                items.append(SvgPolyline(points=pts, stroked=stroked, filled=filled))
    return items


def polyline_length(points: Sequence[Point2]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        total += math.hypot(dx, dy)
    return total


def cumulative_lengths(points: Sequence[Point2]) -> List[float]:
    out = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        out.append(out[-1] + math.hypot(dx, dy))
    return out


def interp_on_polyline(points: Sequence[Point2], cumlen: Sequence[float], s: float) -> Point2:
    if not points:
        return (0.0, 0.0)
    if len(points) == 1:
        return points[0]
    if s <= 0.0:
        return points[0]
    if s >= cumlen[-1]:
        return points[-1]

    idx = bisect.bisect_right(cumlen, s) - 1
    idx = max(0, min(idx, len(points) - 2))
    s0, s1 = cumlen[idx], cumlen[idx + 1]
    if s1 <= s0:
        return points[idx]
    t = (s - s0) / (s1 - s0)
    x = points[idx][0] * (1.0 - t) + points[idx + 1][0] * t
    y = points[idx][1] * (1.0 - t) + points[idx + 1][1] * t
    return (x, y)


def ensure_closed(points: Sequence[Point2], eps: float = 1e-9) -> List[Point2]:
    if not points:
        return []
    if len(points) < 2:
        return list(points)
    dx = points[0][0] - points[-1][0]
    dy = points[0][1] - points[-1][1]
    if dx * dx + dy * dy <= eps * eps:
        return list(points)
    return list(points) + [points[0]]


def polygon_area(points: Sequence[Point2]) -> float:
    poly = ensure_closed(points)
    if len(poly) < 4:
        return 0.0
    s = 0.0
    for i in range(len(poly) - 1):
        x0, y0 = poly[i]
        x1, y1 = poly[i + 1]
        s += x0 * y1 - x1 * y0
    return 0.5 * s


def polygon_centroid(points: Sequence[Point2]) -> Point2:
    poly = ensure_closed(points)
    if len(poly) < 4:
        if not points:
            return (0.0, 0.0)
        sx = sum(p[0] for p in points)
        sy = sum(p[1] for p in points)
        n = float(len(points))
        return (sx / n, sy / n)

    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(len(poly) - 1):
        x0, y0 = poly[i]
        x1, y1 = poly[i + 1]
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    if abs(area2) < 1e-12:
        sx = sum(p[0] for p in points)
        sy = sum(p[1] for p in points)
        n = float(len(points))
        return (sx / n, sy / n)

    return (cx / (3.0 * area2), cy / (3.0 * area2))


def choose_profile_polyline(polys: Sequence[SvgPolyline]) -> List[Point2]:
    candidates = [p.points for p in polys if p.stroked and len(p.points) >= 2]
    if not candidates:
        candidates = [p.points for p in polys if len(p.points) >= 2]
    if not candidates:
        raise ValueError("No usable profile polyline found.")
    candidates.sort(key=polyline_length, reverse=True)
    return dedupe_consecutive(candidates[0])


def choose_contours(polys: Sequence[SvgPolyline], min_len: float) -> List[List[Point2]]:
    candidates: List[List[Point2]] = []
    for item in polys:
        if not item.stroked:
            continue
        pts = dedupe_consecutive(item.points)
        if len(pts) < 3:
            continue
        if polyline_length(pts) < min_len:
            continue
        candidates.append(ensure_closed(pts))
    if not candidates:
        raise ValueError("No usable contour polylines found.")

    candidates.sort(key=lambda p: abs(polygon_area(p)), reverse=True)
    return candidates


def orient_profile_from_top(profile_yz: Sequence[Point2]) -> List[Point2]:
    if len(profile_yz) < 2:
        return list(profile_yz)

    top_idx = max(range(len(profile_yz)), key=lambda i: profile_yz[i][1])
    if top_idx == 0:
        return list(profile_yz)
    if top_idx == len(profile_yz) - 1:
        return list(reversed(profile_yz))

    left = list(reversed(profile_yz[: top_idx + 1]))
    right = list(profile_yz[top_idx:])

    def score(path: Sequence[Point2]) -> Tuple[int, float, float]:
        drop = path[0][1] - path[-1][1]
        downhill = 1 if drop > 0.0 else 0
        return (downhill, drop, polyline_length(path))

    return left if score(left) >= score(right) else right


def build_profile_yz(
    profile_points_svg: Sequence[Point2],
    axis_x: Optional[float],
    profile_scale: float,
    y_flip: bool,
) -> List[Point2]:
    if len(profile_points_svg) < 2:
        raise ValueError("Profile must contain at least two points.")

    xs = [p[0] for p in profile_points_svg]
    ys = [p[1] for p in profile_points_svg]
    axis = min(xs) if axis_x is None else axis_x
    y_ref = max(ys) if y_flip else min(ys)

    yz: List[Point2] = []
    for x, y in profile_points_svg:
        radius = max(0.0, (x - axis) * profile_scale)
        z = (y_ref - y) * profile_scale if y_flip else (y - y_ref) * profile_scale
        yz.append((radius, z))

    yz = dedupe_consecutive(yz)
    if len(yz) < 2:
        raise ValueError("Profile collapsed to too few unique points after conversion.")
    return orient_profile_from_top(yz)


def contour_max_radius(points: Sequence[Point2], center: Point2) -> float:
    if not points:
        return 0.0
    return max(math.hypot(p[0] - center[0], p[1] - center[1]) for p in points)


def map_contours_to_profile_steps(
    contours: Sequence[Sequence[Point2]],
    profile_yz: Sequence[Point2],
    step_distance: float,
    start_offset: float,
    contour_scale: float,
    overflow: str,
) -> Tuple[List[List[Point3]], List[Tuple[int, float, float, float]]]:
    if step_distance < 0.0:
        raise ValueError("step_distance must be >= 0.")
    if start_offset < 0.0:
        raise ValueError("start_offset must be >= 0.")
    if contour_scale <= 0.0:
        raise ValueError("contour_scale must be > 0.")

    profile_cum = cumulative_lengths(profile_yz)
    max_s = profile_cum[-1]

    centers = [polygon_centroid(c) for c in contours]
    outer_ref = contour_max_radius(contours[0], centers[0]) if contours else 0.0
    if outer_ref <= 0.0:
        raise ValueError("Outermost contour has zero radius; cannot scale contours.")

    mapped: List[List[Point3]] = []
    summary: List[Tuple[int, float, float, float]] = []

    for idx, contour in enumerate(contours):
        s = start_offset + idx * step_distance
        if s > max_s:
            if overflow == "stop":
                break
            s = max_s

        radius_target, z_target = interp_on_polyline(profile_yz, profile_cum, s)
        center = centers[idx]
        factor = contour_scale * radius_target / outer_ref

        out: List[Point3] = []
        for x, y in contour:
            out.append(((x - center[0]) * factor, (y - center[1]) * factor, z_target))

        mapped.append(out)
        summary.append((idx, s, radius_target, z_target))

    return mapped, summary


def v_sub(a: Point3, b: Point3) -> Point3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_add(a: Point3, b: Point3) -> Point3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_mul(a: Point3, k: float) -> Point3:
    return (a[0] * k, a[1] * k, a[2] * k)


def v_norm(a: Point3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def v_unit(a: Point3, fallback: Point3 = (1.0, 0.0, 0.0)) -> Point3:
    n = v_norm(a)
    if n <= 1e-12:
        return fallback
    return (a[0] / n, a[1] / n, a[2] / n)


def build_contour_ribbons(
    contour_lines: Sequence[Sequence[Point3]], width: float, lift: float
) -> Tuple[List[Point3], List[Tuple[int, int, int]]]:
    ribbon_vertices: List[Point3] = []
    ribbon_faces: List[Tuple[int, int, int]] = []
    if width <= 0.0:
        return ribbon_vertices, ribbon_faces

    half = 0.5 * width
    for line in contour_lines:
        if len(line) < 2:
            continue

        closed = (
            len(line) >= 3
            and (line[0][0] - line[-1][0]) ** 2 + (line[0][1] - line[-1][1]) ** 2 + (line[0][2] - line[-1][2]) ** 2
            <= 1e-12
        )
        core = list(line[:-1]) if closed else list(line)
        n = len(core)
        if n < 2:
            continue

        start = len(ribbon_vertices)
        for i in range(n):
            p = core[i]
            if closed:
                prev_p = core[(i - 1) % n]
                next_p = core[(i + 1) % n]
            else:
                prev_p = core[i - 1] if i > 0 else core[i]
                next_p = core[i + 1] if i < n - 1 else core[i]

            tangent = v_unit(v_sub(next_p, prev_p), fallback=(1.0, 0.0, 0.0))
            side = v_unit((-tangent[1], tangent[0], 0.0), fallback=(1.0, 0.0, 0.0))
            offset = v_mul(side, half)
            lift_vec = (0.0, 0.0, lift)

            left = v_add(v_sub(p, offset), lift_vec)
            right = v_add(v_add(p, offset), lift_vec)
            ribbon_vertices.append(left)
            ribbon_vertices.append(right)

        span = n if closed else (n - 1)
        for i in range(span):
            j = (i + 1) % n
            a = start + 2 * i
            b = start + 2 * i + 1
            c = start + 2 * j
            d = start + 2 * j + 1
            ribbon_faces.append((a, b, c))
            ribbon_faces.append((b, d, c))

    return ribbon_vertices, ribbon_faces


def write_obj(
    output_path: Path,
    contour_lines: Sequence[Sequence[Point3]],
    render: str,
    line_width: float,
    line_lift: float,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Mapped contour lines (profile-stepped)\n")
        v_offset = 1
        if render in ("lines", "both"):
            for line_idx, line in enumerate(contour_lines):
                f.write(f"g contour_line_{line_idx:04d}\n")
                for x, y, z in line:
                    f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                ids = [str(v_offset + i) for i in range(len(line))]
                if len(ids) >= 2:
                    f.write(f"l {' '.join(ids)}\n")
                v_offset += len(line)

        if render in ("ribbon", "both"):
            ribbon_vertices, ribbon_faces = build_contour_ribbons(contour_lines, width=line_width, lift=line_lift)
            if ribbon_vertices:
                f.write("g contour_ribbons\n")
                start = v_offset
                for x, y, z in ribbon_vertices:
                    f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                    v_offset += 1
                for a, b, c in ribbon_faces:
                    f.write(f"f {start + a} {start + b} {start + c}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Map contour SVG lines to profile positions by fixed step distance. "
            "Outermost contour starts at profile top."
        )
    )
    parser.add_argument("--contours", required=True, type=Path, help="Input contour SVG.")
    parser.add_argument("--profile", required=True, type=Path, help="Input profile SVG.")
    parser.add_argument("--output", required=True, type=Path, help="Output OBJ path.")
    parser.add_argument(
        "--step-distance",
        required=True,
        type=float,
        help="Distance (model units) between mapped contour levels along the profile polyline.",
    )
    parser.add_argument(
        "--start-offset",
        type=float,
        default=0.0,
        help="Initial distance from profile top to place the outermost contour.",
    )
    parser.add_argument(
        "--overflow",
        choices=("stop", "clamp"),
        default="stop",
        help="When contours exceed profile length: stop writing, or clamp at profile end.",
    )
    parser.add_argument(
        "--contour-scale",
        type=float,
        default=1.0,
        help="Extra multiplier on mapped contour size.",
    )
    parser.add_argument(
        "--profile-axis-x",
        type=float,
        default=None,
        help="Axis X in profile SVG; defaults to profile min X.",
    )
    parser.add_argument(
        "--profile-scale",
        type=float,
        default=1.0,
        help="Scale from profile SVG units to model units.",
    )
    parser.add_argument(
        "--no-y-flip",
        action="store_true",
        help="Disable SVG-Y inversion when building profile Z.",
    )
    parser.add_argument(
        "--min-contour-length",
        type=float,
        default=1.0,
        help="Ignore contour paths shorter than this SVG-unit length.",
    )
    parser.add_argument(
        "--max-contours",
        type=int,
        default=0,
        help="Optional cap on number of contours to map (0 = all).",
    )
    parser.add_argument(
        "--curve-samples",
        type=int,
        default=24,
        help="Number of line samples used to flatten each curve segment in SVG paths.",
    )
    parser.add_argument(
        "--render",
        choices=("ribbon", "lines", "both"),
        default="ribbon",
        help="OBJ output mode (default: ribbon for FreeCAD visibility).",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=0.0,
        help="Ribbon width in model units (<=0 means auto).",
    )
    parser.add_argument(
        "--line-lift",
        type=float,
        default=0.0,
        help="Optional Z lift applied to ribbons.",
    )
    args = parser.parse_args(argv)

    if args.step_distance < 0.0:
        raise ValueError("--step-distance must be >= 0.")
    if args.start_offset < 0.0:
        raise ValueError("--start-offset must be >= 0.")
    if args.profile_scale <= 0.0:
        raise ValueError("--profile-scale must be > 0.")
    if args.curve_samples < 2:
        raise ValueError("--curve-samples must be >= 2.")

    contour_items = load_svg_polylines(args.contours, curve_samples=args.curve_samples)
    profile_items = load_svg_polylines(args.profile, curve_samples=args.curve_samples)

    contours = choose_contours(contour_items, min_len=args.min_contour_length)
    if args.max_contours > 0:
        contours = contours[: args.max_contours]

    profile_svg_points = choose_profile_polyline(profile_items)
    profile_yz = build_profile_yz(
        profile_svg_points,
        axis_x=args.profile_axis_x,
        profile_scale=args.profile_scale,
        y_flip=not args.no_y_flip,
    )

    mapped, summary = map_contours_to_profile_steps(
        contours=contours,
        profile_yz=profile_yz,
        step_distance=args.step_distance,
        start_offset=args.start_offset,
        contour_scale=args.contour_scale,
        overflow=args.overflow,
    )

    if not mapped:
        raise ValueError("No contours were mapped. Check step/offset/overflow settings.")

    max_r = max((math.hypot(p[0], p[1]) for line in mapped for p in line), default=1.0)
    line_width = args.line_width if args.line_width > 0.0 else max(1e-4, 0.008 * max_r)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_obj(
        args.output,
        mapped,
        render=args.render,
        line_width=line_width,
        line_lift=args.line_lift,
    )

    profile_len = cumulative_lengths(profile_yz)[-1]
    print(f"Loaded contours: {len(contours)}")
    print(f"Mapped contours: {len(mapped)}")
    print(f"Profile length: {profile_len:.6f}")
    print(f"Top profile point (r,z): ({profile_yz[0][0]:.6f}, {profile_yz[0][1]:.6f})")
    for idx, s, r, z in summary[: min(8, len(summary))]:
        print(f"Contour {idx}: s={s:.6f}, r={r:.6f}, z={z:.6f}")
    if len(summary) > 8:
        print(f"... {len(summary) - 8} more contour mappings")
    print(f"Render mode: {args.render}")
    if args.render in ("ribbon", "both"):
        print(f"Ribbon width: {line_width:.6f}")
        print(f"Ribbon lift: {args.line_lift:.6f}")
    print(f"Wrote OBJ: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
