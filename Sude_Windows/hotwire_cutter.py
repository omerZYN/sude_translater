#!/usr/bin/env python3
"""
Sude - HotWire Foam Cutter
DXF to G-code (Grbl HotWire 6.5 XYZA) - 4-axis hotwire foam cutting G-code generator.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import ezdxf
from datetime import datetime
import os
import json
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Machine limits (70cm x 40cm hareket alani)
MAX_X = 700.0   # Sol carriage yatay (mm)
MAX_Z = 700.0   # Sag carriage yatay (mm)
MAX_Y = 400.0   # Sol carriage dikey (mm)
MAX_A = 400.0   # Sag carriage dikey (mm)

DEFAULT_FEED = 300
DEFAULT_SPAN = 1000.0
DEFAULT_FOAM_WIDTH = 300.0
DEFAULT_FOAM_LEFT_OFFSET = 350.0
DEFAULT_KERF = 0.0
DEFAULT_SAFE_Y = 7.35
DEFAULT_NUM_POINTS = 200

# Persistent user settings — last entered values are remembered across sessions
SETTINGS_FILE = os.path.expanduser("~/.sude_hotwire_settings.json")


def _flatten_entity(entity):
    """Flatten a DXF entity to a list of (x, y) tuples."""
    t = entity.dxftype()
    if t == "LINE":
        return [
            (entity.dxf.start.x, entity.dxf.start.y),
            (entity.dxf.end.x, entity.dxf.end.y),
        ]
    elif t in ("ARC", "ELLIPSE", "SPLINE"):
        return [(p.x, p.y) for p in entity.flattening(0.1)]
    elif t == "LWPOLYLINE":
        return list(entity.get_points(format="xy"))
    elif t == "POLYLINE":
        return [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
    return []


def _chain_entities(segments, tol=0.5):
    """Chain a list of point-segments into a single ordered point list.
    Each segment is a list of (x,y) points. We find the order that connects
    end-to-end within tolerance and build one continuous loop."""
    if not segments:
        return []

    remaining = list(range(len(segments)))
    # Start with the longest segment (likely the spline / main profile)
    longest = max(remaining, key=lambda i: len(segments[i]))
    remaining.remove(longest)

    chain = list(segments[longest])

    while remaining:
        tail = np.array(chain[-1])
        best_idx = None
        best_flip = False
        best_dist = float("inf")

        for i in remaining:
            seg = segments[i]
            d_start = np.linalg.norm(tail - np.array(seg[0]))
            d_end = np.linalg.norm(tail - np.array(seg[-1]))
            if d_start < best_dist:
                best_dist = d_start
                best_idx = i
                best_flip = False
            if d_end < best_dist:
                best_dist = d_end
                best_idx = i
                best_flip = True

        if best_dist > tol * 50:  # No nearby segment found
            break

        remaining.remove(best_idx)
        seg = segments[best_idx]
        if best_flip:
            seg = list(reversed(seg))
        # Skip first point if it overlaps with chain tail
        if np.linalg.norm(np.array(chain[-1]) - np.array(seg[0])) < tol:
            seg = seg[1:]
        chain.extend(seg)

    return chain


def extract_profile_from_dxf(filepath):
    """Extract closed polyline/spline coordinates from a DXF file."""
    doc = ezdxf.readfile(filepath)
    msp = doc.modelspace()
    points = []

    # Try LWPOLYLINE first (single closed entity)
    for entity in msp.query("LWPOLYLINE"):
        pts = list(entity.get_points(format="xy"))
        if entity.closed or (len(pts) > 2 and np.allclose(pts[0], pts[-1], atol=0.01)):
            points = pts
            break

    # Try POLYLINE
    if not points:
        for entity in msp.query("POLYLINE"):
            if entity.is_2d_polyline or entity.is_3d_polyline:
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
                if entity.is_closed or (len(pts) > 2 and np.allclose(pts[0], pts[-1], atol=0.01)):
                    points = pts
                    break

    # Try closed SPLINE
    if not points:
        for entity in msp.query("SPLINE"):
            if entity.closed:
                pts = [(p.x, p.y) for p in entity.flattening(0.1)]
                points = pts
                break

    # Try assembling a composite profile from all entities (SPLINE + LINE + ARC etc.)
    if not points:
        all_entities = list(msp)
        segments = []
        for entity in all_entities:
            seg = _flatten_entity(entity)
            if len(seg) >= 2:
                segments.append(seg)

        if segments:
            chain = _chain_entities(segments, tol=0.5)
            if len(chain) > 4:
                # Check if it forms a closed loop
                if np.linalg.norm(np.array(chain[0]) - np.array(chain[-1])) < 1.0:
                    points = chain

    if not points:
        raise ValueError("DXF dosyasinda kapali profil bulunamadi!")

    return np.array(points, dtype=float)


def normalize_profile(points):
    """Normalize profile so that LE is at X=0 and the profile bottom (min Y)
    rests on the Y=0 plane. This makes the default zero reference the motor
    side: hucum kenari X=0'da, profil tabani Y=0'da."""
    # Remove duplicate closing point if present
    if np.allclose(points[0], points[-1], atol=0.01):
        points = points[:-1]

    # Find leading edge (minimum X = most forward point)
    le_idx = np.argmin(points[:, 0])

    # Rotate array so leading edge is first
    points = np.roll(points, -le_idx, axis=0)

    # Shift: X from LE, Y from profile bottom (min Y across all points)
    x_shift = points[0, 0]
    y_shift = float(np.min(points[:, 1]))
    shift = np.array([x_shift, y_shift])
    points = points - shift

    return points, shift


def _resample_by_arc_length(polyline, n):
    """Sample n points evenly by arc length along an OPEN polyline.
    Returns an Nx2 array. Endpoint excluded so n points fit cleanly without
    duplicating the final vertex."""
    if len(polyline) < 2 or n <= 0:
        return np.tile(polyline[0], (max(n, 1), 1)) if len(polyline) else np.zeros((n, 2))
    deltas = np.diff(polyline, axis=0)
    seg_lens = np.sqrt(np.sum(deltas ** 2, axis=1))
    cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = float(cum_lens[-1])
    if total_len < 1e-9:
        return np.tile(polyline[0], (n, 1))
    target_lens = np.linspace(0, total_len, n, endpoint=False)
    x = np.interp(target_lens, cum_lens, polyline[:, 0])
    y = np.interp(target_lens, cum_lens, polyline[:, 1])
    return np.column_stack([x, y])


def interpolate_profile(points, num_points):
    """Split-at-TE arc-length resampling of a closed profile.

    Unlike plain chord-fraction interpolation (which fails for vertical
    edges) or plain arc-length (which doesn't align LE/TE features between
    different profiles), this hybrid:
      1. Splits the profile at its rightmost vertex (TE = argmax X).
      2. Resamples the lower surface (LE -> TE) with n_lower arc-length
         evenly-spaced points.
      3. Resamples the upper surface (TE -> LE) with n_upper arc-length
         evenly-spaced points.
      4. Concatenates the two halves.

    Result: index 0 is LE, index n_lower is TE, and both halves have points
    distributed evenly by arc length within their surface. For any two
    profiles in the same winding (CCW) this guarantees root[i] and tip[i]
    correspond to the same feature (LE at 0, TE at n_lower, and surface
    fractions in between), so the wire links them consistently even when
    the profiles have vertical edges or very different aspect ratios.

    EXPECTS CCW winding (call _ensure_ccw first).
    """
    if np.allclose(points[0], points[-1], atol=1e-6):
        points = points[:-1]

    te_idx = int(np.argmax(points[:, 0]))

    # CCW order: LE(0) -> ...lower... -> TE(te_idx) -> ...upper... -> LE
    lower = points[:te_idx + 1]                       # LE ... TE
    upper = np.vstack([points[te_idx:], points[0:1]]) # TE ... LE (closed back)

    n_lower = num_points // 2
    n_upper = num_points - n_lower

    lower_pts = _resample_by_arc_length(lower, n_lower)
    upper_pts = _resample_by_arc_length(upper, n_upper)

    return np.vstack([lower_pts, upper_pts])


def _ensure_ccw(points):
    """Return a copy of the points in counter-clockwise order.
    Reverses the array if the polygon's signed area is negative (CW)."""
    xs = points[:, 0]
    ys = points[:, 1]
    signed_area = 0.5 * float(
        np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys)
    )
    if signed_area < 0:  # CW -> reverse to get CCW
        return points[::-1].copy()
    return points


def _orient_cw(points):
    """Reverse array if needed so that the polygon winding is clockwise
    (signed area < 0 in Y-up conventions). Makes cut direction deterministic
    regardless of the DXF's original winding."""
    xs = points[:, 0]
    ys = points[:, 1]
    # Shoelace signed area: positive = CCW, negative = CW
    signed_area = 0.5 * float(
        np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys)
    )
    if signed_area > 0:
        points = points[::-1].copy()
    return points


def _roll_to_mid_lower(points):
    """Roll the point array so that the starting point (index 0) is the
    point on the LOWER surface closest to mid-chord (X = chord/2).

    Expects a CW-oriented profile produced by _orient_cw where the upper
    surface runs LE -> TE (indices 0..te_idx) and the lower surface runs
    TE -> LE (indices te_idx..n-1). Returns (rolled_points, start_idx)."""
    if len(points) < 4:
        return points, 0
    te_idx = int(np.argmax(points[:, 0]))
    x_min = float(points[:, 0].min())
    x_max = float(points[:, 0].max())
    x_mid = (x_min + x_max) / 2.0

    tail = points[te_idx:]
    if len(tail) == 0:
        return points, 0
    rel = int(np.argmin(np.abs(tail[:, 0] - x_mid)))
    start_idx = te_idx + rel
    rolled = np.roll(points, -start_idx, axis=0)
    return rolled, start_idx


def apply_kerf_offset(points, kerf):
    """Apply kerf (wire diameter) offset to the profile."""
    if kerf <= 0:
        return points

    n = len(points)
    offset_pts = np.zeros_like(points)

    for i in range(n):
        p_prev = points[(i - 1) % n]
        p_curr = points[i]
        p_next = points[(i + 1) % n]

        # Edge vectors
        v1 = p_curr - p_prev
        v2 = p_next - p_curr

        # Normals (pointing outward assuming CCW winding)
        n1 = np.array([-v1[1], v1[0]])
        n2 = np.array([-v2[1], v2[0]])

        len1 = np.linalg.norm(n1)
        len2 = np.linalg.norm(n2)

        if len1 > 1e-10:
            n1 /= len1
        if len2 > 1e-10:
            n2 /= len2

        # Average normal
        n_avg = n1 + n2
        n_len = np.linalg.norm(n_avg)
        if n_len > 1e-10:
            n_avg /= n_len

        # Offset inward (shrink profile) - sign depends on winding direction
        offset_pts[i] = p_curr + n_avg * (kerf / 2.0)

    return offset_pts


def apply_taper_projection(root_pts, tip_pts, span, foam_width, foam_left_offset):
    """Project foam-face coordinates to carriage positions using triangle similarity.

    The wire stretches across the full span, but the foam is narrower.
    We extrapolate the cut path outward to where the carriages actually are.

    root_pts: Nx2 array - profile coords at foam left face
    tip_pts:  Nx2 array - profile coords at foam right face
    span: total wire distance between carriages (mm)
    foam_width: width of foam block (mm)
    foam_left_offset: distance from left carriage to foam left face (mm)

    Returns: (left_carriage_pts, right_carriage_pts) as Nx2 arrays
    """
    # No projection needed when foam fills the entire span
    if np.isclose(foam_width, span):
        return root_pts.copy(), tip_pts.copy()

    delta = tip_pts - root_pts

    left_ratio = foam_left_offset / foam_width
    right_ratio = (span - foam_left_offset - foam_width) / foam_width

    left_carriage_pts = root_pts - delta * left_ratio
    right_carriage_pts = tip_pts + delta * right_ratio

    return left_carriage_pts, right_carriage_pts


def build_machine_toolpath(root_pts, tip_pts, params):
    """Apply kerf, X offsets, taper projection and Y/Z offsets to produce the
    final machine-coordinate toolpath for both carriages.

    Cut flow convention: the machine starts and ends at (0,0,0,0) = motor home.
    A LEAD-IN cut line goes from (0,0) straight to the first profile point,
    then the profile is cut clockwise, then a LEAD-OUT cut line retraces back
    to (0,0). Both the lead-in and lead-out are regular G01 cut moves at the
    feed rate (hotwire is always hot), and they cut a thin slot from home to
    the profile start.

    Returns a dict with:
      left:        Nx2 array, (X, Y) points for the left (root) carriage
      right:       Nx2 array, (A, Z) points for the right (tip) carriage
      home:        (0, 0, 0, 0) — constant, kept for symmetry with render code
      root_chord:  chord length after kerf, before taper projection
      tip_chord:   same for tip
      total_offset: kerf + wire tolerance
      feed:        feed rate (mm/min)
    Both input arrays are COPIED internally — the caller's arrays are preserved.
    """
    feed = params["feed_rate"]
    span = params.get("span", params["foam_width"])
    foam_width = params["foam_width"]
    foam_left_offset = params.get("foam_left_offset", 0.0)
    root_x_offset = params.get("root_x_offset", params.get("x_offset", 0.0))
    tip_x_offset = params.get("tip_x_offset", params.get("x_offset", 0.0))
    root_y_offset = params.get("root_y_offset", 0.0)
    tip_z_offset = params.get("tip_z_offset", 0.0)
    kerf = params["kerf"]
    wire_tol = params.get("wire_tolerance", 0.0)

    root = root_pts.copy().astype(float)
    tip = tip_pts.copy().astype(float)

    # Kerf + wire tolerance (systematic path offset)
    total_offset = kerf + wire_tol
    if total_offset > 0:
        root = apply_kerf_offset(root, total_offset)
        tip = apply_kerf_offset(tip, total_offset)

    # Chord lengths (post-kerf)
    root_chord = float(np.max(root[:, 0]) - np.min(root[:, 0]))
    tip_chord = float(np.max(tip[:, 0]) - np.min(tip[:, 0]))

    # X offsets are applied BEFORE taper projection so that sweep (tip shifted
    # relative to root in X) is correctly baked into delta = tip - root and
    # propagates through the triangle-similarity extrapolation. This is what
    # lets the user align trailing edges on a tapered wing by setting
    # tip_x_offset = root_chord - tip_chord (plus any root_x_offset).
    root[:, 0] += root_x_offset
    tip[:, 0] += tip_x_offset

    # Taper projection in the (now shifted) profile frame
    left, right = apply_taper_projection(
        root, tip, span, foam_width, foam_left_offset
    )
    left = left.copy()
    right = right.copy()

    # Y offsets are applied at the carriage level (post-projection). They just
    # lift each carriage's cut path vertically; they don't participate in the
    # taper extrapolation because the wire doesn't "tilt" in Y due to offsets.
    left[:, 1] += root_y_offset
    right[:, 1] += tip_z_offset

    return {
        "left": left,
        "right": right,
        "home": (0.0, 0.0, 0.0, 0.0),
        "root_chord": root_chord,
        "tip_chord": tip_chord,
        "total_offset": total_offset,
        "feed": feed,
    }


def generate_gcode(root_pts, tip_pts, params):
    """Generate G-code for 4-axis hotwire cutting."""
    span = params.get("span", params["foam_width"])
    foam_width = params["foam_width"]
    foam_left_offset = params.get("foam_left_offset", 0.0)
    root_x_offset = params.get("root_x_offset", params.get("x_offset", 0.0))
    tip_x_offset = params.get("tip_x_offset", params.get("x_offset", 0.0))
    root_y_offset = params.get("root_y_offset", 0.0)
    tip_z_offset = params.get("tip_z_offset", 0.0)
    kerf = params["kerf"]
    wire_tol = params.get("wire_tolerance", 0.0)
    profile_name = params.get("profile_name", "Profil")

    tp = build_machine_toolpath(root_pts, tip_pts, params)
    left = tp["left"]
    right = tp["right"]
    feed = tp["feed"]
    root_chord = tp["root_chord"]
    tip_chord = tp["tip_chord"]
    total_offset = tp["total_offset"]

    # Build G-code lines
    lines = []
    lines.append(f"(Sude HotWire Foam Cutter - {profile_name})")
    lines.append(f"(Root: {root_chord:.1f}mm, Tip: {tip_chord:.1f}mm)")
    lines.append(f"(Root X Offset: {root_x_offset:.1f}mm, Tip X Offset: {tip_x_offset:.1f}mm)")
    lines.append(f"(Root Y Offset: {root_y_offset:.1f}mm, Tip Z Offset: {tip_z_offset:.1f}mm)")
    lines.append(f"(Tel Mesafesi: {span:.1f}mm)")
    lines.append(f"(Kopuk Genisligi: {foam_width:.1f}mm)")
    lines.append(f"(Sol Bosluk: {foam_left_offset:.1f}mm, Sag Bosluk: {span - foam_left_offset - foam_width:.1f}mm)")
    lines.append(f"(Zero Point: Motor tarafi - X/A=0 ve Y/Z=0 her iki carriage icin)")
    lines.append(f"(Profil Normalize: LE -> X=0, Taban -> Y=0 (alt duzlem))")
    lines.append(f"(Speed: {feed} mm/min)")
    lines.append(f"(Kerf: {kerf:.2f}mm, Tel Tolerans: {wire_tol:.2f}mm, Toplam Offset: {total_offset:.2f}mm)")
    if not np.isclose(foam_width, span):
        lines.append(f"(Taper Projeksiyon: AKTIF)")
    lines.append(f"(Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')})")
    lines.append("G21 (Metric)")
    lines.append("G90 (Absolute)")
    lines.append("G49 (Cancel offsets)")
    lines.append("()")

    # Start position - machine home (0, 0, 0, 0)
    # Axis mapping for Grbl HotWire 6.5:
    #   X = root horizontal (left), Y = root vertical (left)
    #   A = tip horizontal (right), Z = tip vertical (right)
    start_x = float(left[0][0])
    start_y = float(left[0][1])
    start_a = float(right[0][0])
    start_z = float(right[0][1])

    lines.append("(Baslangic - Motor home: X0 Y0 Z0 A0)")
    lines.append("G00 X0.00 Y0.00 Z0.00 A0.00")
    lines.append("()")

    # L-shaped lead-in:
    #   Segment 1: X / A move only   (Y=Z=0 stays)
    #   Segment 2: Y / Z move up to the first cut point
    # Both carriages move simultaneously on each G01, so each segment takes
    # one line of G-code.
    lines.append("(Lead-in L-sekli: once X/A ekseninde, sonra Y/Z ekseninde)")
    lines.append(
        f"G01 X{start_x:.2f} Y0.00 Z0.00 A{start_a:.2f} F{feed}"
    )
    lines.append(
        f"G01 X{start_x:.2f} Y{start_y:.2f} Z{start_z:.2f} A{start_a:.2f} F{feed}"
    )
    lines.append("()")

    # Cutting pass — walk every point in the prepared (CW, mid-lower start) order
    lines.append(f"(Kesim Basliyor - Hiz F{feed} - Saat Yonu)")
    for i in range(len(left)):
        x = left[i][0]       # left horizontal
        y = left[i][1]       # left vertical (root_y_offset already baked in)
        a = right[i][0]      # right horizontal
        z = right[i][1]      # right vertical (tip_z_offset already baked in)
        lines.append(f"G01 X{x:.2f} Y{y:.2f} Z{z:.2f} A{a:.2f} F{feed}")

    # Close the profile - return to first cutting point
    lines.append(
        f"G01 X{start_x:.2f} Y{start_y:.2f} Z{start_z:.2f} A{start_a:.2f} F{feed}"
    )
    lines.append("()")

    # L-shaped lead-out (reverse of lead-in):
    #   Segment 1: Y / Z back to 0 (keeping X / A at start)
    #   Segment 2: X / A back to 0
    lines.append("(Lead-out L-sekli: once Y/Z 0'a, sonra X/A 0'a)")
    lines.append(
        f"G01 X{start_x:.2f} Y0.00 Z0.00 A{start_a:.2f} F{feed}"
    )
    lines.append(f"G01 X0.00 Y0.00 Z0.00 A0.00 F{feed}")
    lines.append("(Kesim Bitti)")

    return "\n".join(lines)


def generate_spar_gcode(root_pts, tip_pts, params):
    """Generate G-code section for spar hole cutting.
    Same home -> lead-in -> CW cut -> lead-out -> home pattern as the main."""
    tp = build_machine_toolpath(root_pts, tip_pts, params)
    left = tp["left"]
    right = tp["right"]
    feed = tp["feed"]

    start_x = float(left[0][0])
    start_y = float(left[0][1])
    start_a = float(right[0][0])
    start_z = float(right[0][1])

    lines = []
    lines.append("()")
    lines.append("(=== SPAR DELIGI KESIMI ===)")
    lines.append("(Spar baslangic - Motor home)")
    lines.append("G00 X0.00 Y0.00 Z0.00 A0.00")
    lines.append("(Spar Lead-in L-sekli: once X/A, sonra Y/Z)")
    lines.append(f"G01 X{start_x:.2f} Y0.00 Z0.00 A{start_a:.2f} F{feed}")
    lines.append(
        f"G01 X{start_x:.2f} Y{start_y:.2f} Z{start_z:.2f} A{start_a:.2f} F{feed}"
    )
    lines.append("()")
    lines.append(f"(Spar Kesim Basliyor - Hiz F{feed} - Saat Yonu)")

    for i in range(len(left)):
        x = left[i][0]
        y = left[i][1]
        a = right[i][0]
        z = right[i][1]
        lines.append(f"G01 X{x:.2f} Y{y:.2f} Z{z:.2f} A{a:.2f} F{feed}")

    # Close
    lines.append(
        f"G01 X{start_x:.2f} Y{start_y:.2f} Z{start_z:.2f} A{start_a:.2f} F{feed}"
    )
    lines.append("()")

    lines.append("(Spar Lead-out L-sekli: once Y/Z 0, sonra X/A 0)")
    lines.append(f"G01 X{start_x:.2f} Y0.00 Z0.00 A{start_a:.2f} F{feed}")
    lines.append(f"G01 X0.00 Y0.00 Z0.00 A0.00 F{feed}")
    lines.append("(Spar Kesim Bitti)")

    return "\n".join(lines)


def check_machine_limits(root_pts, tip_pts, root_y_offset=0.0, tip_z_offset=0.0):
    """Check if coordinates exceed machine limits.
    Axis mapping: X=root horiz, Y=root vert, A=tip horiz, Z=tip vert."""
    warnings = []

    root_x_max = np.max(root_pts[:, 0])                   # X axis
    root_y_max = np.max(root_pts[:, 1]) + root_y_offset   # Y axis
    tip_a_max = np.max(tip_pts[:, 0])                     # A axis (tip horizontal)
    tip_z_max = np.max(tip_pts[:, 1]) + tip_z_offset      # Z axis (tip vertical)

    if root_x_max > MAX_X:
        warnings.append(f"X max ({root_x_max:.1f}mm) makine limitini asiyor ({MAX_X}mm)")
    if root_y_max > MAX_Y:
        warnings.append(f"Y max ({root_y_max:.1f}mm) makine limitini asiyor ({MAX_Y}mm)")
    if tip_a_max > MAX_X:
        warnings.append(f"A max ({tip_a_max:.1f}mm) makine limitini asiyor ({MAX_X}mm)")
    if tip_z_max > MAX_Y:
        warnings.append(f"Z max ({tip_z_max:.1f}mm) makine limitini asiyor ({MAX_Y}mm)")

    root_x_min = np.min(root_pts[:, 0])
    tip_a_min = np.min(tip_pts[:, 0])

    if root_x_min < -MAX_X:
        warnings.append(f"X min ({root_x_min:.1f}mm) negatif limiti asiyor")
    if tip_a_min < -MAX_X:
        warnings.append(f"A min ({tip_a_min:.1f}mm) negatif limiti asiyor")

    return warnings


class Tooltip:
    """Lightweight hover tooltip for Tk widgets. Shows a small yellow popup
    with the given text when the mouse enters the widget, and hides it again
    on leave. Used for the ⓘ info icons next to each parameter input."""

    def __init__(self, widget, text, wraplength=340):
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self.tip_window = None
        widget.bind("<Enter>", self._on_enter)
        widget.bind("<Leave>", self._on_leave)
        widget.bind("<ButtonPress>", self._on_leave)

    def _on_enter(self, _event=None):
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        try:
            tw.tk.call("::tk::unsupported::MacWindowStyle", "style",
                       tw._w, "help", "noActivates")
        except tk.TclError:
            pass
        tk.Label(
            tw, text=self.text, justify=tk.LEFT,
            background="#ffffe0", foreground="#222",
            relief=tk.SOLID, borderwidth=1,
            font=("Arial", 9), wraplength=self.wraplength,
            padx=8, pady=5,
        ).pack()

    def _on_leave(self, _event=None):
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


# Human-readable descriptions for every configurable input.
# Shown when the user hovers the ⓘ icon next to each field.
PARAM_TOOLTIPS = {
    "feed_rate": (
        "Kesim hizi (mm/dakika).\n\n"
        "Sicak telin kopuk uzerinde ne kadar hizli ilerleyecegini belirler. "
        "Dusuk degerlerde kesim daha temiz olur ama erime artar; yuksek "
        "degerlerde tel kopuge takilabilir. Tipik: 200-500 mm/min."
    ),
    "span": (
        "Tel mesafesi (mm).\n\n"
        "Sol ve sag carriage arasindaki toplam mesafe — yani makinenin iki "
        "ucundaki tel tutucularinin arasi. Tel bu uzunlukta gerilir."
    ),
    "foam_width": (
        "Kopuk blogu genisligi (mm).\n\n"
        "Kesilecek kopuk bloğunun span yonundeki fiziksel genisligi. "
        "Tel mesafesinden kucuk olabilir; o zaman taper projeksiyonu devreye "
        "girer ve carriage'lar daha genis bir yaya hareket eder."
    ),
    "foam_left_offset": (
        "Sol bosluk (mm).\n\n"
        "Sol carriage ile kopugun sol yuzu arasindaki mesafe. Kopugun tel "
        "hatti uzerinde tam olarak nereye oturdugunu anlatir ve taper "
        "projeksiyonunun uclugenini (triangle similarity) kurar."
    ),
    "root_x_offset": (
        "Root (sol carriage) X offset (mm).\n\n"
        "Profilin sol carriage'taki sol-alt kose (BB) X konumu. "
        "Lead-in L-seklindedir: once X ekseninde bu degere kadar gider, "
        "sonra Y ekseninde Y offset'e kadar. Profil bu noktadan yukari/saga "
        "dogru uzar ve tamamen pozitif bolgede kalir."
    ),
    "tip_x_offset": (
        "Tip (sag carriage) A offset (mm).\n\n"
        "Sag carriage (A ekseni) icin BB kose X konumu. Sol ile ayni deger "
        "-> straight wing; farkli -> sweep/yellene."
    ),
    "root_y_offset": (
        "Root (sol carriage) Y offset (mm).\n\n"
        "Profilin sol carriage'taki sol-alt kose (BB) Y konumu. "
        "Lead-in L'nin ikinci (dikey) ayagi bu yuksekligine cikar, sonra "
        "kesim baslar. 0 -> profil tabani motor 0 hizasinda. 20 -> profil "
        "tabani motor 0'dan 20mm yukarida."
    ),
    "tip_z_offset": (
        "Tip (sag carriage) Z offset (mm).\n\n"
        "Sag carriage (Z ekseni) icin BB kose Y konumu. Sol carriage'dan "
        "bagimsizdir; iki carriage arasinda yukseklik farki verebilirsin."
    ),
    "kerf": (
        "Kerf offset — tel cap telafisi (mm).\n\n"
        "Sicak telin kopukte actigi olugun genisligi (yaklasik tel capi). "
        "Kesim yolu profilin icine bu mesafe kadar kaydirilir ki nihai "
        "parcanin olculeri DXF ile ayni olsun. 0 -> telafi yok."
    ),
    "wire_tolerance": (
        "Tel tolerans — isil erime payi (mm).\n\n"
        "Tel etrafinda kopugun eriyerek genisleyen bolgesi. Kerf'e ek bir "
        "icsel offset olarak uygulanir. Kopuk tipi ve telin sicakligina "
        "gore deneyerek ayarlanir (tipik 0.5-1.5 mm)."
    ),
    "safe_y": (
        "Guvenli yukseklik (mm) — (su an kullanilmiyor).\n\n"
        "Yeni akista kesim dogrudan motor home (X0 Y0 Z0 A0) noktasindan "
        "baslar ve oraya doner; lead-in/lead-out cizgileri tam (0,0)'dan "
        "profile gider. safe_y parametresi geriye donuk uyumluluk icin "
        "duruyor, uretilen G-code'a etkisi yok."
    ),
    "num_points": (
        "Nokta sayisi (taper icin).\n\n"
        "Taper (farkli root/tip profilleri) varsa her iki profil chord "
        "fraksiyonuyla bu kadar noktaya yeniden orneklenir ki wire "
        "esleşmesi korunsun. DUZ KANATTA kullanilmaz — ham DXF noktalari "
        "aynen korunur. Tipik: 150-300."
    ),
    "profile_name": (
        "Profil adi.\n\n"
        "G-code basliginda yorum olarak ve .nc dosyasinin varsayilan "
        "isminde kullanilir. Orn: 'NACA2412' -> NACA2412_20260413.nc"
    ),
}

DXF_TOOLTIPS = {
    "root": (
        "Root profil DXF (sol / kanat kok profili).\n\n"
        "Kesilecek kanadin kok (sol carriage tarafi) profili. Kapali bir "
        "LWPOLYLINE / SPLINE / LINE+ARC zinciri olmasi gerekir."
    ),
    "tip": (
        "Tip profil DXF (sag / kanat uc profili).\n\n"
        "Kanadin uc tarafindaki profil. Taper icin zorunlu; duz kanatta "
        "bos birakilabilir (root profili her iki tarafta da kullanilir)."
    ),
    "spar": (
        "Spar profil DXF (opsiyonel).\n\n"
        "Kanat icine acilacak spar (kiris) deliginin profil cizimi. "
        "Yuklenirse ana kesim bittikten sonra ayri bir G-code bolumunde "
        "kesilir. Bos birakilabilir."
    ),
}


class HotWireCutterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sude - HotWire Foam Cutter (Grbl HotWire 6.5 XYZA)")
        self.root.geometry("1100x820")
        self.root.minsize(900, 700)

        self.root_profile = None
        self.tip_profile = None
        self.spar_root_profile = None
        self.spar_tip_profile = None
        self.root_dxf_path = None
        self.tip_dxf_path = None
        self.spar_dxf_path = None
        self.generated_gcode = None

        self._build_ui()
        self._load_settings()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_settings(self):
        """Load last-used values from SETTINGS_FILE into param_vars.
        Missing / unreadable file is silently ignored."""
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        if not isinstance(data, dict):
            return
        for key, val in data.items():
            if key == "profile_name":
                if hasattr(self, "profile_name_var"):
                    self.profile_name_var.set(str(val))
            elif key in self.param_vars:
                try:
                    self.param_vars[key].set(str(val))
                except Exception:
                    pass

    def _save_settings(self):
        """Persist current parameter values to SETTINGS_FILE.
        Silent failure — missing home dir or write permission should not
        interrupt the workflow."""
        data = {key: var.get() for key, var in self.param_vars.items()}
        data["profile_name"] = self.profile_name_var.get()
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def _on_close(self):
        """Persist settings on window close."""
        try:
            self._save_settings()
        finally:
            self.root.destroy()

    def _build_ui(self):
        # Main container
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # Top section: file loading + parameters side by side
        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(0, 5))

        # --- Left: File loading ---
        file_frame = ttk.LabelFrame(top, text="DXF Dosyalari", padding=8)
        file_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        def _info_icon(parent, tip_text):
            """Create a small blue ⓘ label with a hover tooltip and return it."""
            lbl = ttk.Label(parent, text=" \u24D8", foreground="#0066CC",
                            cursor="question_arrow")
            Tooltip(lbl, tip_text)
            return lbl

        # Root DXF
        rf = ttk.Frame(file_frame)
        rf.pack(fill=tk.X, pady=2)
        ttk.Label(rf, text="Root Profil:").pack(side=tk.LEFT)
        _info_icon(rf, DXF_TOOLTIPS["root"]).pack(side=tk.LEFT)
        self.root_label = ttk.Label(rf, text="Yuklenmedi", foreground="gray")
        self.root_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(rf, text="Yukle", command=self._load_root).pack(side=tk.RIGHT)

        # Tip DXF
        tf = ttk.Frame(file_frame)
        tf.pack(fill=tk.X, pady=2)
        ttk.Label(tf, text="Tip Profil:  ").pack(side=tk.LEFT)
        _info_icon(tf, DXF_TOOLTIPS["tip"]).pack(side=tk.LEFT)
        self.tip_label = ttk.Label(tf, text="Yuklenmedi (duz kanat icin bos birakin)", foreground="gray")
        self.tip_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(tf, text="Yukle", command=self._load_tip).pack(side=tk.RIGHT)

        # Spar DXF
        sf = ttk.Frame(file_frame)
        sf.pack(fill=tk.X, pady=2)
        ttk.Label(sf, text="Spar Profil: ").pack(side=tk.LEFT)
        _info_icon(sf, DXF_TOOLTIPS["spar"]).pack(side=tk.LEFT)
        self.spar_label = ttk.Label(sf, text="Opsiyonel", foreground="gray")
        self.spar_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(sf, text="Yukle", command=self._load_spar).pack(side=tk.RIGHT)

        # Profile name
        nf = ttk.Frame(file_frame)
        nf.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(nf, text="Profil Adi:").pack(side=tk.LEFT)
        _info_icon(nf, PARAM_TOOLTIPS["profile_name"]).pack(side=tk.LEFT)
        self.profile_name_var = tk.StringVar(value="NACA2412")
        ttk.Entry(nf, textvariable=self.profile_name_var, width=20).pack(side=tk.LEFT, padx=5)

        # --- Right: Parameters ---
        param_frame = ttk.LabelFrame(top, text="Kesim Parametreleri", padding=8)
        param_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(5, 0))

        params = [
            ("Feed Rate (mm/min):", "feed_rate", DEFAULT_FEED),
            ("Tel Mesafesi (mm):", "span", DEFAULT_SPAN),
            ("Kopuk Genisligi (mm):", "foam_width", DEFAULT_FOAM_WIDTH),
            ("Sol Bosluk (mm):", "foam_left_offset", DEFAULT_FOAM_LEFT_OFFSET),
            ("Root X Offset (mm):", "root_x_offset", 0.0),
            ("Tip X Offset (mm):", "tip_x_offset", 0.0),
            ("Root Y Offset (mm):", "root_y_offset", 0.0),
            ("Tip Z Offset (mm):", "tip_z_offset", 0.0),
            ("Kerf Offset (mm):", "kerf", DEFAULT_KERF),
            ("Tel Tolerans (mm):", "wire_tolerance", 1.0),
            ("Guvenli Yukseklik (mm):", "safe_y", DEFAULT_SAFE_Y),
            ("Nokta Sayisi:", "num_points", DEFAULT_NUM_POINTS),
        ]

        self.param_vars = {}
        for i, (label, key, default) in enumerate(params):
            ttk.Label(param_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=1)
            var = tk.StringVar(value=str(default))
            self.param_vars[key] = var
            entry = ttk.Entry(param_frame, textvariable=var, width=10)
            entry.grid(row=i, column=1, padx=5, pady=1)
            entry.bind("<Return>", lambda e: self._preview())
            info = _info_icon(param_frame, PARAM_TOOLTIPS.get(key, ""))
            info.grid(row=i, column=2, sticky=tk.W, padx=(2, 4))

        # Middle section: Canvas + G-code preview
        mid = ttk.Frame(main)
        mid.pack(fill=tk.BOTH, expand=True)

        # Canvas for profile preview
        canvas_frame = ttk.LabelFrame(mid, text="Profil Onizleme", padding=4)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=1,
                                highlightbackground="#ccc")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # G-code preview
        gcode_frame = ttk.LabelFrame(mid, text="G-code Onizleme", padding=4)
        gcode_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.gcode_text = tk.Text(gcode_frame, wrap=tk.NONE, font=("Courier", 10),
                                  state=tk.DISABLED, bg="#1e1e1e", fg="#d4d4d4",
                                  insertbackground="white")
        gcode_scroll_y = ttk.Scrollbar(gcode_frame, orient=tk.VERTICAL,
                                       command=self.gcode_text.yview)
        gcode_scroll_x = ttk.Scrollbar(gcode_frame, orient=tk.HORIZONTAL,
                                       command=self.gcode_text.xview)
        self.gcode_text.configure(yscrollcommand=gcode_scroll_y.set,
                                  xscrollcommand=gcode_scroll_x.set)
        gcode_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        gcode_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.gcode_text.pack(fill=tk.BOTH, expand=True)

        # Bottom: buttons
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(btn_frame, text="Profil Onizle", command=self._preview).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="3D Onizle", command=self._preview_3d).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="G-code Uret", command=self._generate).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="G-code Kaydet (.nc)", command=self._save).pack(side=tk.LEFT, padx=3)

        # Status bar
        self.status_var = tk.StringVar(value="Hazir.")
        status = ttk.Label(main, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status.pack(fill=tk.X, pady=(5, 0))

    def _load_dxf(self, title):
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[("DXF Dosyalari", "*.dxf"), ("Tum Dosyalar", "*.*")]
        )
        return path if path else None

    def _load_root(self):
        path = self._load_dxf("Root Profil DXF Sec")
        if not path:
            return
        try:
            pts = extract_profile_from_dxf(path)
            self.root_profile = pts
            self.root_dxf_path = path
            name = os.path.basename(path)
            self.root_label.config(text=name, foreground="blue")
            self.status_var.set(f"Root profil yuklendi: {len(pts)} nokta")
            self._preview()
        except Exception as e:
            messagebox.showerror("Hata", f"Root DXF okunamadi:\n{e}")

    def _load_tip(self):
        path = self._load_dxf("Tip Profil DXF Sec")
        if not path:
            return
        try:
            pts = extract_profile_from_dxf(path)
            self.tip_profile = pts
            self.tip_dxf_path = path
            name = os.path.basename(path)
            self.tip_label.config(text=name, foreground="red")
            self.status_var.set(f"Tip profil yuklendi: {len(pts)} nokta")
            self._preview()
        except Exception as e:
            messagebox.showerror("Hata", f"Tip DXF okunamadi:\n{e}")

    def _load_spar(self):
        path = self._load_dxf("Spar Profil DXF Sec")
        if not path:
            return
        try:
            pts = extract_profile_from_dxf(path)
            self.spar_root_profile = pts
            self.spar_tip_profile = pts.copy()  # Same spar for both sides by default
            self.spar_dxf_path = path
            name = os.path.basename(path)
            self.spar_label.config(text=name, foreground="green")
            self.status_var.set(f"Spar profil yuklendi: {len(pts)} nokta")
        except Exception as e:
            messagebox.showerror("Hata", f"Spar DXF okunamadi:\n{e}")

    def _get_params(self):
        try:
            return {
                "feed_rate": int(float(self.param_vars["feed_rate"].get())),
                "span": float(self.param_vars["span"].get()),
                "foam_width": float(self.param_vars["foam_width"].get()),
                "foam_left_offset": float(self.param_vars["foam_left_offset"].get()),
                "root_x_offset": float(self.param_vars["root_x_offset"].get()),
                "tip_x_offset": float(self.param_vars["tip_x_offset"].get()),
                "root_y_offset": float(self.param_vars["root_y_offset"].get()),
                "tip_z_offset": float(self.param_vars["tip_z_offset"].get()),
                "kerf": float(self.param_vars["kerf"].get()),
                "wire_tolerance": float(self.param_vars["wire_tolerance"].get()),
                "safe_y": float(self.param_vars["safe_y"].get()),
                "num_points": int(float(self.param_vars["num_points"].get())),
                "profile_name": self.profile_name_var.get(),
            }
        except ValueError as e:
            messagebox.showerror("Parametre Hatasi", f"Gecersiz parametre degeri:\n{e}")
            return None

    def _get_preview_offsets(self):
        """Return the four offsets used by the preview. Falls back to 0 on parse errors."""
        def read(key):
            try:
                return float(self.param_vars[key].get())
            except (ValueError, KeyError):
                return 0.0
        return read("root_x_offset"), read("tip_x_offset"), \
               read("root_y_offset"), read("tip_z_offset")

    def _normalized_profile(self, raw):
        """Return the profile prepared the same way the G-code pipeline does:
        normalize (LE→X=0, taban→Y=0), orient clockwise, then roll so the first
        point is the mid-chord lower-surface start. This keeps the 2D preview's
        Baslangic marker aligned with the actual cut start."""
        if raw is None:
            return None
        norm, _ = normalize_profile(raw.copy())
        norm = _orient_cw(norm)
        norm, _ = _roll_to_mid_lower(norm)
        return norm

    def _draw_profile(self, points, color, label, x_offset=0.0, y_offset=0.0):
        """Draw a profile on the canvas with optional X/Y offset."""
        canvas = self.canvas
        w = canvas.winfo_width()
        h = canvas.winfo_height()

        if w < 10 or h < 10:
            return

        root_xoff, tip_xoff, root_yoff, tip_zoff = self._get_preview_offsets()

        # Find bounds across all profiles WITH their offsets for consistent scaling.
        # Use normalized profiles so the preview matches the G-code coordinate system.
        all_pts = []
        root_norm = self._normalized_profile(self.root_profile)
        tip_norm = self._normalized_profile(self.tip_profile)
        if root_norm is not None:
            shifted = root_norm.copy()
            shifted[:, 0] += root_xoff
            shifted[:, 1] += root_yoff
            all_pts.append(shifted)
        if tip_norm is not None:
            shifted = tip_norm.copy()
            shifted[:, 0] += tip_xoff
            shifted[:, 1] += tip_zoff
            all_pts.append(shifted)
        if not all_pts:
            return

        combined = np.vstack(all_pts)
        x_min, y_min = combined.min(axis=0)
        x_max, y_max = combined.max(axis=0)
        # Always include the motor 0 line in the visible area
        y_min = min(y_min, 0.0)
        x_min = min(x_min, 0.0)

        x_range = x_max - x_min if x_max > x_min else 1
        y_range = y_max - y_min if y_max > y_min else 1

        margin = 40
        scale_x = (w - 2 * margin) / x_range
        scale_y = (h - 2 * margin) / y_range
        scale = min(scale_x, scale_y)

        cx = w / 2
        cy = h / 2
        ox = (x_min + x_max) / 2
        oy = (y_min + y_max) / 2

        def transform(px, py):
            sx = cx + (px - ox) * scale
            sy = cy - (py - oy) * scale  # Flip Y
            return sx, sy

        # Draw motor 0 reference lines once (tagged separately so they don't
        # stack up when multiple profiles are drawn)
        if not canvas.find_withtag("zero_axes"):
            zx_a, zy_a = transform(x_min, 0.0)
            zx_b, zy_b = transform(x_max, 0.0)
            canvas.create_line(zx_a, zy_a, zx_b, zy_b, fill="#AAAAAA",
                               dash=(4, 3), tags=("profile", "zero_axes"))
            zx_c, zy_c = transform(0.0, y_min)
            zx_d, zy_d = transform(0.0, y_max)
            canvas.create_line(zx_c, zy_c, zx_d, zy_d, fill="#AAAAAA",
                               dash=(4, 3), tags=("profile", "zero_axes"))
            canvas.create_text(zx_b - 4, zy_b - 8, text="Motor 0 (Y)",
                               anchor=tk.E, fill="#888", font=("Arial", 8),
                               tags=("profile", "zero_axes"))

        # Draw profile with offset applied
        coords = []
        for pt in points:
            sx, sy = transform(pt[0] + x_offset, pt[1] + y_offset)
            coords.extend([sx, sy])
        # Close the loop
        sx, sy = transform(points[0][0] + x_offset, points[0][1] + y_offset)
        coords.extend([sx, sy])

        if len(coords) >= 4:
            canvas.create_line(coords, fill=color, width=2, tags="profile")

        # Mark starting point (mid-lower, CW direction)
        sx, sy = transform(points[0][0] + x_offset, points[0][1] + y_offset)
        r = 5
        canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill=color, outline="black",
                           tags="profile")
        canvas.create_text(sx + 12, sy - 10, text=f"{label} Baslangic",
                           fill=color, font=("Arial", 9, "bold"), tags="profile")

    def _preview(self):
        """Draw profile preview on canvas."""
        if self.root_profile is None:
            return

        self.canvas.delete("profile")
        self.canvas.update_idletasks()

        root_xoff, tip_xoff, root_yoff, tip_zoff = self._get_preview_offsets()

        root_norm = self._normalized_profile(self.root_profile)
        self._draw_profile(root_norm, "#2060FF", "Root",
                           x_offset=root_xoff, y_offset=root_yoff)

        tip_norm = self._normalized_profile(self.tip_profile)
        if tip_norm is not None:
            self._draw_profile(tip_norm, "#FF3030", "Tip",
                               x_offset=tip_xoff, y_offset=tip_zoff)

        # Legend
        w = self.canvas.winfo_width()
        self.canvas.create_text(w - 10, 15, text="Root (mavi) / Tip (kirmizi)",
                                anchor=tk.E, font=("Arial", 9), fill="#555", tags="profile")

    def _preview_3d(self):
        """Show 3D preview of the cutting operation."""
        if self.root_profile is None:
            messagebox.showwarning("Uyari", "Lutfen once bir Root profil DXF yukleyin!")
            return

        params = self._get_params()
        if params is None:
            return

        try:
            root_interp, tip_interp = self._process_profiles(params)
        except Exception as e:
            messagebox.showerror("Hata", f"Profil isleme hatasi:\n{e}")
            return

        root_xoff = params["root_x_offset"]
        tip_xoff = params["tip_x_offset"]
        root_yoff = params["root_y_offset"]
        tip_zoff = params["tip_z_offset"]
        foam_width = params["foam_width"]
        span = params["span"]
        foam_lo = params["foam_left_offset"]

        # Root profile at Y=0 (foam left face), Tip at Y=foam_width (foam right face)
        root_pts = root_interp.copy()
        tip_pts = tip_interp.copy()
        root_pts[:, 0] += root_xoff
        tip_pts[:, 0] += tip_xoff

        # Close profiles for drawing
        root_closed = np.vstack([root_pts, root_pts[0:1]])
        tip_closed = np.vstack([tip_pts, tip_pts[0:1]])

        # Projected carriage positions
        left_car, right_car = apply_taper_projection(
            root_pts.copy(), tip_pts.copy(), span, foam_width, foam_lo
        )
        left_closed = np.vstack([left_car, left_car[0:1]])
        right_closed = np.vstack([right_car, right_car[0:1]])

        # Create 3D plot window
        win = tk.Toplevel(self.root)
        win.title("Sude - 3D Kesim Onizleme")
        win.geometry("900x700")

        fig = plt.Figure(figsize=(10, 8), facecolor="#2b2b2b")
        ax = fig.add_subplot(111, projection="3d", facecolor="#2b2b2b")

        # Vertical offsets: left carriage uses root_y_offset (Y axis),
        # right carriage uses tip_z_offset (Z axis). Profile display at foam
        # faces is schematic — lerp between the two for a readable picture.
        def lerp(a, b, t):
            return a * (1.0 - t) + b * t

        span_s = max(span, 1e-9)
        foam_lo_off = lerp(root_yoff, tip_zoff, foam_lo / span_s)
        foam_ro_off = lerp(root_yoff, tip_zoff, (foam_lo + foam_width) / span_s)

        # Y axis = span direction
        # Root profile at foam left face
        ax.plot(root_closed[:, 0], np.full(len(root_closed), foam_lo),
                root_closed[:, 1] + foam_lo_off,
                color="#4488FF", linewidth=2, label="Root (kopuk sol yuz)")

        # Tip profile at foam right face
        ax.plot(tip_closed[:, 0], np.full(len(tip_closed), foam_lo + foam_width),
                tip_closed[:, 1] + foam_ro_off,
                color="#FF4444", linewidth=2, label="Tip (kopuk sag yuz)")

        # Left carriage at Y=0 (motor 0 + root_y_offset)
        ax.plot(left_closed[:, 0], np.zeros(len(left_closed)),
                left_closed[:, 1] + root_yoff,
                color="#44FF44", linewidth=1.5, linestyle="--", alpha=0.7,
                label="Sol Carriage")

        # Right carriage at Y=span (motor 0 + tip_z_offset)
        ax.plot(right_closed[:, 0], np.full(len(right_closed), span),
                right_closed[:, 1] + tip_zoff,
                color="#FFAA00", linewidth=1.5, linestyle="--", alpha=0.7,
                label="Sag Carriage")

        # Wire lines connecting corresponding points (every 10th)
        step = max(1, len(root_pts) // 20)
        for i in range(0, len(root_pts), step):
            wire_x = [left_car[i, 0], root_pts[i, 0], tip_pts[i, 0], right_car[i, 0]]
            wire_y = [0, foam_lo, foam_lo + foam_width, span]
            wire_z = [left_car[i, 1] + root_yoff,
                      root_pts[i, 1] + foam_lo_off,
                      tip_pts[i, 1] + foam_ro_off,
                      right_car[i, 1] + tip_zoff]
            ax.plot(wire_x, wire_y, wire_z, color="white", linewidth=0.4, alpha=0.3)

        # Draw foam block outline (transparent box)
        x_min = min(root_pts[:, 0].min(), tip_pts[:, 0].min()) - 10
        x_max = max(root_pts[:, 0].max(), tip_pts[:, 0].max()) + 10
        z_max = max(root_pts[:, 1].max() + root_yoff,
                    tip_pts[:, 1].max() + tip_zoff) + 10

        # Foam block edges
        y0, y1 = foam_lo, foam_lo + foam_width
        for yy in [y0, y1]:
            ax.plot([x_min, x_max, x_max, x_min, x_min],
                    [yy, yy, yy, yy, yy],
                    [0, 0, z_max, z_max, 0],
                    color="cyan", linewidth=0.5, alpha=0.3)
        for xx, zz in [(x_min, 0), (x_max, 0), (x_max, z_max), (x_min, z_max)]:
            ax.plot([xx, xx], [y0, y1], [zz, zz], color="cyan", linewidth=0.5, alpha=0.3)

        # Styling
        ax.set_xlabel("Chord (mm)", color="white", fontsize=10)
        ax.set_ylabel("Span (mm)", color="white", fontsize=10)
        ax.set_zlabel("Yukseklik (mm)", color="white", fontsize=10)
        ax.set_title("3D Kesim Onizleme", color="white", fontsize=13, pad=10)

        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#555555")
        ax.yaxis.pane.set_edgecolor("#555555")
        ax.zaxis.pane.set_edgecolor("#555555")
        ax.grid(True, alpha=0.2)

        legend = ax.legend(loc="upper left", fontsize=9, facecolor="#333333",
                           edgecolor="#555555", labelcolor="white")

        # Embed in tkinter
        canvas_widget = FigureCanvasTkAgg(fig, master=win)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add matplotlib toolbar for rotation/zoom
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas_widget, win)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def _process_profiles(self, params):
        """Prepare root and tip profiles for cutting.

        Steps:
          1. normalize_profile (LE at X=0, taban at Y=0, array starts at LE)
          2. _ensure_ccw: force both profiles into counter-clockwise order so
             that arc-length resampling lines up root[i] with tip[i]. Without
             this, a DXF drawn CW and another drawn CCW would produce mirror
             indexed points and the taper projection would amplify the mismatch
             into huge Y jumps during the cut.
          3. Straight wing: use raw normalized points (DXF fidelity — no resampling).
             Tapered wing: arc-length resample to num_points. Arc-length (not
             chord-fraction) correctly samples vertical edges, so fuselage /
             body cross-sections with flat sides work.
          4. _orient_cw — reverse both simultaneously so cut direction is CW.
          5. _roll_to_mid_lower — start at mid-chord on the lower surface.
             Root's start index is also applied to the tip so the wire points
             correspond 1:1 across the span.
        """
        num_points = params["num_points"]

        root_raw = self.root_profile.copy()
        root_norm, _ = normalize_profile(root_raw)
        root_norm = _ensure_ccw(root_norm)

        if self.tip_profile is not None:
            tip_raw = self.tip_profile.copy()
            tip_norm, _ = normalize_profile(tip_raw)
            tip_norm = _ensure_ccw(tip_norm)
            # Tapered wing: arc-length resample both (same CCW winding + same
            # LE-anchored start -> root[i] and tip[i] are at matching
            # perimeter fractions).
            root_prepared = interpolate_profile(root_norm, num_points)
            tip_prepared = interpolate_profile(tip_norm, num_points)
        else:
            # Straight wing: use raw normalized points exactly.
            root_prepared = root_norm
            tip_prepared = root_norm.copy()

        # Reverse both (still in lock-step) so the cut direction is clockwise
        root_prepared = _orient_cw(root_prepared)
        tip_prepared = _orient_cw(tip_prepared)

        # Roll so cut starts at mid-chord on the lower surface.
        # Use the SAME start index for root and tip when point counts match
        # so the wire points stay linked across the span.
        root_prepared, start_idx = _roll_to_mid_lower(root_prepared)
        if len(tip_prepared) == len(root_prepared):
            tip_prepared = np.roll(tip_prepared, -start_idx, axis=0)
        else:
            tip_prepared, _ = _roll_to_mid_lower(tip_prepared)

        return root_prepared, tip_prepared

    def _process_spar_profiles(self, params):
        """Prepare spar profiles with the same CCW-interp/CW-cut convention as
        the main profile, so the spar G-code follows the same start pattern."""
        num_points = params["num_points"]
        if self.spar_root_profile is None:
            return None, None

        spar_root_norm, _ = normalize_profile(self.spar_root_profile.copy())
        spar_root_norm = _ensure_ccw(spar_root_norm)

        if self.spar_tip_profile is not None:
            spar_tip_norm, _ = normalize_profile(self.spar_tip_profile.copy())
            spar_tip_norm = _ensure_ccw(spar_tip_norm)
            spar_root_prep = interpolate_profile(spar_root_norm, num_points)
            spar_tip_prep = interpolate_profile(spar_tip_norm, num_points)
        else:
            spar_root_prep = spar_root_norm
            spar_tip_prep = spar_root_norm.copy()

        spar_root_prep = _orient_cw(spar_root_prep)
        spar_tip_prep = _orient_cw(spar_tip_prep)

        spar_root_prep, s_idx = _roll_to_mid_lower(spar_root_prep)
        if len(spar_tip_prep) == len(spar_root_prep):
            spar_tip_prep = np.roll(spar_tip_prep, -s_idx, axis=0)
        else:
            spar_tip_prep, _ = _roll_to_mid_lower(spar_tip_prep)
        return spar_root_prep, spar_tip_prep

    def _generate(self):
        """Build the toolpath, show the 3D confirmation preview, and only
        emit G-code after the user approves."""
        if self.root_profile is None:
            messagebox.showwarning("Uyari", "Lutfen once bir Root profil DXF yukleyin!")
            return

        params = self._get_params()
        if params is None:
            return

        # Validate taper parameters
        span = params["span"]
        foam_w = params["foam_width"]
        foam_lo = params["foam_left_offset"]
        if foam_w <= 0:
            messagebox.showerror("Hata", "Kopuk genisligi sifirdan buyuk olmali!")
            return
        if foam_lo < 0:
            messagebox.showerror("Hata", "Sol bosluk negatif olamaz!")
            return
        if foam_lo + foam_w > span + 0.01:
            messagebox.showerror("Hata",
                f"Kopuk genisligi ({foam_w:.1f}) + Sol bosluk ({foam_lo:.1f}) = "
                f"{foam_lo + foam_w:.1f}mm\n"
                f"Tel mesafesinden ({span:.1f}mm) buyuk olamaz!")
            return

        try:
            root_interp, tip_interp = self._process_profiles(params)
            spar_root, spar_tip = self._process_spar_profiles(params)

            # Compute the final machine toolpath(s) once, to both render the
            # preview and (on confirm) emit the G-code from the same data.
            main_tp = build_machine_toolpath(root_interp, tip_interp, params)
            spar_tp = None
            if spar_root is not None:
                spar_tp = build_machine_toolpath(spar_root, spar_tip, params)

            warnings = check_machine_limits(
                main_tp["left"], main_tp["right"],
                root_y_offset=0.0, tip_z_offset=0.0,  # already baked in
            )
            if warnings:
                msg = "Makine limiti uyarilari:\n\n" + "\n".join(warnings)
                msg += "\n\nYine de onay penceresine gecmek istiyor musunuz?"
                if not messagebox.askyesno("Uyari", msg):
                    return

            self._show_toolpath_confirm(
                main_tp, spar_tp, root_interp, tip_interp,
                spar_root, spar_tip, params,
            )

        except Exception as e:
            messagebox.showerror("Hata", f"Toolpath hesabi sirasinda hata:\n{e}")

    def _finalize_gcode(self, root_interp, tip_interp, spar_root, spar_tip, params):
        """Called after the user confirms the toolpath preview. Produces the
        G-code text, writes it to the main window, and persists settings."""
        try:
            gcode = generate_gcode(root_interp.copy(), tip_interp.copy(), params)

            if spar_root is not None:
                spar_gcode = generate_spar_gcode(
                    spar_root.copy(), spar_tip.copy(), params
                )
                # Drop the main cut's L-shaped lead-out + "Kesim Bitti" lines
                # so the spar section can start fresh from home (it emits its
                # own home + lead-in block). Main lead-out is 4 lines:
                #   (Lead-out L-sekli ...)  <- comment
                #   G01 ... Y0 Z0 ...        <- Y/Z to 0
                #   G01 X0 Y0 Z0 A0          <- X/A to 0
                #   (Kesim Bitti)            <- comment
                main_lines = gcode.split("\n")
                gcode = "\n".join(main_lines[:-4]) + "\n" + spar_gcode

            self.generated_gcode = gcode

            self.gcode_text.config(state=tk.NORMAL)
            self.gcode_text.delete("1.0", tk.END)
            self.gcode_text.insert("1.0", gcode)
            self.gcode_text.config(state=tk.DISABLED)

            line_count = gcode.count("\n") + 1
            self.status_var.set(f"G-code uretildi: {line_count} satir")

            # Persist the values the user just approved
            self._save_settings()
        except Exception as e:
            messagebox.showerror("Hata", f"G-code uretim hatasi:\n{e}")

    def _show_toolpath_confirm(self, main_tp, spar_tp,
                               root_interp, tip_interp,
                               spar_root, spar_tip, params):
        """Open a modal-style 3D window showing the exact toolpath the machine
        will follow (both carriages, approach/retract, start marker). The user
        must click Onayla to produce the G-code — Iptal just closes."""
        win = tk.Toplevel(self.root)
        win.title("Sude - Toolpath Onayi")
        win.geometry("1000x720")
        win.minsize(800, 560)
        win.transient(self.root)

        fig = plt.Figure(figsize=(11, 8), facecolor="#2b2b2b")
        ax = fig.add_subplot(111, projection="3d", facecolor="#2b2b2b")

        span = params["span"]
        foam_w = params["foam_width"]
        foam_lo = params["foam_left_offset"]

        def _draw_toolpath(tp, label_prefix, cut_color_left, cut_color_right,
                           approach_color):
            left = tp["left"]
            right = tp["right"]
            # Cut path closed for drawing
            left_closed = np.vstack([left, left[0:1]])
            right_closed = np.vstack([right, right[0:1]])

            ax.plot(left_closed[:, 0], np.zeros(len(left_closed)),
                    left_closed[:, 1],
                    color=cut_color_left, linewidth=2,
                    label=f"{label_prefix} Sol Carriage (X,Y)")
            ax.plot(right_closed[:, 0], np.full(len(right_closed), span),
                    right_closed[:, 1],
                    color=cut_color_right, linewidth=2,
                    label=f"{label_prefix} Sag Carriage (A,Z)")

            # L-shaped lead-in for each carriage:
            #   (0, 0) -> (start_x, 0) -> (start_x, start_y)
            lx0 = left[0, 0]; ly0 = left[0, 1]
            ax.plot([0, lx0, lx0], [0, 0, 0], [0, 0, ly0],
                    color=approach_color, linewidth=1.8, linestyle="--",
                    label=f"{label_prefix} Lead-in / Lead-out (L)")
            rx0 = right[0, 0]; ry0 = right[0, 1]
            ax.plot([0, rx0, rx0], [span, span, span], [0, 0, ry0],
                    color=approach_color, linewidth=1.8, linestyle="--")

            # (0, 0) home marker — where the cut literally starts
            ax.scatter([0], [0], [0],
                       color="#FFDD00", s=90, marker="s",
                       edgecolors="black", linewidths=1.5,
                       label=f"{label_prefix} Home (0,0)")
            ax.scatter([0], [span], [0],
                       color="#FFDD00", s=90, marker="s",
                       edgecolors="black", linewidths=1.5)

            # Start point markers (first profile point reached via lead-in)
            ax.scatter([left[0, 0]], [0], [left[0, 1]],
                       color="#00FF88", s=60, marker="o",
                       edgecolors="white", linewidths=1,
                       label=f"{label_prefix} Profil Baslangic (orta-alt)")
            ax.scatter([right[0, 0]], [span], [right[0, 1]],
                       color="#00FF88", s=60, marker="o",
                       edgecolors="white", linewidths=1)

            # Direction arrows (every ~15% of path)
            n = len(left)
            step = max(1, n // 8)
            for i in range(0, n - step, step):
                dx = left[i + step, 0] - left[i, 0]
                dy = left[i + step, 1] - left[i, 1]
                ax.quiver(left[i, 0], 0, left[i, 1],
                          dx, 0, dy, color=cut_color_left,
                          arrow_length_ratio=0.3, linewidth=1.0, alpha=0.6)

        _draw_toolpath(main_tp, "Ana", "#4488FF", "#FF4444", "#FFAA00")
        if spar_tp is not None:
            _draw_toolpath(spar_tp, "Spar", "#88AAFF", "#FFAA88", "#FFDD55")

        # Foam block outline for reference
        left_all = main_tp["left"]
        right_all = main_tp["right"]
        x_min = min(left_all[:, 0].min(), right_all[:, 0].min()) - 10
        x_max = max(left_all[:, 0].max(), right_all[:, 0].max()) + 10
        z_max = max(left_all[:, 1].max(), right_all[:, 1].max()) + 10

        y0, y1 = foam_lo, foam_lo + foam_w
        for yy in [y0, y1]:
            ax.plot([x_min, x_max, x_max, x_min, x_min],
                    [yy, yy, yy, yy, yy],
                    [0, 0, z_max, z_max, 0],
                    color="cyan", linewidth=0.5, alpha=0.3)
        for xx, zz in [(x_min, 0), (x_max, 0), (x_max, z_max), (x_min, z_max)]:
            ax.plot([xx, xx], [y0, y1], [zz, zz],
                    color="cyan", linewidth=0.5, alpha=0.3)

        ax.set_xlabel("X / A (mm)", color="white", fontsize=10)
        ax.set_ylabel("Span (mm)", color="white", fontsize=10)
        ax.set_zlabel("Y / Z (mm)", color="white", fontsize=10)

        # SolidWorks-style view presets keyed to Space / 1-4:
        #   1 = Iso, 2 = On, 3 = Ust, 4 = Yan, Space cycles through
        view_presets = [
            ("Iso",  30, -45),   # default perspective
            ("On",    0, -90),   # front: chord + height (XY profile shape)
            ("Ust",  90, -90),   # top:   chord + span   (looking down Z)
            ("Yan",   0,   0),   # side:  span + height  (wire direction)
        ]
        view_state = {"idx": 0}

        def _apply_view(idx):
            idx = idx % len(view_presets)
            view_state["idx"] = idx
            name, elev, azim = view_presets[idx]
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(
                f"Toolpath Onay - {name}   "
                f"(Space: sonraki gorunum  |  1=Iso 2=On 3=Ust 4=Yan)",
                color="white", fontsize=11, pad=10,
            )
            canvas_widget.draw()

        def _cycle(_event=None):
            _apply_view(view_state["idx"] + 1)

        # Initial view + title hint
        ax.set_title(
            "Toolpath Onay - Iso   "
            "(Space: sonraki gorunum  |  1=Iso 2=On 3=Ust 4=Yan)",
            color="white", fontsize=11, pad=10,
        )
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#555555")
        ax.yaxis.pane.set_edgecolor("#555555")
        ax.zaxis.pane.set_edgecolor("#555555")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", fontsize=8, facecolor="#333333",
                  edgecolor="#555555", labelcolor="white")

        # IMPORTANT: Pack the button row FIRST at side=BOTTOM so it reserves
        # its strip of space before the expanding 3D canvas claims everything.
        # Tk pack works outside-in: later elements fill what the earlier ones
        # left over, so the canvas must be packed LAST with expand=True.
        def _confirm():
            win.destroy()
            self._finalize_gcode(root_interp, tip_interp,
                                 spar_root, spar_tip, params)

        def _cancel():
            win.destroy()
            self.status_var.set("Toolpath iptal edildi.")

        # Bottom button bar — tall, bold, always visible
        bottom = ttk.Frame(win, padding=10)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        info_text = (
            f"Ana kesim: {len(main_tp['left'])} nokta  |  "
            f"Root chord: {main_tp['root_chord']:.1f}mm  |  "
            f"Tip chord: {main_tp['tip_chord']:.1f}mm  |  "
            f"Feed: {main_tp['feed']} mm/min  |  "
            f"Kerf+Tol: {main_tp['total_offset']:.2f}mm"
        )
        if spar_tp is not None:
            info_text += f"  |  Spar: {len(spar_tp['left'])} nokta"
        ttk.Label(bottom, text=info_text, foreground="#333").pack(
            side=tk.LEFT, padx=5
        )

        # Make the confirm button visually prominent (tk.Button supports
        # background color, ttk.Button does not on macOS)
        tk.Button(
            bottom, text="  Onayla ve G-code Uret  ", command=_confirm,
            bg="#00A86B", fg="white", activebackground="#008c5a",
            activeforeground="white",
            font=("Arial", 11, "bold"), relief=tk.RAISED, bd=2,
            cursor="hand2", padx=8, pady=4,
        ).pack(side=tk.RIGHT, padx=4)
        tk.Button(
            bottom, text="  Iptal  ", command=_cancel,
            bg="#C83232", fg="white", activebackground="#a02020",
            activeforeground="white",
            font=("Arial", 11, "bold"), relief=tk.RAISED, bd=2,
            cursor="hand2", padx=8, pady=4,
        ).pack(side=tk.RIGHT, padx=4)

        # Matplotlib navigation toolbar — packs above the button bar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar_frame = ttk.Frame(win)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        canvas_widget = FigureCanvasTkAgg(fig, master=win)
        toolbar = NavigationToolbar2Tk(canvas_widget, toolbar_frame)
        toolbar.update()

        # 3D canvas last — fills whatever vertical space the bottom bars left
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Cancel on window-close as well
        win.protocol("WM_DELETE_WINDOW", _cancel)

        # SolidWorks-style keyboard view shortcuts.
        # Bind to the Toplevel so they work anywhere inside the window; also
        # to the canvas widget so clicks inside matplotlib don't lose focus.
        def _key_handler(event):
            k = event.keysym.lower()
            if k == "space":
                _cycle()
            elif k in ("1", "2", "3", "4"):
                _apply_view(int(k) - 1)

        win.bind("<Key>", _key_handler)
        canvas_widget.get_tk_widget().bind("<Key>", _key_handler)
        canvas_widget.get_tk_widget().bind(
            "<Button-1>", lambda e: canvas_widget.get_tk_widget().focus_set()
        )
        win.focus_set()

    def _save(self):
        """Save generated G-code to .nc file."""
        if not self.generated_gcode:
            messagebox.showwarning("Uyari", "Once G-code uretin!")
            return

        profile_name = self.profile_name_var.get().replace(" ", "_")
        date_str = datetime.now().strftime("%Y%m%d")
        default_name = f"{profile_name}_{date_str}.nc"

        path = filedialog.asksaveasfilename(
            title="G-code Kaydet",
            defaultextension=".nc",
            initialfile=default_name,
            filetypes=[("NC Dosyasi", "*.nc"), ("G-code", "*.gcode"), ("Tum Dosyalar", "*.*")]
        )

        if not path:
            return

        try:
            with open(path, "w") as f:
                f.write(self.generated_gcode)
            self.status_var.set(f"Kaydedildi: {path}")
            messagebox.showinfo("Basarili", f"G-code kaydedildi:\n{path}")
        except Exception as e:
            messagebox.showerror("Hata", f"Dosya kaydedilemedi:\n{e}")


def main():
    root = tk.Tk()
    app = HotWireCutterApp(root)

    # Redraw on resize
    def on_resize(event):
        if app.root_profile is not None:
            app.root.after(100, app._preview)

    app.canvas.bind("<Configure>", on_resize)

    root.mainloop()


if __name__ == "__main__":
    main()
