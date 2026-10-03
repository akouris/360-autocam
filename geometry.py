#!/usr/bin/env python3
"""
Shared 360° geometry — the ONE place that defines how a (yaw, pitch) maps to
pixels. Detection (Python remap) and rendering (FFmpeg v360) both go through
here so they can never disagree again.

Coordinate frame ("global", geometry v2)
  yaw   : degrees, 0 = front-lens optical axis, +90 = right, 180 = back-lens axis.
  pitch : degrees, + = up.
  Front lens local yaw L  → global L.   Back lens local yaw L → global 180 + L.
  (No mirroring for either lens — verified against v360=dfisheye, which treats
  the back lens the same way.)

Fisheye mode
  Both lens streams are hstacked [back | front] and fed to v360=dfisheye, so the
  front axis lands on yaw 0 and a crop can straddle the lens seam. The lenses
  cover MORE than 180°; ih_fov/iv_fov = the calibrated lens FOV (data/lens.json,
  written by step 1). With the wrong FOV, angles near the lens edge are off and
  the seam doesn't line up.

Equirect mode
  Single equirectangular stream; yaw/pitch are used as-is.
"""

import json, math, os
import numpy as np

DEFAULT_LENS_FOV = 192.0      # Insta360 X-series ≈ 190–197°; calibrated per video
LENS_JSON        = "data/lens.json"


def norm_yaw(y):
    """Wrap to [-180, 180)."""
    return ((y + 180.0) % 360.0) - 180.0


def video_mode():
    return "equirect" if os.environ.get("VIDEO_MODE") == "equirect" else "fisheye"


def lens_fov(path=LENS_JSON):
    try:
        with open(path) as f:
            return float(json.load(f).get("fov", DEFAULT_LENS_FOV))
    except Exception:
        return DEFAULT_LENS_FOV


# ── FFmpeg side ───────────────────────────────────────────────────────────────

def v360_source(mode=None, fps=None):
    """Filter-graph prefix producing the panoramic source as [src].
    fps: optionally decimate each lens stream first (cheap analysis passes).
    Returns (graph_prefix, input_format, extra_v360_opts)."""
    mode = mode or video_mode()
    pre = f"fps={fps}" if fps else "null"
    if mode == "equirect":
        return f"[0:v:0]{pre}[src]", "equirect", ""
    fov = lens_fov()
    return (f"[0:v:1]{pre}[_bk];[0:v:0]{pre}[_fr];[_bk][_fr]hstack=inputs=2[src]",
            "dfisheye", f":ih_fov={fov:.2f}:iv_fov={fov:.2f}")


def v360_view(yaw, pitch, h_fov, v_fov, w, h, mode=None):
    """v360 filter (without input label) for a rectilinear view."""
    _, inp, extra = v360_source(mode)
    return (f"v360={inp}:flat{extra}:yaw={norm_yaw(yaw):.3f}:pitch={pitch:.3f}"
            f":h_fov={h_fov:.3f}:v_fov={v_fov:.3f}:w={int(w)}:h={int(h)}")


def views_graph(views, layout="vstack", mode=None, fps=None):
    """
    Build a filter_complex that renders several rectilinear views from ONE
    decode and stacks them. views = [(yaw, pitch, h_fov, v_fov, w, h), ...].
    layout: "vstack" | "hstack" | "top_pair" (view 0 on top, views 1 and 2 side
    by side underneath). Output label: [out].
    """
    prefix = v360_source(mode, fps)[0]
    n = len(views)
    parts = [prefix]
    if n == 1:
        parts.append(f"[src]{v360_view(*views[0], mode=mode)}[out]")
        return ";".join(parts)
    parts.append(f"[src]split={n}" + "".join(f"[s{i}]" for i in range(n)))
    for i, v in enumerate(views):
        parts.append(f"[s{i}]{v360_view(*v, mode=mode)}[p{i}]")
    if layout == "top_pair" and n == 3:
        parts.append("[p1][p2]hstack=inputs=2[bot]")
        parts.append("[p0][bot]vstack=inputs=2[out]")
    else:
        parts.append("".join(f"[p{i}]" for i in range(n)) + f"{layout}=inputs={n}[out]")
    return ";".join(parts)


# ── Python side (frame analysis) ──────────────────────────────────────────────

def view_rays(yaw, pitch, h_fov, v_fov, w, h):
    """Unit ray directions (x right, y up, z = front axis) for every output pixel
    of a rectilinear view. Rotation order matches v360: pitch, then yaw."""
    tx = math.tan(math.radians(h_fov / 2))
    ty = math.tan(math.radians(v_fov / 2))
    xs = np.linspace(-tx, tx, w, dtype=np.float64)
    ys = np.linspace(ty, -ty, h, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    return rotate(X, Y, np.ones_like(X), yaw, pitch)


def rotate(X, Y, Z, yaw, pitch):
    n = np.sqrt(X * X + Y * Y + Z * Z)
    X, Y, Z = X / n, Y / n, Z / n
    p, yw = math.radians(pitch), math.radians(yaw)
    Y, Z = Y * math.cos(p) + Z * math.sin(p), -Y * math.sin(p) + Z * math.cos(p)
    X, Z = X * math.cos(yw) + Z * math.sin(yw), -X * math.sin(yw) + Z * math.cos(yw)
    return X, Y, Z


def ray_to_yaw_pitch(X, Y, Z):
    return np.degrees(np.arctan2(X, Z)), np.degrees(np.arcsin(np.clip(Y, -1, 1)))


def view_point_to_yaw_pitch(u, v, yaw, pitch, h_fov, v_fov):
    """Normalized (u,v ∈ [0,1], v down) point in a rectilinear view → global
    (yaw, pitch) in degrees. Exact inverse of the view projection."""
    x = (2 * u - 1) * math.tan(math.radians(h_fov / 2))
    y = (1 - 2 * v) * math.tan(math.radians(v_fov / 2))
    X, Y, Z = rotate(np.array(x), np.array(y), np.array(1.0), yaw, pitch)
    gy, gp = ray_to_yaw_pitch(X, Y, Z)
    return float(gy), float(gp)


class Panorama:
    """Samples rectilinear views from one moment of the 360 video.
    fisheye: front + back lens images (BGR); equirect: one equirect image."""

    _cache: dict = {}

    def __init__(self, front=None, back=None, equirect=None, fov=None):
        self.front, self.back, self.eq = front, back, equirect
        self.fov = fov or lens_fov()

    def _maps(self, yaw, pitch, h_fov, v_fov, w, h):
        src = self.eq if self.eq is not None else self.front
        key = (src.shape[:2], round(yaw, 2), round(pitch, 2), round(h_fov, 2),
               round(v_fov, 2), w, h, self.eq is not None, round(self.fov, 2))
        if key in Panorama._cache:
            return Panorama._cache[key]
        X, Y, Z = view_rays(yaw, pitch, h_fov, v_fov, w, h)
        if self.eq is not None:
            H, W = self.eq.shape[:2]
            gy, gp = ray_to_yaw_pitch(X, Y, Z)
            mx = ((gy + 180.0) / 360.0 * W).astype(np.float32)
            my = ((90.0 - gp) / 180.0 * H).astype(np.float32)
            maps = [(None, mx, my, None)]
        else:
            H, W = self.front.shape[:2]
            R, cx, cy = min(W, H) / 2.0, W / 2.0, H / 2.0
            half = math.radians(self.fov / 2)
            maps = []
            # front lens looks down +Z; back lens down -Z (rotated 180° about Y)
            for lens, (lx, lz) in (("front", (X, Z)), ("back", (-X, -Z))):
                th = np.arccos(np.clip(lz, -1, 1))
                ph = np.arctan2(Y, lx)
                r = R * th / half
                mx = (cx + r * np.cos(ph)).astype(np.float32)
                my = (cy - r * np.sin(ph)).astype(np.float32)
                use = (lz >= 0)   # each pixel from the lens it faces
                maps.append((lens, mx, my, use))
        if len(Panorama._cache) > 256:
            Panorama._cache.clear()
        Panorama._cache[key] = maps
        return maps

    def view(self, yaw, pitch, h_fov, v_fov, w, h):
        import cv2
        out = None
        for lens, mx, my, use in self._maps(yaw, pitch, h_fov, v_fov, w, h):
            img = self.eq if lens is None else (self.front if lens == "front" else self.back)
            part = cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if use is None:
                return part
            if out is None:
                out = np.zeros_like(part)
            out[use] = part[use]
        return out


# ── Lens FOV calibration ──────────────────────────────────────────────────────

def seam_error(front, back, fov, size=900):
    """Mismatch across the two lens seams of a stitch at this FOV: compare the
    pixel columns immediately either side of each seam (±~0.4°)."""
    import cv2
    pano = Panorama(front=front, back=back, fov=fov)
    err = 0.0
    for seam in (90.0, -90.0):
        strip = pano.view(seam, 0, 1.6, 120, 8, size)      # 0.2°/px across the seam
        g = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(float)
        err += np.abs(g[:, 1:3].mean(1) - g[:, 5:7].mean(1)).mean()
    return err


def calibrate_fov(pairs, lo=184.0, hi=204.0, step=0.5):
    """Pick the lens FOV that makes the seams line up best across frame pairs."""
    fovs = np.arange(lo, hi + 1e-6, step)
    scores = np.array([sum(seam_error(f, b, fv) for f, b in pairs) for fv in fovs])
    best = float(fovs[int(np.argmin(scores))])
    return best, dict(zip([float(x) for x in fovs], [float(s) for s in scores]))


# ── Framing ───────────────────────────────────────────────────────────────────

# Face size targets (face height as a fraction of pane height) and where the
# face centre sits vertically (fraction from the top) for each pane shape.
FRAMING = {
    # pane aspect (w/h) bucket : (face_frac, face_y)
    "tall":  (0.17, 0.38),   # 1080×1920 single
    "half":  (0.30, 0.45),   # 1080×960 split
    "third": (0.36, 0.47),   # 1080×640 strip
    "box":   (0.25, 0.40),   # 540×960 box (bottom pair of a 1 + 2 triple)
}
MIN_H_FOV, MAX_H_FOV = 28.0, 95.0
DEFAULT_FACE_DEG     = 10.0


def frame_view(face_yaw, face_pitch, face_deg, w, h, yaw_spread=0.0, kind=None):
    """
    Camera (yaw, pitch, h_fov, v_fov) that frames a face of angular height
    face_deg at (face_yaw, face_pitch) nicely in a w×h pane. yaw_spread widens
    the shot so a face that moves within the shot stays in frame.
    """
    aspect = w / h
    if kind is None:
        kind = ("box" if w <= 720 else "tall") if aspect < 0.8 else ("half" if aspect < 1.4 else "third")
    face_frac, face_y = FRAMING[kind]
    face_deg = face_deg if face_deg and face_deg > 1 else DEFAULT_FACE_DEG

    v_fov = face_deg / face_frac
    h_fov = 2 * math.degrees(math.atan(math.tan(math.radians(v_fov / 2)) * aspect))
    # keep a moving face inside the frame with ~a face-width margin each side
    need = yaw_spread + 2.2 * face_deg
    h_fov = max(h_fov, need)
    h_fov = min(max(h_fov, MIN_H_FOV), MAX_H_FOV)
    v_fov = 2 * math.degrees(math.atan(math.tan(math.radians(h_fov / 2)) / aspect))
    v_fov = min(v_fov, 150.0)

    # tilt so the face centre sits at face_y from the top
    off = math.degrees(math.atan((1 - 2 * face_y) * math.tan(math.radians(v_fov / 2))))
    pitch = max(-80.0, min(80.0, face_pitch - off))
    return norm_yaw(face_yaw), pitch, h_fov, v_fov
