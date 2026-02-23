"""Simple QA metrics + plots.

This project intentionally avoids heavy scientific plotting dependencies.
Plots are generated as SVG so it works on Windows without extra compilers.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import numpy as np


def _safe_float(v) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def _roi(arr: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    r0 = max(0, min(arr.shape[0], r0))
    r1 = max(0, min(arr.shape[0], r1))
    c0 = max(0, min(arr.shape[1], c0))
    c1 = max(0, min(arr.shape[1], c1))
    return arr[r0:r1, c0:c1]


def _status_badge(status: str) -> str:
    status = (status or "").lower()
    if status in {"pass", "fail", "na", "na"}:
        return status
    return "na"


def _metric_row(name: str, value: str | None, expected: str, status: str, notes: str = "") -> dict[str, Any]:
    return {
        "label": name,
        "value": "—" if value is None else value,
        "expected": expected,
        "status": _status_badge(status),
        "notes": notes,
    }


def _write_svg_plot(out_path: str, title: str, x: np.ndarray, series: list[tuple[str, np.ndarray]]) -> None:
    """Write a small SVG line plot (no extra dependencies)."""

    # Canvas
    W, H = 900, 420
    pad_l, pad_r, pad_t, pad_b = 70, 20, 45, 55
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b

    x = np.asarray(x, dtype=float)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_max == x_min:
        x_max = x_min + 1.0

    ys = [np.asarray(y, dtype=float) for _, y in series]
    y_min = float(np.min([np.min(y) for y in ys]))
    y_max = float(np.max([np.max(y) for y in ys]))
    if y_max == y_min:
        y_max = y_min + 1.0

    def sx(v: float) -> float:
        return pad_l + (v - x_min) / (x_max - x_min) * pw

    def sy(v: float) -> float:
        return pad_t + (1.0 - (v - y_min) / (y_max - y_min)) * ph

    # Palette
    colors = ["#2563eb", "#ef4444", "#10b981", "#a855f7"]

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    parts: list[str] = []
    parts.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>")
    parts.append(f"<rect x='0' y='0' width='{W}' height='{H}' rx='16' fill='white' stroke='#e5e7eb' />")
    parts.append(
        f"<text x='{pad_l}' y='28' font-family='Segoe UI, Arial' font-size='18' font-weight='600' fill='#111827'>{esc(title)}</text>"
    )

    # grid
    for i in range(6):
        yy = pad_t + i * (ph / 5)
        parts.append(f"<line x1='{pad_l}' y1='{yy:.2f}' x2='{pad_l + pw}' y2='{yy:.2f}' stroke='#f3f4f6' />")
    for i in range(6):
        xx = pad_l + i * (pw / 5)
        parts.append(f"<line x1='{xx:.2f}' y1='{pad_t}' x2='{xx:.2f}' y2='{pad_t + ph}' stroke='#f3f4f6' />")

    # axes
    parts.append(f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' y2='{pad_t + ph}' stroke='#9ca3af' />")
    parts.append(f"<line x1='{pad_l}' y1='{pad_t + ph}' x2='{pad_l + pw}' y2='{pad_t + ph}' stroke='#9ca3af' />")

    # series lines + legend
    legend_x = pad_l + pw - 170
    legend_y = 60
    for idx, (label, y) in enumerate(series):
        col = colors[idx % len(colors)]
        y = np.asarray(y, dtype=float)
        pts = " ".join([f"{sx(float(x[i])):.2f},{sy(float(y[i])):.2f}" for i in range(len(x))])
        parts.append(f"<polyline fill='none' stroke='{col}' stroke-width='2.5' points='{pts}' />")
        ly = legend_y + idx * 20
        parts.append(f"<line x1='{legend_x}' y1='{ly-6}' x2='{legend_x+26}' y2='{ly-6}' stroke='{col}' stroke-width='3' />")
        parts.append(
            f"<text x='{legend_x+32}' y='{ly-2}' font-family='Segoe UI, Arial' font-size='12' fill='#111827'>{esc(label)}</text>"
        )

    # labels
    parts.append(
        f"<text x='{pad_l + pw/2:.2f}' y='{H-18}' text-anchor='middle' font-family='Segoe UI, Arial' font-size='12' fill='#374151'>Pixel Number</text>"
    )
    parts.append(
        f"<text x='18' y='{pad_t + ph/2:.2f}' transform='rotate(-90 18 {pad_t + ph/2:.2f})' text-anchor='middle' font-family='Segoe UI, Arial' font-size='12' fill='#374151'>Pixel Value</text>"
    )
    parts.append("</svg>\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def run_basic_metrics(dicom_datasets, plot_dir: str) -> dict[str, Any]:
    """Compute a lightweight QA report.

    The metrics here are intentionally simple and based on basic image statistics
    (ROI checks) plus DICOM tags. More ACR-specific algorithms can be added later.
    """

    if not dicom_datasets:
        return {
            "title": "ACR QA Report",
            "status": "error",
            "message": "No DICOM datasets provided.",
            "plots": [],
            "sections": [],
        }

    ds0 = dicom_datasets[0]
    rows = int(getattr(ds0, "Rows", 0) or 0)
    cols = int(getattr(ds0, "Columns", 0) or 0)

    # Pull pixel data (first readable slice)
    pixel = None
    for ds in dicom_datasets:
        try:
            pixel = ds.pixel_array.astype(np.float32)
            ds0 = ds
            break
        except Exception:
            continue

    # Slice thickness from DICOM tag
    slice_thickness = _safe_float(getattr(ds0, "SliceThickness", None))
    st_status = "pass" if slice_thickness is not None else "na"

    # Pixel spacing / basic geometric check
    ps = getattr(ds0, "PixelSpacing", None)
    ps_y = ps_x = None
    if ps is not None and len(ps) >= 2:
        ps_y = _safe_float(ps[0])
        ps_x = _safe_float(ps[1])

    geom_value = None
    geom_status = "na"
    if ps_x and ps_y and rows and cols:
        fov_x = ps_x * cols
        fov_y = ps_y * rows
        geom_value = f"FOV ~ {fov_x:.1f} mm x {fov_y:.1f} mm (pixel {ps_x:.3f} x {ps_y:.3f} mm)"
        geom_status = "pass" if abs(ps_x - ps_y) <= max(ps_x, ps_y) * 0.02 else "fail"

    # Simple ROI metrics
    snr = piu = ghost = None
    snr_status = piu_status = ghost_status = "na"
    if pixel is not None and isinstance(pixel, np.ndarray) and pixel.ndim == 2 and rows and cols:
        cr0, cr1 = int(rows * 0.40), int(rows * 0.60)
        cc0, cc1 = int(cols * 0.40), int(cols * 0.60)
        center = _roi(pixel, cr0, cr1, cc0, cc1)
        c_mean = float(np.mean(center))
        c_std = float(np.std(center) + 1e-6)
        snr = c_mean / c_std
        snr_status = "pass" if snr >= 20 else "fail"

        # PIU using 4 peripheral ROIs
        q = []
        q.append(float(np.mean(_roi(pixel, int(rows * 0.20), int(rows * 0.35), int(cols * 0.20), int(cols * 0.35)))))
        q.append(float(np.mean(_roi(pixel, int(rows * 0.20), int(rows * 0.35), int(cols * 0.65), int(cols * 0.80)))))
        q.append(float(np.mean(_roi(pixel, int(rows * 0.65), int(rows * 0.80), int(cols * 0.20), int(cols * 0.35)))))
        q.append(float(np.mean(_roi(pixel, int(rows * 0.65), int(rows * 0.80), int(cols * 0.65), int(cols * 0.80)))))
        q_max, q_min = max(q), min(q)
        piu = 100.0 * (1.0 - (q_max - q_min) / (q_max + q_min + 1e-6))
        piu_status = "pass" if piu >= 85 else "fail"

        # Ghosting proxy
        corner = _roi(pixel, 0, int(rows * 0.10), 0, int(cols * 0.10))
        ghost = float(np.mean(corner) / (c_mean + 1e-6))
        ghost_status = "pass" if ghost <= 0.025 else "fail"

    # Plots (SVG)
    plots: list[dict[str, str]] = []
    if pixel is not None and isinstance(pixel, np.ndarray) and pixel.ndim == 2 and rows and cols:
        os.makedirs(plot_dir, exist_ok=True)

        # Ramp-style plot: two horizontal profiles
        y_top = int(rows * 0.45)
        y_bottom = int(rows * 0.55)
        prof_top = pixel[y_top, :].astype(np.float32)
        prof_bottom = pixel[y_bottom, :].astype(np.float32)
        x = np.arange(cols, dtype=float)
        name = f"ramp_{uuid.uuid4().hex}.svg"
        _write_svg_plot(
            os.path.join(plot_dir, name),
            title="MTF / Ramp Analysis",
            x=x,
            series=[("Top Ramp", prof_top), ("Bottom Ramp", prof_bottom)],
        )
        plots.append({"title": "MTF / ramp analysis", "url": f"/plots/{name}"})

        # Slice thickness profile proxy: center vertical line
        x2 = np.arange(rows, dtype=float)
        vprof = pixel[:, int(cols / 2)].astype(np.float32)
        name2 = f"slice_{uuid.uuid4().hex}.svg"
        _write_svg_plot(
            os.path.join(plot_dir, name2),
            title="Slice Thickness Profile",
            x=x2,
            series=[("Center Line", vprof)],
        )
        plots.append({"title": "Slice thickness", "url": f"/plots/{name2}"})

    # Build rows
    results_rows = [
        _metric_row(
            "Slice thickness",
            (f"{slice_thickness:.2f} mm" if slice_thickness is not None else None),
            "DICOM tag",
            st_status,
        ),
        _metric_row(
            "Geometric accuracy",
            geom_value,
            "Pixel spacing check",
            geom_status,
        ),
        _metric_row(
            "High-contrast resolution",
            None,
            "Not calculated",
            "na",
            "Algorithm can be added",
        ),
        _metric_row(
            "Low-contrast detectability",
            None,
            "Not calculated",
            "na",
            "Algorithm can be added",
        ),
        _metric_row(
            "Intensity uniformity (PIU)",
            (f"{piu:.2f}%" if piu is not None else None),
            ">= 85%",
            piu_status,
        ),
        _metric_row(
            "Ghosting",
            (f"{ghost:.4f}" if ghost is not None else None),
            "<= 0.025",
            ghost_status,
        ),
        _metric_row(
            "SNR",
            (f"{snr:.2f}" if snr is not None else None),
            ">= 20",
            snr_status,
        ),
        _metric_row(
            "MTF / ramp analysis",
            ("Plot generated" if plots else None),
            "See plot",
            ("pass" if plots else "na"),
        ),
    ]

    return {
        "title": "ACR QA Report",
        "status": "ok",
        "message": "Report generated.",
        "plots": plots,
        "sections": [
            {
                "name": "Input Summary",
                "kind": "kv",
                "rows": [
                    {"label": "DICOM files detected", "value": str(len(dicom_datasets))},
                    {"label": "Slices read", "value": str(len(dicom_datasets))},
                    {"label": "Image shape", "value": f"{rows} x {cols}"},
                ],
            },
            {
                "name": "QA Results",
                "kind": "metrics",
                "rows": results_rows,
            },
        ],
    }


def run_acr_qa(dicom_paths):
    """Entry point used by the Flask route.

    - dicom_paths: list of filesystem paths to potential DICOM files

    Loads readable datasets and returns the report dict.
    """

    import pydicom

    datasets = []
    for p in dicom_paths:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=False, force=True)
            # Basic filter: require pixel data for QA
            if hasattr(ds, "PixelData"):
                datasets.append(ds)
        except Exception:
            continue

    plot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plots"))
    return run_basic_metrics(datasets, plot_dir=plot_dir)
