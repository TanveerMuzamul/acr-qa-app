import json
import os
import io
from datetime import datetime

from flask import Blueprint, current_app, render_template, redirect, url_for, flash, request, send_file, jsonify
from flask_login import login_required, current_user

from app import db
from app.models import Job
from app.services.dicom_service import extract_zip_and_index, load_series_slice
from app.services.qa_metrics import (
    compute_metrics_for_slice,
    build_reasoned_results,
    phantom_likeness_from_image,
    acr_lcd_spokes_total,
)

main_bp = Blueprint("main", __name__)


def _job_by_share_token(token: str):
    if not token:
        return None
    try:
        return Job.query.filter_by(share_token=token).first()
    except Exception:
        return None


@main_bp.get("/")
def index():
    return render_template("index.html")


@main_bp.get("/dashboard")
@login_required
def dashboard():
    jobs = Job.query.filter_by(user_id=current_user.id).order_by(Job.created_at.desc()).all()
    # Backfill share tokens for older jobs (best-effort).
    changed = False
    for j in jobs:
        if not getattr(j, "share_token", None):
            try:
                j.ensure_share_token()
                changed = True
            except Exception:
                pass
    if changed:
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
    return render_template("dashboard.html", jobs=jobs)


@main_bp.post("/upload")
@login_required
def upload():
    if "zip_file" not in request.files:
        flash("No file uploaded.", "danger")
        return redirect(url_for("main.dashboard"))

    f = request.files["zip_file"]
    if not f or f.filename.strip() == "":
        flash("Please choose a ZIP file.", "warning")
        return redirect(url_for("main.dashboard"))

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    job_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], f"job_{current_user.id}_{ts}")
    os.makedirs(job_dir, exist_ok=True)

    zip_path = os.path.join(job_dir, "upload.zip")
    f.save(zip_path)

    try:
        index = extract_zip_and_index(zip_path, job_dir)
    except Exception as e:
        flash(f"Upload failed: {e}", "danger")
        return redirect(url_for("main.dashboard"))

    if index.get("dicom_files_detected", 0) == 0:
        flash("No DICOM files found inside the ZIP (nested folders are OK).", "danger")
        return redirect(url_for("main.dashboard"))

    job = Job(user_id=current_user.id, job_dir=job_dir, summary_json=json.dumps(index))
    # Create a share token so the user can send a link to others (read-only).
    job.ensure_share_token()
    db.session.add(job)
    db.session.commit()

    # Keep the UI clean: users can see the report immediately after redirect.
    # (We still flash errors on failures.)
    return redirect(url_for("main.report", job_id=job.id))


@main_bp.get("/report/<int:job_id>")
@login_required
def report(job_id: int):
    job = Job.query.get_or_404(job_id)
    if job.user_id != current_user.id:
        flash("Not allowed.", "danger")
        return redirect(url_for("main.dashboard"))

    index = json.loads(job.summary_json or "{}")
    if not index.get("series"):
        flash("No DICOM series found.", "danger")
        return redirect(url_for("main.dashboard"))

    # Default selection behavior
    # --------------------------
    # Users frequently upload whole studies. If we default to the merged __ALL__ series,
    # almost every ACR-oriented metric will look like a FAIL because we are mixing
    # different protocols / orientations / pixel spacings.
    #
    # To make the app usable out-of-the-box, when no explicit series is supplied we
    # auto-select the best phantom-like series using the same logic as /api/suggest.

    requested_series_uid = request.args.get("series")
    requested_slice = request.args.get("slice")

    # If caller explicitly requested a series, honor it.
    if requested_series_uid:
        default_series_uid = requested_series_uid
    else:
        # Choose a non-__ALL__ fallback first.
        non_all = [s for s in index.get("series", []) if s.get("series_uid") != "__ALL__"]
        default_series_uid = (non_all[0]["series_uid"] if non_all else index["series"][0]["series_uid"])

        # Try using a cached suggestion, otherwise compute one quickly.
        try:
            cache_dir = os.path.join(job.job_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, "suggestion.json")
            suggestion = None
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    suggestion = json.load(f)
            if not suggestion:
                # Lightweight suggestion (mirrors api_suggest but without a second request)
                series_list = list(index.get("series", []))
                series_list = [s for s in series_list if s.get("series_uid") != "__ALL__"] or series_list
                series_list = sorted(series_list, key=lambda s: -int(s.get("num_slices", 0)))[:12]
                best_uid = default_series_uid
                best_slice = 0
                best_score = 0.0
                from app.services.qa_metrics import phantom_likeness_from_image

                for s in series_list:
                    uid = s.get("series_uid")
                    n = int(s.get("num_slices", 0) or 0)
                    if not uid or n <= 0:
                        continue
                    candidates = {max(0, min(n - 1, n // 2)), max(0, min(n - 1, n // 4)), max(0, min(n - 1, (3 * n) // 4))}
                    for si in sorted(candidates):
                        try:
                            _, img = load_series_slice(job.job_dir, uid, int(si))
                            ph = phantom_likeness_from_image(img)
                            score = float(ph.get("phantom_score") or 0.0)
                        except Exception:
                            continue
                        if score > best_score:
                            best_score = score
                            best_uid = uid
                            best_slice = int(si)
                suggestion = {"series_uid": best_uid, "slice_idx": best_slice, "phantom_score": best_score}
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(suggestion, f)
                except Exception:
                    pass

            if isinstance(suggestion, dict) and suggestion.get("series_uid"):
                default_series_uid = suggestion.get("series_uid")
                if requested_slice is None:
                    requested_slice = str(int(suggestion.get("slice_idx") or 0))
        except Exception:
            pass

    series_obj = next((s for s in index["series"] if s["series_uid"] == default_series_uid), index["series"][0])
    max_idx = max(0, series_obj["num_slices"] - 1)
    slice_idx = int(requested_slice or (max_idx // 2))

    return render_template(
        "report.html",
        job=job,
        index=index,
        default_series_uid=series_obj["series_uid"],
        default_slice_idx=max(0, min(slice_idx, max_idx)),
    )


@main_bp.get("/share/<token>")
def share_report(token: str):
    """Public, read-only report page (no login required)."""
    job = _job_by_share_token(token)
    if not job:
        return render_template("error.html", title="Not found", message="Invalid or expired share link."), 404

    index = json.loads(job.summary_json or "{}")
    if not index.get("series"):
        return render_template("error.html", title="No data", message="No DICOM series found for this job."), 404

    # Use the same selection logic as the private report (but without caching suggestion via login).
    requested_series_uid = request.args.get("series")
    requested_slice = request.args.get("slice")

    non_all = [s for s in index.get("series", []) if s.get("series_uid") != "__ALL__"]
    default_series_uid = requested_series_uid or (non_all[0]["series_uid"] if non_all else index["series"][0]["series_uid"])

    series_obj = next((s for s in index["series"] if s["series_uid"] == default_series_uid), index["series"][0])
    max_idx = max(0, series_obj["num_slices"] - 1)
    slice_idx = int(requested_slice or (max_idx // 2))

    return render_template(
        "share_report.html",
        job=job,
        token=token,
        index=index,
        default_series_uid=series_obj["series_uid"],
        default_slice_idx=max(0, min(slice_idx, max_idx)),
    )


@main_bp.get("/api/report/<int:job_id>")
@login_required
def api_report(job_id: int):
    job = Job.query.get_or_404(job_id)
    if job.user_id != current_user.id:
        return jsonify({"error": "forbidden"}), 403

    index = json.loads(job.summary_json or "{}")
    series_uid = request.args.get("series")
    slice_idx = request.args.get("slice", "0")

    if not series_uid:
        return jsonify({"error": "series is required"}), 400
    if not slice_idx.isdigit():
        return jsonify({"error": "slice must be an integer"}), 400
    slice_idx = int(slice_idx)

    series_obj = next((s for s in index.get("series", []) if s["series_uid"] == series_uid), None)
    if not series_obj:
        return jsonify({"error": "series not found"}), 404

    slice_idx = max(0, min(slice_idx, series_obj["num_slices"] - 1))

    # Per-slice cache: avoids recomputing metrics on every UI refresh.
    cache_dir = os.path.join(job.job_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    safe_uid = series_uid.replace("/", "_")
    metrics_cache_path = os.path.join(cache_dir, f"metrics_{safe_uid}_{slice_idx}.json")

    raw = None
    results = None
    if os.path.exists(metrics_cache_path):
        try:
            with open(metrics_cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            raw = cached.get("raw")
            results = cached.get("results")
        except Exception:
            raw = None
            results = None

    if raw is None or results is None:
        ds, img = load_series_slice(job.job_dir, series_uid, slice_idx)
        raw = compute_metrics_for_slice(ds, img)

        # Determine series kind (T1 vs T2) for ACR LCD thresholds.
        desc_u = (series_obj.get("description") or "").upper()
        if "T2" in desc_u:
            raw["series_kind"] = "T2"
        elif "T1" in desc_u:
            raw["series_kind"] = "T1"
        else:
            # Fall back to TR if available
            try:
                tr = float(getattr(ds, "RepetitionTime", 0) or 0)
            except Exception:
                tr = 0.0
            raw["series_kind"] = "T1" if (tr and tr < 1000) else "T2"

        # Compute ACR LCD spoke count across slices 8–11 when possible (axial ACR series = 11 slices).
        # Cache per-series (not per-slice) to keep the UI snappy.
        lcd_cache_path = os.path.join(cache_dir, f"lcd_{safe_uid}.json")
        lcd_payload = None
        if os.path.exists(lcd_cache_path):
            try:
                with open(lcd_cache_path, "r", encoding="utf-8") as f:
                    lcd_payload = json.load(f)
            except Exception:
                lcd_payload = None

        if lcd_payload is None and int(series_obj.get("num_slices", 0)) >= 11:
            try:
                # 0-based indices for slices 8–11 -> 7..10
                imgs = []
                for si in range(7, 11):
                    _, im = load_series_slice(job.job_dir, series_uid, si)
                    imgs.append(im)
                lcd_payload = acr_lcd_spokes_total(imgs)
                with open(lcd_cache_path, "w", encoding="utf-8") as f:
                    json.dump(lcd_payload, f)
            except Exception:
                lcd_payload = None

        if isinstance(lcd_payload, dict):
            raw.update(lcd_payload)

        results = build_reasoned_results(raw)
        try:
            with open(metrics_cache_path, "w", encoding="utf-8") as f:
                json.dump({"raw": raw, "results": results}, f)
        except Exception:
            pass

    img_url = url_for("main.slice_png", job_id=job.id, series_uid=series_uid, slice_idx=slice_idx)

    return jsonify(
        {
            "dicom_files_detected": index.get("dicom_files_detected", 0),
            "series_found": len(index.get("series", [])),
            "series_uid": series_uid,
            "series_description": series_obj.get("description", ""),
            "num_slices_in_series": series_obj["num_slices"],
            "slice_idx": slice_idx,
            "image_shape": raw.get("image_shape"),
            "metrics": results.get("metrics", []),
            "overall_status": results.get("overall_status"),
            "overall_reason": results.get("overall_reason"),
            "image_url": img_url,
        }
    )


@main_bp.get("/api/share/report/<token>")
def api_share_report(token: str):
    """Public, read-only API for a shared job (no login)."""
    job = _job_by_share_token(token)
    if not job:
        return jsonify({"error": "not found"}), 404

    index = json.loads(job.summary_json or "{}")
    series_uid = request.args.get("series")
    slice_idx = request.args.get("slice", "0")

    if not series_uid:
        return jsonify({"error": "series is required"}), 400
    if not slice_idx.isdigit():
        return jsonify({"error": "slice must be an integer"}), 400
    slice_idx = int(slice_idx)

    series_obj = next((s for s in index.get("series", []) if s["series_uid"] == series_uid), None)
    if not series_obj:
        return jsonify({"error": "series not found"}), 404

    slice_idx = max(0, min(slice_idx, series_obj["num_slices"] - 1))

    # Reuse the same caching as the private API (on disk inside the job dir)
    cache_dir = os.path.join(job.job_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    safe_uid = series_uid.replace("/", "_")
    metrics_cache_path = os.path.join(cache_dir, f"metrics_{safe_uid}_{slice_idx}.json")

    raw = None
    results = None
    if os.path.exists(metrics_cache_path):
        try:
            with open(metrics_cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            raw = cached.get("raw")
            results = cached.get("results")
        except Exception:
            raw = None
            results = None

    if raw is None or results is None:
        ds, img = load_series_slice(job.job_dir, series_uid, slice_idx)
        raw = compute_metrics_for_slice(ds, img)

        desc_u = (series_obj.get("description") or "").upper()
        if "T2" in desc_u:
            raw["series_kind"] = "T2"
        elif "T1" in desc_u:
            raw["series_kind"] = "T1"
        else:
            try:
                tr = float(getattr(ds, "RepetitionTime", 0) or 0)
            except Exception:
                tr = 0.0
            raw["series_kind"] = "T1" if (tr and tr < 1000) else "T2"

        lcd_cache_path = os.path.join(cache_dir, f"lcd_{safe_uid}.json")
        lcd_payload = None
        if os.path.exists(lcd_cache_path):
            try:
                with open(lcd_cache_path, "r", encoding="utf-8") as f:
                    lcd_payload = json.load(f)
            except Exception:
                lcd_payload = None

        if lcd_payload is None and int(series_obj.get("num_slices", 0)) >= 11:
            try:
                imgs = []
                for si in range(7, 11):
                    _, im = load_series_slice(job.job_dir, series_uid, si)
                    imgs.append(im)
                lcd_payload = acr_lcd_spokes_total(imgs)
                with open(lcd_cache_path, "w", encoding="utf-8") as f:
                    json.dump(lcd_payload, f)
            except Exception:
                lcd_payload = None

        if isinstance(lcd_payload, dict):
            raw.update(lcd_payload)

        results = build_reasoned_results(raw)
        try:
            with open(metrics_cache_path, "w", encoding="utf-8") as f:
                json.dump({"raw": raw, "results": results}, f)
        except Exception:
            pass

    img_url = url_for("main.share_slice_png", token=token, series_uid=series_uid, slice_idx=slice_idx)

    return jsonify(
        {
            "dicom_files_detected": index.get("dicom_files_detected", 0),
            "series_found": len(index.get("series", [])),
            "series_uid": series_uid,
            "series_description": series_obj.get("description", ""),
            "num_slices_in_series": series_obj["num_slices"],
            "slice_idx": slice_idx,
            "image_shape": raw.get("image_shape"),
            "metrics": results.get("metrics", []),
            "overall_status": results.get("overall_status"),
            "overall_reason": results.get("overall_reason"),
            "image_url": img_url,
        }
    )




@main_bp.get("/api/suggest/<int:job_id>")
@login_required
def api_suggest(job_id: int):
    """Suggest the most phantom-like series + slice for a given job.

    Many users upload entire studies with multiple series. This helper picks a good default
    series/slice to start with, so the QA report is meaningful immediately.
    The result is cached on disk per-job.
    """
    job = Job.query.get_or_404(job_id)
    if job.user_id != current_user.id:
        return jsonify({"error": "forbidden"}), 403

    cache_dir = os.path.join(job.job_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "suggestion.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except Exception:
            pass

    index = json.loads(job.summary_json or "{}")
    series_list = list(index.get("series", []))

    # Exclude the merged series unless it's the only option.
    non_all = [s for s in series_list if s.get("series_uid") != "__ALL__"]
    if non_all:
        series_list = non_all

    # To keep this fast, only evaluate the largest N series.
    series_list = sorted(series_list, key=lambda s: -int(s.get("num_slices", 0)))[:12]

    best = {
        "series_uid": (series_list[0]["series_uid"] if series_list else None),
        "slice_idx": 0,
        "phantom_score": 0.0,
        "is_phantom_like": False,
        "series_description": (series_list[0].get("description") if series_list else ""),
    }

    for s in series_list:
        uid = s.get("series_uid")
        n = int(s.get("num_slices", 0))
        if not uid or n <= 0:
            continue

        candidates = {max(0, min(n - 1, n // 2)), max(0, min(n - 1, n // 4)), max(0, min(n - 1, (3 * n) // 4))}
        for si in sorted(candidates):
            try:
                ds, img = load_series_slice(job.job_dir, uid, int(si))
                ph = phantom_likeness_from_image(img)
                score = float(ph.get("phantom_score") or 0.0)
            except Exception:
                continue

            if score > float(best.get("phantom_score") or 0.0):
                best = {
                    "series_uid": uid,
                    "slice_idx": int(si),
                    "phantom_score": score,
                    "is_phantom_like": bool(ph.get("is_phantom_like")),
                    "series_description": s.get("description", ""),
                }

    # If we didn't find anything convincing, still return the biggest series mid-slice.
    if best.get("series_uid") is None and series_list:
        s0 = series_list[0]
        n0 = int(s0.get("num_slices", 1))
        best = {
            "series_uid": s0.get("series_uid"),
            "slice_idx": max(0, n0 // 2),
            "phantom_score": 0.0,
            "is_phantom_like": False,
            "series_description": s0.get("description", ""),
        }

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(best, f)
    except Exception:
        pass

    return jsonify(best)

@main_bp.get("/slice/<int:job_id>/<series_uid>/<int:slice_idx>.png")
@login_required
def slice_png(job_id: int, series_uid: str, slice_idx: int):
    job = Job.query.get_or_404(job_id)
    if job.user_id != current_user.id:
        return jsonify({"error": "forbidden"}), 403

    cache_dir = os.path.join(job.job_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    safe_uid = series_uid.replace("/", "_")
    png_path = os.path.join(cache_dir, f"slice_{safe_uid}_{slice_idx}.png")

    if not os.path.exists(png_path):
        # Fast PNG rendering (Pillow) to keep the UI responsive.
        ds, img = load_series_slice(job.job_dir, series_uid, slice_idx)
        import numpy as np
        from PIL import Image

        arr = np.asarray(img).astype(np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        arr8 = (arr * 255.0).clip(0, 255).astype(np.uint8)
        im = Image.fromarray(arr8, mode="L")
        im.save(png_path, format="PNG", optimize=True)

    return send_file(png_path, mimetype="image/png")


@main_bp.get("/share/slice/<token>/<series_uid>/<int:slice_idx>.png")
def share_slice_png(token: str, series_uid: str, slice_idx: int):
    """Public, read-only slice PNG for shared jobs."""
    job = _job_by_share_token(token)
    if not job:
        return jsonify({"error": "not found"}), 404

    cache_dir = os.path.join(job.job_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    safe_uid = series_uid.replace("/", "_")
    png_path = os.path.join(cache_dir, f"slice_{safe_uid}_{slice_idx}.png")

    if not os.path.exists(png_path):
        ds, img = load_series_slice(job.job_dir, series_uid, slice_idx)
        import numpy as np
        from PIL import Image

        arr = np.asarray(img).astype(np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        arr8 = (arr * 255.0).clip(0, 255).astype(np.uint8)
        im = Image.fromarray(arr8, mode="L")
        im.save(png_path, format="PNG", optimize=True)

    return send_file(png_path, mimetype="image/png")


@main_bp.get("/report/<int:job_id>/download.json")
@login_required
def download_report_json(job_id: int):
    """Download the currently selected series/slice report as JSON."""

    job = Job.query.get_or_404(job_id)
    if job.user_id != current_user.id:
        flash("Not allowed.", "danger")
        return redirect(url_for("main.dashboard"))

    index = json.loads(job.summary_json or "{}")
    if not index.get("series"):
        flash("No DICOM series found.", "danger")
        return redirect(url_for("main.dashboard"))

    series_uid = request.args.get("series_uid") or index["series"][0]["series_uid"]
    slice_index = int(request.args.get("slice_index", "0"))

    ds, img = load_series_slice(job.job_dir, series_uid, slice_index)
    raw = compute_metrics_for_slice(ds, img)

    # Series kind inference (T1/T2)
    series_obj = next((s for s in index.get("series", []) if s.get("series_uid") == series_uid), {})
    desc_u = (series_obj.get("description") or "").upper()
    if "T2" in desc_u:
        raw["series_kind"] = "T2"
    elif "T1" in desc_u:
        raw["series_kind"] = "T1"
    else:
        try:
            tr = float(getattr(ds, "RepetitionTime", 0) or 0)
        except Exception:
            tr = 0.0
        raw["series_kind"] = "T1" if (tr and tr < 1000) else "T2"

    # LCD spoke total (slices 8–11) if possible
    try:
        n = int(series_obj.get("num_slices", 0))
    except Exception:
        n = 0
    if n >= 11:
        try:
            imgs = []
            for si in range(7, 11):
                _, im = load_series_slice(job.job_dir, series_uid, si)
                imgs.append(im)
            raw.update(acr_lcd_spokes_total(imgs))
        except Exception:
            pass
    results = build_reasoned_results(raw)

    # Keep the payload stable and reusable: job + selection + metric values + pass/fail reasoning.
    payload = {
        "job": {
            "id": job.id,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            # NOTE: do not rely on optional DB columns (like "zip_name").
            # This project uses create_all() without migrations.
        },
        "selection": {"series_uid": series_uid, "slice_index": slice_index},
        # Keep both raw + reasoned results to make the file useful in other tools.
        "raw": raw,
        "overall_status": results.get("overall_status"),
        "overall_reason": results.get("overall_reason"),
        "metrics_table": results.get("metrics"),
    }

    buf = io.BytesIO(json.dumps(payload, indent=2).encode("utf-8"))
    filename = f"acr_qa_job_{job.id}_slice_{slice_index}.json"
    return send_file(buf, as_attachment=True, download_name=filename, mimetype="application/json")
