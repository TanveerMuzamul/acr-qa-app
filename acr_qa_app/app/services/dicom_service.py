import json
import os
import zipfile
from typing import Dict, List, Tuple, Any

import pydicom
from pydicom.misc import is_dicom


def _is_dicom(path: str) -> bool:
    # Fast header check (avoids parsing the full dataset).
    # Falls back to a minimal read when the file doesn't have a preamble.
    try:
        if is_dicom(path):
            return True
    except Exception:
        pass
    try:
        pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def _safe_extract_zip(zip_path: str, extract_dir: str) -> None:
    """Safely extract ZIP files (protect against path traversal / zip-slip)."""

    os.makedirs(extract_dir, exist_ok=True)
    base = os.path.realpath(extract_dir)

    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            # Skip directories.
            if member.is_dir():
                continue
            # Ensure the final extraction path stays within extract_dir.
            target_path = os.path.realpath(os.path.join(extract_dir, member.filename))
            if not target_path.startswith(base + os.sep) and target_path != base:
                raise ValueError(f"Unsafe ZIP entry detected: {member.filename}")
        z.extractall(extract_dir)


def extract_zip_and_index(zip_path: str, job_dir: str) -> Dict[str, Any]:
    extract_dir = os.path.join(job_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    _safe_extract_zip(zip_path, extract_dir)

    # Collect DICOM files (nested folders are OK)
    dicom_files: List[str] = []
    for root, _, files in os.walk(extract_dir):
        for fn in files:
            p = os.path.join(root, fn)
            if _is_dicom(p):
                dicom_files.append(p)

    # Performance note:
    # For large uploads (hundreds/thousands of slices), indexing time is dominated by
    # DICOM header reads. We therefore read each file header ONCE (stop_before_pixels)
    # and cache the minimal fields we need for series grouping + ordering.
    series_map: Dict[str, Dict[str, Any]] = {}
    file_meta: Dict[str, Dict[str, Any]] = {}

    needed_tags = [
        "SeriesInstanceUID",
        "SeriesDescription",
        "InstanceNumber",
        "ImagePositionPatient",
        "Modality",
        "MagneticFieldStrength",
        "StudyDescription",
        "SeriesNumber",
    ]

    detected_modality = None
    detected_field_strength_t = None

    for p in dicom_files:
        try:
            ds = pydicom.dcmread(
                p,
                stop_before_pixels=True,
                force=True,
                specific_tags=needed_tags,
            )
        except Exception:
            continue

        # Capture a couple of study-level properties once (used for correct pass/fail rules).
        if detected_modality is None:
            detected_modality = getattr(ds, "Modality", None)
        if detected_field_strength_t is None:
            try:
                mfs = getattr(ds, "MagneticFieldStrength", None)
                detected_field_strength_t = float(mfs) if mfs is not None else None
            except Exception:
                detected_field_strength_t = None

        uid = str(getattr(ds, "SeriesInstanceUID", "UNKNOWN_SERIES"))
        desc = str(getattr(ds, "SeriesDescription", "")) if hasattr(ds, "SeriesDescription") else ""
        inst = getattr(ds, "InstanceNumber", None)
        ipp = getattr(ds, "ImagePositionPatient", None)

        ippz = None
        try:
            if ipp is not None and len(ipp) >= 3:
                ippz = float(ipp[2])
        except Exception:
            ippz = None

        file_meta[p] = {
            "uid": uid,
            "desc": desc,
            "inst": int(inst) if inst is not None else None,
            "ippz": ippz,
            "basename": os.path.basename(p),
        }

        series_map.setdefault(uid, {"series_uid": uid, "description": desc, "files": []})["files"].append(p)

    # Also create a merged "ALL" series so users can browse every DICOM file
    # they uploaded, even when the dataset contains multiple series.
    # This avoids the common confusion: "I uploaded 26 files but I only see 11".
    series_map["__ALL__"] = {
        "series_uid": "__ALL__",
        "description": "All detected DICOM files (merged)",
        "files": list(dicom_files),
    }

    def sort_key(p: str):
        m = file_meta.get(p, {})
        inst = m.get("inst")
        if inst is not None:
            return (0, inst)
        ippz = m.get("ippz")
        if ippz is not None:
            return (1, ippz)
        return (2, m.get("basename") or os.path.basename(p))

    series_list: List[Dict[str, Any]] = []
    for uid, entry in series_map.items():
        files_sorted = sorted(entry["files"], key=sort_key)
        series_list.append(
            {
                "series_uid": uid,
                "description": entry.get("description", ""),
                "num_slices": len(files_sorted),
                "files": [os.path.relpath(p, job_dir) for p in files_sorted],
            }
        )

    series_list.sort(key=lambda s: (-s["num_slices"], s.get("description", "")))

    # Prefer showing the merged series LAST in the UI (to avoid accidental scoring on mixed studies).
    series_list.sort(key=lambda s: (1 if s["series_uid"] == "__ALL__" else 0, -s["num_slices"]))

    # Write a cache mapping so subsequent slice loads don't need to rescan the
    # extracted directory (big speedup for large uploads).
    cache_path = os.path.join(job_dir, "cache_series_files.json")
    mapping_abs: Dict[str, List[str]] = {
        s["series_uid"]: [os.path.join(job_dir, rel) for rel in s["files"]] for s in series_list
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(mapping_abs, f)

    # Persist metadata so other parts of the app can use it without re-reading headers.
    meta_path = os.path.join(job_dir, "cache", "job_meta.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "detected_modality": detected_modality,
                "detected_field_strength_t": detected_field_strength_t,
            },
            f,
        )

    return {
        "dicom_files_detected": len(dicom_files),
        "series": series_list,
        "detected_modality": detected_modality,
        "detected_field_strength_t": detected_field_strength_t,
    }


def load_series_slice(job_dir: str, series_uid: str, slice_idx: int) -> Tuple[pydicom.dataset.FileDataset, Any]:
    cache_path = os.path.join(job_dir, "cache_series_files.json")
    series_files: Dict[str, List[str]] = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            series_files = json.load(f)

    # Backward compatibility: if older jobs were created without an __ALL__ entry,
    # synthesize it from the remaining series.
    if series_uid == "__ALL__" and "__ALL__" not in series_files and series_files:
        merged: List[str] = []
        for uid, flist in series_files.items():
            if uid == "__ALL__":
                continue
            merged.extend(list(flist))
        series_files["__ALL__"] = merged

    files = series_files.get(series_uid, [])
    if not files:
        raise FileNotFoundError(f"Series not found: {series_uid}")

    slice_idx = max(0, min(slice_idx, len(files) - 1))
    ds = pydicom.dcmread(files[slice_idx], force=True)
    img = ds.pixel_array
    return ds, img