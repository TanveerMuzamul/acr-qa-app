import os
import zipfile
from typing import List

import pydicom


def extract_zip_to_folder(zip_path: str, out_dir: str) -> None:
    """Extract ZIP safely into out_dir."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # Basic zip-slip protection
            target_path = os.path.abspath(os.path.join(out_dir, member.filename))
            base_dir = os.path.abspath(out_dir) + os.sep
            if not target_path.startswith(base_dir):
                continue
            zf.extract(member, out_dir)


def _looks_like_dicom(path: str) -> bool:
    """Best-effort DICOM detection.

    Many MRI exports do NOT use the .dcm extension (may be .ima, .img, no extension, etc).
    We therefore do not rely on file extensions.

    Strategy:
      1) Quick header check for 'DICM' at offset 128 (fast path).
      2) Fallback: try to read metadata with pydicom (force=True, no pixels) and
         accept if it contains typical DICOM identifiers.
    """
    try:
        with open(path, "rb") as f:
            preamble = f.read(132)
        if len(preamble) >= 132 and preamble[128:132] == b"DICM":
            return True
    except OSError:
        return False

    # Fallback: some valid DICOM files don't have the 'DICM' marker.
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True, specific_tags=[
            "SOPClassUID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "SOPInstanceUID",
            "PatientID",
            "Modality",
        ])
        for key in ("SOPClassUID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "Modality"):
            if getattr(ds, key, None):
                return True
        return False
    except Exception:
        return False


def find_dicom_files(root_dir: str) -> List[str]:
    """Return paths that are likely DICOM files (recursive).

    - Skips nested ZIP files
    - Detects DICOM by header/metadata rather than extension
    """
    paths: List[str] = []
    for folder, _, files in os.walk(root_dir):
        for name in files:
            lower = name.lower()
            if lower.endswith(".zip"):
                continue
            full = os.path.join(folder, name)
            if _looks_like_dicom(full):
                paths.append(full)
    return paths


def extract_dicom_from_zip(zip_path: str, out_dir: str) -> List[str]:
    """Extract a ZIP to out_dir and return a list of detected DICOM file paths.

    Convenience wrapper used by the upload pipeline.
    """
    extract_zip_to_folder(zip_path, out_dir)
    return find_dicom_files(out_dir)
