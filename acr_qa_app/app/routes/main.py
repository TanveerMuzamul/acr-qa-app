import os
import uuid
import json
from datetime import datetime
from flask import Blueprint, render_template, redirect, url_for, flash, current_app, send_file, send_from_directory, request
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from app import db
from app.models import UploadJob
from app.services.dicom_service import extract_dicom_from_zip, extract_zip_to_folder, find_dicom_files
from app.services.qa_metrics import run_acr_qa

main_bp = Blueprint("main", __name__)

@main_bp.get("/")
def index():
    return render_template("index.html")

@main_bp.get("/dashboard")
@login_required
def dashboard():
    jobs = UploadJob.query.filter_by(user_id=current_user.id).order_by(UploadJob.created_at.desc()).all()
    return render_template("dashboard.html", jobs=jobs)

@main_bp.post("/upload")
@login_required
def upload():
    file = request.files.get("zipfile")
    if not file or not file.filename:
        flash("Please choose a ZIP file.", "warning")
        return redirect(url_for("main.dashboard"))

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".zip"):
        flash("Upload must be a .zip file containing DICOM files.", "danger")
        return redirect(url_for("main.dashboard"))

    job_id = str(uuid.uuid4())
    upload_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], job_id)
    os.makedirs(upload_dir, exist_ok=True)

    zip_path = os.path.join(upload_dir, filename)
    file.save(zip_path)

    # Extract DICOM files from ZIP (supports nested folders)
    dicom_paths = extract_dicom_from_zip(zip_path, upload_dir)

    # Fallback: extract everything + scan (extra safety)
    if not dicom_paths:
        extract_zip_to_folder(zip_path, upload_dir)
        dicom_paths = find_dicom_files(upload_dir)

    if not dicom_paths:
        flash(
            "No DICOM files found inside the ZIP. Please ensure the ZIP contains MRI DICOM files (it can be nested in folders).",
            "danger",
        )
        return redirect(url_for("main.dashboard"))

    # Run QA (placeholder calculations for now)
    report = run_acr_qa(dicom_paths)

    # Save report under the configured reports folder so it always exists.
    report_path = os.path.join(current_app.config["REPORTS_FOLDER"], f"report_{job_id}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    job = UploadJob(
        user_id=current_user.id,
        original_filename=filename,
        stored_path=zip_path,
        report_path=report_path,
        created_at=datetime.utcnow(),
    )
    db.session.add(job)
    db.session.commit()

    flash("Upload processed. QA report generated.", "success")
    return redirect(url_for("main.view_report", job_id=job.id))
@main_bp.get("/report/<int:job_id>")
@login_required
def view_report(job_id: int):
    job = UploadJob.query.get_or_404(job_id)
    if job.user_id != current_user.id:
        flash("Not allowed.", "danger")
        return redirect(url_for("main.dashboard"))

    if not job.report_path or not os.path.exists(job.report_path):
        flash("Report not found yet.", "warning")
        return redirect(url_for("main.dashboard"))

    with open(job.report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    return render_template("report.html", job=job, report=report)

@main_bp.get("/report/<int:job_id>/download")
@login_required
def download_report(job_id: int):
    job = UploadJob.query.get_or_404(job_id)
    if job.user_id != current_user.id:
        flash("Not allowed.", "danger")
        return redirect(url_for("main.dashboard"))
    if not job.report_path or not os.path.exists(job.report_path):
        flash("Report not found.", "warning")
        return redirect(url_for("main.dashboard"))

    return send_file(job.report_path, as_attachment=True, download_name=f"acr_report_{job_id}.json")

@main_bp.get("/plots/<path:filename>")
def serve_plot(filename: str):
    """Serve generated plot files (SVG/PNG)."""
    plots_dir = current_app.config.get("PLOTS_FOLDER")
    if not plots_dir:
        plots_dir = os.path.join(current_app.root_path, "plots")
    return send_from_directory(plots_dir, filename)