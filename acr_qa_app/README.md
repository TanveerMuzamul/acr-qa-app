# ACR QA Web App

This repository is a **Flask app scaffold** for a Cloud MRI ACR QA pipeline.

It includes:
- Register / Login / Logout (Flask-Login)
- Upload a **ZIP** of DICOM files
- End-to-end QA runner that generates a JSON report
- Light UI (Bootstrap 5)

>  The QA logic in `app/services/qa_metrics.py` 
> Replace each metric function with the real ACR implementation as the project progresses.

---

## 1) Run locally (Windows / macOS / Linux)

### A. Create venv and install
```bash
cd acr_qa_app
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### B. Create `.env`
Copy `.env.example` to `.env` and edit:
```bash
copy .env.example .env   # Windows
cp .env.example .env     # macOS/Linux
```

Example `.env`:
```env
FLASK_SECRET_KEY=change-me
DATABASE_URL=sqlite:///app.db
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH_MB=200
```

### C. Start the app
```bash
python run.py
```

Open: http://127.0.0.1:5000

---

## 2) How uploads work

- Upload **one** `.zip` file
- The ZIP should contain DICOM files:
  - `.dcm` is fine
  - files **without extension** can also work (pydicom reads with `force=True`)

Reports are saved to:
- `reports/report_<uuid>.json`

---

## 3) GitHub workflow (simple)

Suggested workflow:
1. One person creates the repo on GitHub
2. Everyone clones
3. Create a branch per feature:
   - `feature/auth`
   - `feature/upload`
   - `feature/qa-metrics`
4. Open PRs and review before merging to `main`

---

## 4) Where to edit QA logic

`app/services/qa_metrics.py`

Recommended next steps:
- Group DICOM files into ACR T1 / ACR T2 / Localizer using header tags
- Confirm slice mapping (which slice is used for which test)
- Implement real geometric accuracy, slice position, LCD scoring, etc.
