ACR QA App (MRI)

Flask web application to upload MRI DICOM ZIP archives and review QA metrics per series and per slice.

What it does
- Upload ZIP files (nested folders supported)
- Groups DICOM images into series
- Select series + slice to inspect QA
- Shows PASS/FAIL metrics and reasons
- Clean medical dashboard UI

Run locally (Windows)
If your organization blocks pip inside venv, install dependencies to user site-packages.

py -m pip install --user -r acr_qa_app\requirements.txt
copy acr_qa_app\.env.example acr_qa_app\.env
python acr_qa_app\run.py