Windows quick run

1) Open CMD in this folder (the folder that contains the 'acr_qa_app' directory).
2) Create venv:
   python -m venv .venv
3) Activate:
   .venv\Scripts\activate
4) Install:
   pip install -r requirements.txt
5) Copy env:
   copy acr_qa_app\.env.example acr_qa_app\.env
6) Run:
   python acr_qa_app\run.py

Open: http://127.0.0.1:5000
