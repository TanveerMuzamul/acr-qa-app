from app import create_app

app = create_app()

if __name__ == "__main__":
    # Dev run:
    #   python run.py
    app.run(debug=True)
