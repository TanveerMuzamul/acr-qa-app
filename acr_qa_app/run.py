from app import create_app

app = create_app()

if __name__ == "__main__":
    # Use env-driven debug to avoid accidentally running in debug mode in production.
    import os

    debug = str(os.getenv("FLASK_DEBUG", "0")).lower() in {"1", "true", "yes"}
    app.run(host="127.0.0.1", port=5000, debug=debug)
