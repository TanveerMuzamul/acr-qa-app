import os
from dotenv import load_dotenv
from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "auth.login"


def create_app() -> Flask:
    load_dotenv()

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///app.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    upload_folder = os.getenv("UPLOAD_FOLDER", "instance/uploads")
    app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "..", upload_folder)
    # Upload size cap (Flask will return HTTP 413 if exceeded).
    # DICOM ZIPs can be large, so we default to 1024MB.
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("UPLOAD_MAX_MB", os.getenv("MAX_CONTENT_LENGTH_MB", "1024"))) * 1024 * 1024
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    from app.models import User  # noqa: F401

    with app.app_context():
        db.create_all()

    from app.routes.auth import auth_bp
    from app.routes.main import main_bp
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(main_bp)

    # Friendly message for large uploads (instead of a blank 413 page)
    @app.errorhandler(413)
    def _upload_too_large(_e):
        from flask import render_template

        return (
            render_template(
                "error.html",
                title="Upload too large",
                message=(
                    "Your ZIP file is larger than the server upload limit. "
                    "Increase UPLOAD_MAX_MB in acr_qa_app/.env (example: 1024), "
                    "or upload a smaller ZIP for testing."
                ),
            ),
            413,
        )

    return app
