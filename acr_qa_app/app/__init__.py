import os
from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "auth.login_get"


def create_app():
    # Load environment values from .env (if exists)
    load_dotenv()

    app = Flask(__name__, static_folder="static")

    # Core config
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev_secret_key_change_me")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///acr_qa.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Upload size limit (in bytes). Adjust if needed.
    app.config["MAX_CONTENT_LENGTH"] = 800 * 1024 * 1024  # 800 MB

    # Upload size limit (500 MB)
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

    # Paths
    base_dir = os.path.abspath(os.path.dirname(__file__))
    upload_folder = os.path.join(base_dir, "uploads")
    reports_folder = os.path.join(base_dir, "reports")
    plots_folder = os.path.join(base_dir, "plots")

    app.config["UPLOAD_FOLDER"] = upload_folder
    app.config["REPORTS_FOLDER"] = reports_folder
    app.config["PLOTS_FOLDER"] = plots_folder

    # Ensure folders exist
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(reports_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    # Init extensions
    db.init_app(app)
    login_manager.init_app(app)

    # Import models
    from app.models import User  # noqa: F401

    with app.app_context():
        db.create_all()

    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.main import main_bp

    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(main_bp)

    return app
