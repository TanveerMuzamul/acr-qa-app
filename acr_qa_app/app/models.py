from datetime import datetime
import secrets
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from app import db, login_manager


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(180), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, raw: str) -> None:
        self.password_hash = generate_password_hash(raw)

    def check_password(self, raw: str) -> bool:
        return check_password_hash(self.password_hash, raw)


class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    job_dir = db.Column(db.String(500), nullable=False)
    summary_json = db.Column(db.Text, nullable=True)

    # Public sharing link (optional). When set, anyone with the token can view a read-only report.
    share_token = db.Column(db.String(64), unique=True, nullable=True, index=True)

    def ensure_share_token(self) -> str:
        if not self.share_token:
            # URL-safe token
            self.share_token = secrets.token_urlsafe(24)
        return self.share_token

    user = db.relationship("User", backref=db.backref("jobs", lazy=True))


@login_manager.user_loader
def load_user(user_id: str):
    return User.query.get(int(user_id))
