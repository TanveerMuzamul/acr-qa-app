from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required
from sqlalchemy.exc import IntegrityError

from app import db
from app.forms import RegisterForm, LoginForm
from app.models import User

auth_bp = Blueprint("auth", __name__)


@auth_bp.get("/register")
def register():
    return render_template("auth/register.html", form=RegisterForm())


@auth_bp.post("/register")
def register_post():
    form = RegisterForm()
    if not form.validate_on_submit():
        return render_template("auth/register.html", form=form), 400

    # Normalize once to avoid subtle duplicates (e.g., trailing spaces).
    email = (form.email.data or "").strip().lower()

    if User.query.filter_by(email=email).first():
        flash("Email already registered. Please login.", "warning")
        return redirect(url_for("auth.login"))

    user = User(full_name=form.full_name.data.strip(), email=email)
    user.set_password(form.password.data)
    db.session.add(user)
    try:
        db.session.commit()
    except IntegrityError:
        # In case of a race or a previous DB state, fail gracefully.
        db.session.rollback()
        flash("Email already registered. Please login.", "warning")
        return redirect(url_for("auth.login"))

    flash("Account created. Please login.", "success")
    return redirect(url_for("auth.login"))


@auth_bp.get("/login")
def login():
    return render_template("auth/login.html", form=LoginForm())


@auth_bp.post("/login")
def login_post():
    form = LoginForm()
    if not form.validate_on_submit():
        return render_template("auth/login.html", form=form), 400

    email = (form.email.data or "").strip().lower()
    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(form.password.data):
        flash("Invalid email or password.", "danger")
        return render_template("auth/login.html", form=form), 401

    login_user(user)
    flash("Welcome back!", "success")
    return redirect(request.args.get("next") or url_for("main.dashboard"))


@auth_bp.get("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("main.index"))
