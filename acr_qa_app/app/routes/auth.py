from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from app import db
from app.models import User
from app.forms import RegisterForm, LoginForm

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

@auth_bp.get("/register")
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    form = RegisterForm()
    return render_template("auth/register.html", form=form)

@auth_bp.post("/register")
def register_post():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    form = RegisterForm()
    if not form.validate_on_submit():
        return render_template("auth/register.html", form=form), 400

    existing = User.query.filter_by(email=form.email.data.lower().strip()).first()
    if existing:
        flash("That email is already registered. Please log in.", "warning")
        return redirect(url_for("auth.login"))

    user = User(
        email=form.email.data.lower().strip(),
        full_name=form.full_name.data.strip(),
    )
    user.set_password(form.password.data)
    db.session.add(user)
    db.session.commit()

    flash("Account created. Please log in.", "success")
    return redirect(url_for("auth.login"))

@auth_bp.get("/login")
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    form = LoginForm()
    return render_template("auth/login.html", form=form)

@auth_bp.post("/login")
def login_post():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    form = LoginForm()
    if not form.validate_on_submit():
        return render_template("auth/login.html", form=form), 400

    user = User.query.filter_by(email=form.email.data.lower().strip()).first()
    if not user or not user.check_password(form.password.data):
        flash("Invalid email or password.", "danger")
        return redirect(url_for("auth.login"))

    login_user(user)
    next_url = request.args.get("next")
    return redirect(next_url or url_for("main.dashboard"))

@auth_bp.get("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("auth.login"))
