"""Admin blueprint for rollback operations."""
from __future__ import annotations

import ipaddress
from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from services.rollback_service import RollbackManager, RollbackError

bp = Blueprint("admin_rollback", __name__, url_prefix="/admin/rollback")


def _allowed_ip(remote_addr: str) -> bool:
    cidrs = current_app.config.get("ALLOW_ADMIN_ROLLBACK_IPS", "")
    if not cidrs:
        return True
    try:
        ip = ipaddress.ip_address(remote_addr)
    except ValueError:
        return False
    for raw in cidrs.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            network = ipaddress.ip_network(raw, strict=False)
        except ValueError:
            continue
        if ip in network:
            return True
    return False


@bp.before_request
def _auth_checks() -> None:
    if not session.get("role"):
        abort(403)
    if not _allowed_ip(request.remote_addr or "0.0.0.0"):
        abort(403)


def _ensure_csrf() -> None:
    token = session.get("_csrf_token")
    if not token or token != request.form.get("csrf_token"):
        abort(400)


@bp.route("/", methods=["GET", "POST"])
def index():  # type: ignore[override]
    manager = RollbackManager()
    snapshots = manager.manifest.load()
    confirm_snapshot = None
    error = None
    if request.method == "POST":
        _ensure_csrf()
        snapshot_id = request.form.get("snapshot_id") or (snapshots[0].snapshot_id if snapshots else None)
        step = request.form.get("step", "start")
        if step == "start":
            confirm_snapshot = snapshot_id
        elif step == "execute" and snapshot_id:
            try:
                manager.restore_snapshot(snapshot_id, reason="manual", auto=False)
                flash("ロールバックを開始しました", "success")
                return redirect(url_for("admin_rollback.index"))
            except RollbackError as exc:
                error = str(exc)
    return render_template(
        "admin/rollback.html",
        snapshots=snapshots,
        confirm_snapshot=confirm_snapshot,
        error=error,
    )


@bp.route("/api/restore", methods=["POST"])
def api_restore():
    if request.content_type and "application/json" not in request.content_type:
        abort(415)
    payload = request.get_json(force=True)
    snapshot_id = payload.get("snapshot_id")  # type: ignore[assignment]
    auto = bool(payload.get("auto", False))
    reason = payload.get("reason", "api")  # type: ignore[assignment]
    manager = RollbackManager()
    try:
        entry = manager.restore_snapshot(snapshot_id, reason=reason, auto=auto) if snapshot_id else manager.restore_latest_backup(reason=reason, auto=auto)
    except RollbackError as exc:
        return jsonify({"status": "error", "error": str(exc)}), 400
    return jsonify({"status": "ok", "snapshot": entry.to_dict()})


@bp.route("/snapshots", methods=["GET"])
def list_snapshots():
    manager = RollbackManager()
    entries = [item.to_dict() for item in manager.manifest.load()]
    return jsonify({"snapshots": entries})
