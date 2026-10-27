"""Start the web API and the Expo app, then wait until both stop."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from favlo.cli import repo_root
from favlo.daemon import port_open

API_HOST = "127.0.0.1"
API_PORT = 8130
WEB_HOST = "127.0.0.1"
WEB_PORT = 8081
_READY_SECONDS = 120.0


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        sys.stderr.write("usage: favlo web\n")
        return 2
    model_id = _registered_model()
    if model_id is None:
        sys.stderr.write("favlo web did not start. No model is registered.\n")
        return 1
    if port_open(API_HOST, API_PORT):
        sys.stderr.write(f"{API_PORT} is already in use\n")
        return 1
    if port_open(WEB_HOST, WEB_PORT):
        sys.stderr.write(f"{WEB_PORT} is already in use\n")
        return 1
    root = repo_root()
    spa = root / "web-spa"
    api = root / "web-api"
    if not (spa / "node_modules" / "expo").exists():
        sys.stderr.write("web-spa dependencies are missing. Run npm install in web-spa.\n")
        return 1
    _store_initial_model(model_id)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(api) + os.pathsep + env.get("PYTHONPATH", "")
    env["BROWSER"] = "none"
    env["EXPO_NO_TELEMETRY"] = "1"
    api_proc = _spawn(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", API_HOST, "--port", str(API_PORT)],
        api,
        env,
    )
    web_env = env.copy()
    web_env.pop("PYTHONPATH", None)
    web_env["NODE_OPTIONS"] = "--dns-result-order=ipv4first " + web_env.get("NODE_OPTIONS", "")
    web_proc = _spawn(
        ["npx", "expo", "start", "--web", "--port", str(WEB_PORT), "--host", "localhost"],
        spa,
        web_env,
    )
    try:
        if not _ready(api_proc, web_proc):
            return 1
        sys.stdout.write(f"favlo web is ready at http://{WEB_HOST}:{WEB_PORT}\n")
        sys.stdout.flush()
        return _wait(api_proc, web_proc)
    except KeyboardInterrupt:
        return 0
    finally:
        _stop(api_proc)
        _stop(web_proc)


def _registered_model() -> str | None:
    from favlo.registry import default_llm_id

    try:
        return default_llm_id()
    except LookupError:
        return None


def _store_initial_model(model_id: str) -> None:
    api = repo_root() / "web-api"
    if str(api) not in sys.path:
        sys.path.insert(0, str(api))
    from web_api.api import web_home
    from web_api.store import Store

    store = Store(web_home() / "app.db")
    try:
        store.ensure_initial_model(model_id)
    finally:
        store.close()


def _spawn(command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)


def _ready(api_proc: subprocess.Popen[bytes], web_proc: subprocess.Popen[bytes]) -> bool:
    deadline = time.monotonic() + _READY_SECONDS
    while time.monotonic() < deadline:
        if api_proc.poll() is not None:
            sys.stderr.write("web-api exited before it accepted a connection\n")
            return False
        if web_proc.poll() is not None:
            sys.stderr.write("the Expo dev server exited before it accepted a connection\n")
            return False
        if port_open(API_HOST, API_PORT) and port_open(WEB_HOST, WEB_PORT):
            return True
        time.sleep(0.2)
    sys.stderr.write(f"{API_PORT} or {WEB_PORT} did not accept a connection\n")
    return False


def _wait(api_proc: subprocess.Popen[bytes], web_proc: subprocess.Popen[bytes]) -> int:
    while True:
        api_code = api_proc.poll()
        web_code = web_proc.poll()
        if api_code is None and web_code is None:
            time.sleep(0.2)
            continue
        if api_code not in (None, 0) or web_code not in (None, 0):
            return 1
        return 0


def _stop(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            return
        proc.wait(timeout=5)
