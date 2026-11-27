"""Run the local FastAPI scorer and Streamlit review screen together."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from threading import Thread

import uvicorn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_HOST = "127.0.0.1"
API_PORT = 8000
STARTUP_TIMEOUT_SECONDS = 20


def main() -> int:
    server = uvicorn.Server(
        uvicorn.Config(
            "app.api:app",
            host=API_HOST,
            port=API_PORT,
            log_level="info",
        )
    )
    api_thread = Thread(target=server.run, name="kestrel-api", daemon=True)
    ui_process: subprocess.Popen[bytes] | None = None
    api_thread.start()

    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while not server.started:
            if not api_thread.is_alive():
                raise RuntimeError("The API server stopped before startup completed.")
            if time.monotonic() >= deadline:
                raise TimeoutError("The API server did not become ready in time.")
            time.sleep(0.1)

        print("API ready at http://127.0.0.1:8000; starting Streamlit UI.", flush=True)
        ui_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ROOT / "app" / "streamlit_app.py"),
                "--server.headless",
                "true",
            ],
            cwd=ROOT,
        )
        return ui_process.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        if ui_process is not None and ui_process.poll() is None:
            ui_process.terminate()
            ui_process.wait(timeout=10)
        server.should_exit = True
        api_thread.join(timeout=10)
        if api_thread.is_alive():
            raise RuntimeError("The API server did not shut down cleanly.")


if __name__ == "__main__":
    raise SystemExit(main())
