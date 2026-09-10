from __future__ import annotations

import os

from api.app import app

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("RADONCDOCS_HOST", "127.0.0.1")
    port = int(os.getenv("RADONCDOCS_PORT", "8000"))
    reload = os.getenv("RADONCDOCS_RELOAD", "1") == "1"
    uvicorn.run("api.app:app", host=host, port=port, reload=reload)
