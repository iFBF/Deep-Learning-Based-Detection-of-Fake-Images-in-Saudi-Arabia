"""HTTP contract for the portable local research service."""
from contextlib import asynccontextmanager
from pathlib import Path
import time

from fastapi import FastAPI, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse

from service_runtime import Runtime, InputError


def create_app(release, runner=None):
    release = Path(release)

    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = runner if runner is not None else Runtime(release)
        yield
        app.state.runtime = None

    application = FastAPI(title="Real-Abshar research prototype", version="1.0", lifespan=lifespan)

    @application.get("/", response_class=HTMLResponse)
    def index():
        return (release / "index.html").read_text()

    @application.get("/health")
    def health():
        return application.state.runtime.info()

    @application.post("/predict")
    async def predict(file: UploadFile):
        started = time.perf_counter()
        runtime = application.state.runtime
        limit = int(runtime.manifest["input_contract"]["max_file_bytes"])
        try:
            data = await file.read(limit + 1)
            if len(data) > limit:
                raise InputError("file_too_large", "The uploaded file exceeds the file-size limit.", 413)
            result = await run_in_threadpool(runtime.predict, data)
            response = JSONResponse(result)
            response.headers["X-Handler-Ms"] = f"{(time.perf_counter() - started) * 1000:.6f}"
            response.headers["Cache-Control"] = "no-store"
            return response
        except InputError as error:
            return JSONResponse({"error": {"code": error.code, "message": str(error)}}, status_code=error.status_code)
        except Exception:
            return JSONResponse({"error": {"code": "inference_failed", "message": "Inference failed; no authenticity decision is available."}}, status_code=500)
        finally:
            await file.close()

    return application
