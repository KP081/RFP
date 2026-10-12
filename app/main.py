import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.deps import get_embedding_service, get_qdrant_service, get_rfp_service
from app.api.routes.rag import router as rag_router
from app.api.routes.rfp import router as rfp_router
from app.core.config import get_settings
from app.core.exceptions import AppError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_embedding_service()  # model loads once
    get_qdrant_service().create_collection()  # collection + index ready
    get_rfp_service().recover_interrupted()
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


api_router = APIRouter(prefix=settings.api_prefix)
api_router.include_router(rfp_router)
api_router.include_router(rag_router)
app.include_router(api_router)


@app.get("/health")
def health_check():
    qdrant_ok = get_qdrant_service().is_healthy()
    return {"status": "ok" if qdrant_ok else "degraded", "qdrant": qdrant_ok}
