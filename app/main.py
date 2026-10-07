from fastapi import FastAPI

from app.api.routes.rfp import router as rfp_router

app = FastAPI(title="RFP RAG API", version="1.0.0")

app.include_router(rfp_router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
