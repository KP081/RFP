from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile, status

from app.api.deps import get_rfp_service
from app.schemas.rfp import RFPResponse
from app.services.rfp_service import RFPService

router = APIRouter(prefix="/rfps", tags=["RFPs"])


@router.post("", response_model=RFPResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_rfp(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    service: RFPService = Depends(get_rfp_service),
):
    content = await file.read()
    record = service.create_from_upload(file.filename or "", content)

    # Response go fast, ingestion runs in background.
    background_tasks.add_task(service.process, record.rfp_id)

    return record


@router.get("", response_model=list[RFPResponse])
def list_rfps(service: RFPService = Depends(get_rfp_service)):
    return service.list_all()


@router.get("/{rfp_id}", response_model=RFPResponse)
def get_rfp(rfp_id: str, service: RFPService = Depends(get_rfp_service)):
    return service.get(rfp_id)


@router.post(
    "/{rfp_id}/reingest",
    response_model=RFPResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def reingest_rfp(
    rfp_id: str,
    background_tasks: BackgroundTasks,
    service: RFPService = Depends(get_rfp_service),
):
    record = service.reingest(rfp_id)
    background_tasks.add_task(service.process, rfp_id)
    return record


@router.delete("/{rfp_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_rfp(rfp_id: str, service: RFPService = Depends(get_rfp_service)):
    service.delete(rfp_id)
