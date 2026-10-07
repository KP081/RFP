from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, UploadFile, HTTPException, status

from app.services.document_parser import extract_text

router = APIRouter(prefix="/rfps", tags=["RPF"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE = 20 * 1024 * 1024


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF and DOCX files are supported.",
        )

    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size must be less than 20 MB.",
        )

    rfp_id = str(uuid4())

    saved_filename = f"{rfp_id}{extension}"
    file_path = UPLOAD_DIR / saved_filename

    file_path.write_bytes(content)

    try:
        text = extract_text(file_path=file_path)
    except Exception as e:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not extract document text: {e}",
        ) from e

    if not text.strip():
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The document contents on extractable text.",
        )

    return {
        "message": "RFP uploaded and text extracted successfully.",
        "rfp_id": rfp_id,
        "file_name": file.filename,
        "saved_path": str(file_path),
        "size_bytes": len(content),
        "text_length": len(text),
        "text_preview": text[:1000],
    }
