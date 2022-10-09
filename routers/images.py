import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile

router = APIRouter()


@router.post("/images/")
async def upload_image(file: UploadFile):
    path = Path("data/"+file.filename)
    with path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    file.file.close()
    return {"filename": file.filename}
