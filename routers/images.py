import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile

from processing.segmentation import Segmentation

router = APIRouter()


@router.post("/images/")
async def upload_image(file: UploadFile):
    filename = file.filename
    path = Path("data/"+filename)
    with path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    file.file.close()
    Segmentation.k_means(path)
    return {"filename": file.filename}
