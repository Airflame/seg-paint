from fastapi import APIRouter, UploadFile

from processing.segmentation import Segmentation

router = APIRouter()


@router.post("/images/")
async def upload_image(file: UploadFile):
    Segmentation.k_means(file)
    return {"filename": file.filename}
