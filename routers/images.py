from fastapi import APIRouter, UploadFile

from processing.segmentation import Segmentation
from processing.thresholding import Thresholding

router = APIRouter()


@router.post("/images/k-means")
async def upload_image(file: UploadFile):
    Segmentation.k_means(file)
    return {"filename": file.filename}


@router.post("/images/binarisation")
async def upload_image(file: UploadFile):
    Thresholding.binarisation(file)
    return {"filename": file.filename}
