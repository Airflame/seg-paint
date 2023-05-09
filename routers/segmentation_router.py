from fastapi import APIRouter, UploadFile

from processing.segmentation import Segmentation

router = APIRouter()


@router.post("/segmentation/watershed", tags=["segmentation"])
async def upload_image_watershed(file: UploadFile):
    return Segmentation.watershed(file)


@router.post("/segmentation/contour", tags=["segmentation"])
async def upload_image_contour(file: UploadFile):
    return Segmentation.contour(file)
