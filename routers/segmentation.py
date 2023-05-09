from fastapi import APIRouter, UploadFile

from processing.metrics import Metrics
from processing.segmentation import Segmentation
from processing.utility import Utility

router = APIRouter()


@router.post("/segmentation/k-means", tags=["segmentation"])
async def k_means(file: UploadFile):
    Segmentation.k_means(file)
    return {"filename": file.filename}


@router.post("/segmentation/watershed", tags=["segmentation"])
async def watershed(file: UploadFile):
    Segmentation.watershed(file)
    return {"filename": file.filename}


@router.post("/segmentation/contour", tags=["segmentation"])
async def contour(file: UploadFile):
    Segmentation.contour(file)
    return {"filename": file.filename}


@router.post("/segmentation/calculate-iou", tags=["segmentation"])
async def calculate_iou(image: UploadFile,
                        ground_truth: UploadFile):
    iou = Metrics.calculate_iou(Utility.extract_image(image), Utility.extract_image(ground_truth))
    return iou
