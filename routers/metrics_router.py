from fastapi import APIRouter, UploadFile

from processing.metrics import Metrics
from processing.utility import Utility

router = APIRouter()


@router.post("/metrics/calculate-iou", tags=["metrics"])
async def calculate_iou(image: UploadFile,
                        ground_truth: UploadFile):
    iou = Metrics.calculate_iou(Utility.extract_image(image), Utility.extract_image(ground_truth))
    return iou
