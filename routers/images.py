from fastapi import APIRouter, UploadFile

from processing.segmentation import Segmentation
from processing.thresholding import Thresholding

router = APIRouter()


@router.post("/images/k-means")
async def upload_image(file: UploadFile):
    Segmentation.k_means(file)
    return {"filename": file.filename}


@router.post("/images/watershed")
async def upload_image(file: UploadFile):
    Segmentation.watershed(file)
    return {"filename": file.filename}


@router.post("/images/binarisation")
async def upload_image_binarisation(file: UploadFile):
    Thresholding.binarisation(file)
    return {"filename": file.filename}


@router.post("/images/otsu")
async def upload_image_otsu(file: UploadFile):
    Thresholding.otsu(file)
    return {"filename": file.filename}


@router.post("/images/adaptive")
async def upload_image_adaptive(file: UploadFile):
    Thresholding.adaptive(file)
    return {"filename": file.filename}


@router.post("/images/niblack")
async def upload_image_niblack(file: UploadFile):
    Thresholding.niblack(file)
    return {"filename": file.filename}
