from fastapi import APIRouter, UploadFile

from processing.segmentation import Segmentation
from processing.thresholding import Thresholding
from processing.utility import Utility

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
    Thresholding.binarisation(Utility.extract_image_gray(file), "binarisation")
    return {"filename": file.filename}


@router.post("/images/otsu")
async def upload_image_otsu(file: UploadFile):
    Thresholding.otsu(Utility.extract_image_gray(file), "otsu")
    return {"filename": file.filename}


@router.post("/images/adaptive")
async def upload_image_adaptive(file: UploadFile):
    Thresholding.adaptive(file)
    return {"filename": file.filename}


@router.post("/images/niblack")
async def upload_image_niblack(file: UploadFile):
    Thresholding.niblack(file)
    return {"filename": file.filename}


@router.post("/images/tests")
async def perform_tests(reference_file: UploadFile, photo_file: UploadFile):
    Thresholding.perform_test(reference_file, photo_file)
    return {"filename": photo_file.filename}


@router.post("/images/contour")
async def upload_image_contour(file: UploadFile):
    Segmentation.contour(file)
    return {"filename": file.filename}
