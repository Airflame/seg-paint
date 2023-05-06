from fastapi import APIRouter, UploadFile

from processing.metrics import Metrics
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


@router.post("/images/adaptive-gaussian")
async def upload_image_adaptive_gaussian(file: UploadFile):
    Thresholding.adaptive_gaussian(Utility.extract_image_gray(file), "adaptive-gaussian")
    return {"filename": file.filename}


@router.post("/images/adaptive-mean")
async def upload_image_adaptive_mean(file: UploadFile):
    Thresholding.adaptive_mean(Utility.extract_image_gray(file), "adaptive-mean")
    return {"filename": file.filename}


@router.post("/images/niblack")
async def upload_image_niblack(file: UploadFile):
    Thresholding.niblack(Utility.extract_image_gray(file), "niblack")
    return {"filename": file.filename}


@router.post("/images/tests")
async def perform_tests(reference_file: UploadFile, photo_file: UploadFile):
    Thresholding.perform_test(reference_file, photo_file)
    return {"filename": photo_file.filename}


@router.post("/images/sauvola-test")
async def perform_sauvola_test(reference_file: UploadFile,
                               photo_file_1: UploadFile,
                               photo_file_2: UploadFile,
                               photo_file_3: UploadFile,
                               photo_file_4: UploadFile,
                               photo_file_5: UploadFile):
    Thresholding.perform_sauvola_test(reference_file,
                                      [photo_file_1, photo_file_2, photo_file_3, photo_file_4, photo_file_5])
    return {"filename": photo_file_1.filename}


@router.post("/images/contour")
async def upload_image_contour(file: UploadFile):
    Segmentation.contour(file)
    return {"filename": file.filename}


@router.post("/images/iou")
async def calculate_iou(image: UploadFile,
                        ground_truth: UploadFile):
    iou = Metrics.calculate_iou(Utility.extract_image(image), Utility.extract_image(ground_truth))
    return iou
