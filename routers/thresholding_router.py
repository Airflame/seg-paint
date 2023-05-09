from fastapi import APIRouter, UploadFile

from processing.thresholding import Thresholding
from processing.utility import Utility

router = APIRouter()


@router.post("/thresholding/naive", tags=["thresholding"])
async def upload_image_naive(file: UploadFile):
    return Thresholding.naive(Utility.extract_image_gray(file), "binarisation")


@router.post("/thresholding/otsu", tags=["thresholding"])
async def upload_image_otsu(file: UploadFile):
    return Thresholding.otsu(Utility.extract_image_gray(file), "otsu")


@router.post("/thresholding/adaptive-gaussian", tags=["thresholding"])
async def upload_image_adaptive_gaussian(file: UploadFile):
    return Thresholding.adaptive_gaussian(Utility.extract_image_gray(file), "adaptive-gaussian")


@router.post("/thresholding/adaptive-mean", tags=["thresholding"])
async def upload_image_adaptive_mean(file: UploadFile):
    return Thresholding.adaptive_mean(Utility.extract_image_gray(file), "adaptive-mean")


@router.post("/thresholding/niblack", tags=["thresholding"])
async def upload_image_niblack(file: UploadFile, block_size: int = 41, k: float = 0.2):
    return Thresholding.niblack(Utility.extract_image_gray(file), "niblack", block_size, k)


@router.post("/thresholding/sauvola", tags=["thresholding"])
async def upload_image_sauvola(file: UploadFile, block_size: int = 41, k: float = 0.2):
    return Thresholding.sauvola(Utility.extract_image_gray(file), "sauvola", block_size, k)


@router.post("/thresholding/wolf", tags=["thresholding"])
async def upload_image_wolf(file: UploadFile, block_size: int = 41, k: float = 0.2):
    return Thresholding.wolf(Utility.extract_image_gray(file), "wolf", block_size, k)


@router.post("/thresholding/nick", tags=["thresholding"])
async def upload_image_nick(file: UploadFile, block_size: int = 41, k: float = -0.2):
    return Thresholding.nick(Utility.extract_image_gray(file), "nick", block_size, k)


@router.post("/thresholding/test", tags=["thresholding"])
async def perform_test(reference_file: UploadFile, photo_file: UploadFile):
    Thresholding.perform_test(reference_file, photo_file)
    return {"filename": photo_file.filename}


@router.post("/thresholding/calibrate-method", tags=["thresholding"])
async def calibrate_method(reference_file: UploadFile,
                               photo_file_1: UploadFile,
                               photo_file_2: UploadFile,
                               photo_file_3: UploadFile,
                               photo_file_4: UploadFile,
                               photo_file_5: UploadFile,
                               method: str):
    Thresholding.calibrate_method(reference_file,
                                  [photo_file_1, photo_file_2, photo_file_3, photo_file_4, photo_file_5],
                                  method)
    return {"filename": photo_file_1.filename}

