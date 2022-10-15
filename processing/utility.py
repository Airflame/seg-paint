import cv2
from fastapi import UploadFile
import numpy as np


class Utility:
    @staticmethod
    def extract_image_binarisation(file: UploadFile):
        img = cv2.imdecode(Utility.extract_image(file), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def extract_image(file: UploadFile):
        contents = file.file.read()
        return np.fromstring(contents, np.uint8)
