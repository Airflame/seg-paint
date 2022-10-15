import cv2
from fastapi import UploadFile
import numpy as np


class Utility:
    @staticmethod
    def extract_image_gray(file: UploadFile):
        img = Utility.extract_image(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def extract_image(file: UploadFile):
        contents = file.file.read()
        img = cv2.imdecode(np.fromstring(contents, np.uint8), cv2.IMREAD_COLOR)
        return img
