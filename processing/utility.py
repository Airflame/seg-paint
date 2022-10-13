from fastapi import UploadFile
import numpy as np


class Utility:
    @staticmethod
    def extract_image(file: UploadFile):
        contents = file.file.read()
        return np.fromstring(contents, np.uint8)