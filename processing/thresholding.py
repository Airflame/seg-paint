from fastapi import UploadFile
from matplotlib import pyplot as plt
import cv2
import time

from processing.utility import Utility


class Thresholding:
    @staticmethod
    def binarisation(file: UploadFile):
        ret, thresh = cv2.threshold(
            Utility.extract_image_binarisation(file), 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite("data/" + file.filename, thresh)

        plt.imshow(thresh)
        plt.show()

    @staticmethod
    def otsu(file: UploadFile):
        ret, thresh = cv2.threshold(
            Utility.extract_image_binarisation(file), 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite("data/" + file.filename, thresh)

        plt.imshow(thresh)
        plt.show()

    @staticmethod
    def adaptive(file: UploadFile):
        thresh = cv2.adaptiveThreshold(
            Utility.extract_image_binarisation(file), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 5)
        cv2.imwrite("data/" + file.filename, thresh)

        plt.imshow(thresh)
        plt.show()



