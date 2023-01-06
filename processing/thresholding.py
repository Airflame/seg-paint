from fastapi import UploadFile
from matplotlib import pyplot as plt
import cv2.cv2 as cv2

from processing.metrics import Metrics
from processing.utility import Utility


class Thresholding:
    @staticmethod
    def binarisation(file: UploadFile):
        ret, thresh = cv2.threshold(
            Utility.extract_image_gray(file), 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite("data/" + file.filename, thresh)

        print("Noise level {}".format(Metrics.estimate_noise(thresh)))

        plt.imshow(thresh)
        plt.show()

    @staticmethod
    def otsu(file: UploadFile):
        ret, thresh = cv2.threshold(
            Utility.extract_image_gray(file), 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite("data/" + file.filename, thresh)

        print("Noise level {}".format(Metrics.estimate_noise(thresh)))

        plt.imshow(thresh)
        plt.show()

    @staticmethod
    def adaptive(file: UploadFile):
        thresh = cv2.adaptiveThreshold(
            Utility.extract_image_gray(file), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 5)
        cv2.imwrite("data/" + file.filename, thresh)

        print("Noise level {}".format(Metrics.estimate_noise(thresh)))

        plt.imshow(thresh)
        plt.show()

    @staticmethod
    def niblack(file: UploadFile):
        thresh = cv2.ximgproc.niBlackThreshold(
            Utility.extract_image_gray(file), 255, cv2.THRESH_BINARY, 11, 0)
        cv2.imwrite("data/" + file.filename, thresh)

        print("Noise level {}".format(Metrics.estimate_noise(thresh)))

        plt.imshow(thresh)
        plt.show()


