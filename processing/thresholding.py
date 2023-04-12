import time

from fastapi import UploadFile
from matplotlib import pyplot as plt
import cv2.cv2 as cv2
import pytesseract
import numpy as np
from PIL import Image

from processing.metrics import Metrics
from processing.utility import Utility
from difflib import SequenceMatcher


class Thresholding:
    @staticmethod
    def perform_test(reference_file: UploadFile, photo_file: UploadFile):
        reference_img = Utility.extract_image_gray(reference_file)
        photo_img = Utility.extract_image_gray(photo_file)
        reference_text = pytesseract.image_to_string(reference_img, config='--psm 6', lang='pol').strip()

        Thresholding.binarisation(photo_img, "binarisation")
        Thresholding._ocr("data/binarisation.png", reference_text)
        Thresholding.otsu(photo_img, "otsu")
        Thresholding._ocr("data/otsu.png", reference_text)

    @staticmethod
    def _ocr(filename: str, reference_text: str):
        img = np.array(Image.open(filename))
        text = pytesseract.image_to_string(img, config='--psm 6', lang='pol').strip()
        print(SequenceMatcher(None, reference_text, text).ratio())

    @staticmethod
    def binarisation(image, filename: str):
        start = time.time()
        ret, thresh = cv2.threshold(
            image, 127, 255, cv2.THRESH_BINARY)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        print("Time elapsed {} s".format(end - start))
        print("Noise level {}".format(Metrics.estimate_noise(thresh)))

        plt.imshow(thresh)
        plt.show()

    @staticmethod
    def otsu(image, filename: str):
        start = time.time()
        ret, thresh = cv2.threshold(
            image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        end = time.time()

        cv2.imwrite("data/" + filename + ".png", thresh)

        print("Time elapsed {} s".format(end - start))
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


