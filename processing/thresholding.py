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
        Thresholding.adaptive(photo_img, "adaptive")
        Thresholding._ocr("data/adaptive.png", reference_text)
        Thresholding.niblack(photo_img, "niblack")
        Thresholding._ocr("data/niblack.png", reference_text)

    @staticmethod
    def _ocr(filename: str, reference_text: str):
        img = np.array(Image.open(filename))
        text = pytesseract.image_to_string(img, config='--psm 6', lang='pol').strip()
        print("Text matching ratio " + '{:.2%}'.format(SequenceMatcher(None, reference_text, text).ratio()))

    @staticmethod
    def binarisation(image, filename: str):
        start = time.time()
        ret, thresh = cv2.threshold(
            image, 127, 255, cv2.THRESH_BINARY)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        print("----------BINARISATION----------")
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

        print("----------OTSU----------")
        print("Time elapsed {} s".format(end - start))
        print("Noise level {}".format(Metrics.estimate_noise(thresh)))

        plt.imshow(thresh)
        plt.show()

    @staticmethod
    def adaptive(image, filename: str):
        start = time.time()
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 5)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        print("----------ADAPTIVE----------")
        print("Time elapsed {} s".format(end - start))
        print("Noise level {}".format(Metrics.estimate_noise(thresh)))

        plt.imshow(thresh)
        plt.show()

    @staticmethod
    def niblack(image, filename: str):
        start = time.time()
        thresh = cv2.ximgproc.niBlackThreshold(
            image, 255, cv2.THRESH_BINARY, 11, 0)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        print("----------NIBLACK----------")
        print("Time elapsed {} s".format(end - start))
        print("Noise level {}".format(Metrics.estimate_noise(thresh)))

        plt.imshow(thresh)
        plt.show()


