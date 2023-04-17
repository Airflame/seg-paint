import time
import fastwer
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
        Thresholding.adaptive_gaussian(photo_img, "adaptive-gaussian")
        Thresholding._ocr("data/adaptive-gaussian.png", reference_text)
        Thresholding.adaptive_mean(photo_img, "adaptive-mean")
        Thresholding._ocr("data/adaptive-mean.png", reference_text)
        Thresholding.niblack(photo_img, "niblack")
        Thresholding._ocr("data/niblack.png", reference_text)
        Thresholding.sauvola(photo_img, "sauvola")
        Thresholding._ocr("data/sauvola.png", reference_text)
        Thresholding.nick(photo_img, "nick")
        Thresholding._ocr("data/nick.png", reference_text)

    @staticmethod
    def _ocr(filename: str, reference_text: str):
        img = np.array(Image.open(filename))
        text = pytesseract.image_to_string(img, config='--psm 6', lang='pol').strip()
        print("Text matching ratio " + '{:.2%}'.format(SequenceMatcher(None, reference_text, text).ratio()))
        print("Character error rate " + str(fastwer.score_sent(text, reference_text, char_level=True)))
        print("Word error rate " + str(fastwer.score_sent(text, reference_text, char_level=False)))

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
    def adaptive_gaussian(image, filename: str):
        start = time.time()
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 5)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        print("----------ADAPTIVE-GAUSSIAN----------")
        print("Time elapsed {} s".format(end - start))
        print("Noise level {}".format(Metrics.estimate_noise(thresh)))

        plt.imshow(thresh)
        plt.show()

    @staticmethod
    def adaptive_mean(image, filename: str):
        start = time.time()
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 11, 5)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        print("----------ADAPTIVE-MEAN---------")
        print("Time elapsed {} s".format(end - start))
        print("Noise level {}".format(Metrics.estimate_noise(thresh)))

        plt.imshow(thresh)
        plt.show()

    @staticmethod
    def niblack(image, filename: str):
        start = time.time()
        thresh = cv2.ximgproc.niBlackThreshold(
            image, 255, cv2.THRESH_BINARY, 41, 0.2, binarizationMethod=cv2.ximgproc.BINARIZATION_NIBLACK)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        print("----------NIBLACK----------")
        print("Time elapsed {} s".format(end - start))
        print("Noise level {}".format(Metrics.estimate_noise(thresh)))

        plt.imshow(thresh)
        plt.show()

    @staticmethod
    def sauvola(image, filename: str):
        start = time.time()
        thresh = cv2.ximgproc.niBlackThreshold(
            image, 255, cv2.THRESH_BINARY, 41, 0.2, binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        print("----------SAUVOLA----------")
        print("Time elapsed {} s".format(end - start))
        print("Noise level {}".format(Metrics.estimate_noise(thresh)))

        plt.imshow(thresh)
        plt.show()

    @staticmethod
    def nick(image, filename: str):
        start = time.time()
        thresh = cv2.ximgproc.niBlackThreshold(
            image, 255, cv2.THRESH_BINARY, 41, -0.2, binarizationMethod=cv2.ximgproc.BINARIZATION_NICK)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        print("----------NICK----------")
        print("Time elapsed {} s".format(end - start))
        print("Noise level {}".format(Metrics.estimate_noise(thresh)))

        plt.imshow(thresh)
        plt.show()


