import time
from typing import List
import numpy as np
from fastapi import UploadFile
from matplotlib import pyplot as plt
import cv2.cv2 as cv2
import pytesseract

from processing.metrics import Metrics
from processing.utility import Utility


class Thresholding:
    @staticmethod
    def perform_test(reference_file: UploadFile, photo_file: UploadFile):
        reference_img = Utility.extract_image_gray(reference_file)
        photo_img = Utility.extract_image_gray(photo_file)
        reference_text = pytesseract.image_to_string(reference_img, config='--psm 6', lang='pol').strip()

        Thresholding.binarisation(photo_img, "binarisation")
        Metrics.estimate_text_readability("data/binarisation.png", reference_text)
        Thresholding.otsu(photo_img, "otsu")
        Metrics.estimate_text_readability("data/otsu.png", reference_text)
        Thresholding.adaptive_gaussian(photo_img, "adaptive-gaussian")
        Metrics.estimate_text_readability("data/adaptive-gaussian.png", reference_text)
        Thresholding.adaptive_mean(photo_img, "adaptive-mean")
        Metrics.estimate_text_readability("data/adaptive-mean.png", reference_text)
        Thresholding.niblack(photo_img, "niblack")
        Metrics.estimate_text_readability("data/niblack.png", reference_text)
        Thresholding.sauvola(photo_img, "sauvola")
        Metrics.estimate_text_readability("data/sauvola.png", reference_text)
        Thresholding.nick(photo_img, "nick")
        Metrics.estimate_text_readability("data/nick.png", reference_text)
        Thresholding.wolf(photo_img, "wolf")
        Metrics.estimate_text_readability("data/wolf.png", reference_text)

    @staticmethod
    def perform_sauvola_test(reference_file: UploadFile, photo_files: List[UploadFile]):
        reference_img = Utility.extract_image_gray(reference_file)
        reference_text = pytesseract.image_to_string(reference_img, config='--psm 6', lang='pol').strip()
        photo_images = []
        for photo_file in photo_files:
            photo_images.append(Utility.extract_image_gray(photo_file))

        for block_size in range(3, 51, 2):
            for k in np.arange(0.1, 0.8, 0.1):
                results = {"gestalt": [], "cer": [], "wer": []}
                for photo_img in photo_images:
                    Thresholding.sauvola(photo_img, "sauvola", block_size, k, False)
                    single_result = Metrics.estimate_text_readability("data/sauvola.png", reference_text, False)
                    results["gestalt"].append(single_result["gestalt"])
                    results["cer"].append(single_result["cer"])
                    results["wer"].append(single_result["wer"])
                print(str(block_size) + "," + str(k) + " ---> "
                      + str(sum(results["gestalt"])/len(results["gestalt"])))

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
    def sauvola(image, filename: str, block_size: int = 41, k: int = 0.2, print_info: bool = True):
        start = time.time()
        thresh = cv2.ximgproc.niBlackThreshold(
            image, 255, cv2.THRESH_BINARY, block_size, k, binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        if print_info:
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

    @staticmethod
    def wolf(image, filename: str):
        start = time.time()
        thresh = cv2.ximgproc.niBlackThreshold(
            image, 255, cv2.THRESH_BINARY, 41, 0.2, binarizationMethod=cv2.ximgproc.BINARIZATION_WOLF)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        print("----------WOLF----------")
        print("Time elapsed {} s".format(end - start))
        print("Noise level {}".format(Metrics.estimate_noise(thresh)))

        plt.imshow(thresh)
        plt.show()

