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

        Thresholding.naive(photo_img, "binarisation")
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
    def calibrate_method(reference_file: UploadFile, photo_files: List[UploadFile], method: str):
        reference_img = Utility.extract_image_gray(reference_file)
        reference_text = pytesseract.image_to_string(reference_img, config='--psm 6', lang='pol').strip()
        photo_images = []
        for photo_file in photo_files:
            photo_images.append(Utility.extract_image_gray(photo_file))

        z = []
        max_match = 0
        selected_block = 0
        selected_k = 0
        for block_size in range(3, 61, 2):
            row = []
            k_range = np.arange(0.1, 0.9, 0.05) if method != "nick" else np.arange(-0.9, -0.1, 0.05)
            for k in k_range:
                results = {"gestalt": [], "cer": [], "wer": []}
                for photo_img in photo_images:
                    if method == "sauvola":
                        Thresholding.sauvola(photo_img, "sauvola", block_size, k, False)
                        single_result = Metrics.estimate_text_readability("data/sauvola.png", reference_text, False)
                    elif method == "wolf":
                        Thresholding.sauvola(photo_img, "nick", block_size, k, False)
                        single_result = Metrics.estimate_text_readability("data/nick.png", reference_text, False)
                    else:
                        Thresholding.sauvola(photo_img, "wolf", block_size, k, False)
                        single_result = Metrics.estimate_text_readability("data/wolf.png", reference_text, False)
                    results["gestalt"].append(single_result["gestalt"])
                    results["cer"].append(single_result["cer"])
                    results["wer"].append(single_result["wer"])
                match = sum(results["gestalt"])/len(results["gestalt"])
                print(str(block_size) + "," + str(k) + " ---> "
                      + str(match))
                row.append(match)
                if match > max_match:
                    max_match = match
                    selected_block = block_size
                    selected_k = k
            z.append(row)
        Utility.plot_3d(np.array(z))
        print(str(max_match) + " " + str(selected_block) + " " + str(selected_k))

    @staticmethod
    def naive(image, filename: str):
        start = time.time()
        ret, thresh = cv2.threshold(
            image, 127, 255, cv2.THRESH_BINARY)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        time_elapsed = end - start
        noise = Metrics.estimate_noise(thresh)
        print("----------NAIVE----------")
        print("Time elapsed {} s".format(time_elapsed))
        print("Noise level {}".format(noise))

        plt.imshow(thresh)
        plt.show()

        return {"filename": filename + ".png", "time-elapsed": time_elapsed, "noise-variance": noise}

    @staticmethod
    def otsu(image, filename: str):
        start = time.time()
        ret, thresh = cv2.threshold(
            image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        time_elapsed = end - start
        noise = Metrics.estimate_noise(thresh)
        print("----------OTSU----------")
        print("Time elapsed {} s".format(time_elapsed))
        print("Noise level {}".format(noise))

        plt.imshow(thresh)
        plt.show()

        return {"filename": filename + ".png", "time-elapsed": time_elapsed, "noise-variance": noise}

    @staticmethod
    def adaptive_gaussian(image, filename: str):
        start = time.time()
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 5)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        time_elapsed = end - start
        noise = Metrics.estimate_noise(thresh)
        print("----------ADAPTIVE-GAUSSIAN----------")
        print("Time elapsed {} s".format(time_elapsed))
        print("Noise level {}".format(noise))

        plt.imshow(thresh)
        plt.show()

        return {"filename": filename + ".png", "time-elapsed": time_elapsed, "noise-variance": noise}

    @staticmethod
    def adaptive_mean(image, filename: str):
        start = time.time()
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 11, 5)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        time_elapsed = end - start
        noise = Metrics.estimate_noise(thresh)
        print("----------ADAPTIVE-MEAN---------")
        print("Time elapsed {} s".format(time_elapsed))
        print("Noise level {}".format(noise))

        plt.imshow(thresh)
        plt.show()

        return {"filename": filename + ".png", "time-elapsed": time_elapsed, "noise-variance": noise}

    @staticmethod
    def niblack(image, filename: str, block_size: int = 41, k: int = 0.2, print_info: bool = True):
        start = time.time()
        thresh = cv2.ximgproc.niBlackThreshold(
            image, 255, cv2.THRESH_BINARY, block_size, k, binarizationMethod=cv2.ximgproc.BINARIZATION_NIBLACK)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        time_elapsed = end - start
        noise = Metrics.estimate_noise(thresh)
        if print_info:
            print("----------NIBLACK----------")
            print("Time elapsed {} s".format(time_elapsed))
            print("Noise level {}".format(noise))

        plt.imshow(thresh)
        plt.show()

        return {"filename": filename + ".png", "time-elapsed": time_elapsed, "noise-variance": noise}

    @staticmethod
    def sauvola(image, filename: str, block_size: int = 41, k: int = 0.2, print_info: bool = True):
        start = time.time()
        thresh = cv2.ximgproc.niBlackThreshold(
            image, 255, cv2.THRESH_BINARY, block_size, k, binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        time_elapsed = end - start
        noise = Metrics.estimate_noise(thresh)
        if print_info:
            print("----------SAUVOLA----------")
            print("Time elapsed {} s".format(time_elapsed))
            print("Noise level {}".format(noise))
            plt.imshow(thresh)
            plt.show()

        return {"filename": filename + ".png", "time-elapsed": time_elapsed, "noise-variance": noise}

    @staticmethod
    def nick(image, filename: str, block_size: int = 41, k: int = -0.2, print_info: bool = True):
        start = time.time()
        thresh = cv2.ximgproc.niBlackThreshold(
            image, 255, cv2.THRESH_BINARY, block_size, k, binarizationMethod=cv2.ximgproc.BINARIZATION_NICK)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        time_elapsed = end - start
        noise = Metrics.estimate_noise(thresh)
        if print_info:
            print("----------NICK----------")
            print("Time elapsed {} s".format(time_elapsed))
            print("Noise level {}".format(noise))
            plt.imshow(thresh)
            plt.show()

        return {"filename": filename + ".png", "time-elapsed": time_elapsed, "noise-variance": noise}

    @staticmethod
    def wolf(image, filename: str, block_size: int = 41, k: int = 0.2, print_info: bool = True):
        start = time.time()
        thresh = cv2.ximgproc.niBlackThreshold(
            image, 255, cv2.THRESH_BINARY, block_size, k, binarizationMethod=cv2.ximgproc.BINARIZATION_WOLF)
        end = time.time()
        cv2.imwrite("data/" + filename + ".png", thresh)

        time_elapsed = end - start
        noise = Metrics.estimate_noise(thresh)
        if print_info:
            print("----------WOLF----------")
            print("Time elapsed {} s".format(time_elapsed))
            print("Noise level {}".format(noise))
            plt.imshow(thresh)
            plt.show()

        return {"filename": filename + ".png", "time-elapsed": time_elapsed, "noise-variance": noise}

