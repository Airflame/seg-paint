from fastapi import UploadFile
from matplotlib import pyplot as plt
import numpy as np
import cv2.cv2 as cv2
import time

from processing.utility import Utility


class Segmentation:
    @staticmethod
    def k_means(file: UploadFile):
        img = Utility.extract_image(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        two_dimage = img.reshape((-1, 3))
        two_dimage = np.float32(two_dimage)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        attempts = 10

        start = time.time()
        ret, label, center = cv2.kmeans(two_dimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        end = time.time()

        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape(img.shape)
        cv2.imwrite("data/"+file.filename, result_image)

        print("Time elapsed {} s".format(end-start))
        plt.imshow(result_image)
        plt.show()

    @staticmethod
    def watershed(file: UploadFile):
        img = Utility.extract_image(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(img, markers)
        img[markers == -1] = [0, 255, 0]

        cv2.imwrite("data/" + file.filename, img)
        cv2.imwrite("data/markers-" + file.filename, markers)

        plt.imshow(dist_transform)
        plt.show()

    @staticmethod
    def contour(file: UploadFile):
        img = Utility.extract_image(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        contours, hierarchy = cv2.findContours(sure_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(img, contours, -1, (0,255,0), 1)

        cv2.imwrite("data/" + file.filename, img)

        plt.imshow(sure_bg)
        plt.show()