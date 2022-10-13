from fastapi import UploadFile
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time

from processing.utility import Utility


class Segmentation:
    @staticmethod
    def k_means(file: UploadFile):
        img = cv2.imdecode(Utility.extract_image(file), cv2.IMREAD_COLOR)
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
