from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import cv2
import time


class Segmentation:
    @staticmethod
    def k_means(path: Path):
        img = cv2.imread(str(path))
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

        plt.imshow(result_image)
        plt.show()
