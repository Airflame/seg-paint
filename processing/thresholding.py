from fastapi import UploadFile
from matplotlib import pyplot as plt
import cv2
import time

from processing.utility import Utility


class Thresholding:
    @staticmethod
    def binarisation(file: UploadFile):
        img = cv2.imdecode(Utility.extract_image(file), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite("data/" + file.filename, thresh1)

        print("Time elapsed {} s".format(end - start))
        plt.imshow(thresh1)
        plt.show()
