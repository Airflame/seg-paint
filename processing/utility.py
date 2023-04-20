import cv2
from fastapi import UploadFile
import matplotlib.pyplot as plt
import numpy as np


class Utility:
    @staticmethod
    def extract_image_gray(file: UploadFile):
        img = Utility.extract_image(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def extract_image(file: UploadFile):
        contents = file.file.read()
        img = cv2.imdecode(np.fromstring(contents, np.uint8), cv2.IMREAD_COLOR)
        return img

    @staticmethod
    def plot_3d(z: np.array):
        x, y = np.meshgrid(range(z.shape[1]), range(z.shape[0]))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z)
        plt.title('z as 3d height map')
        plt.show()

        plt.figure()
        plt.title('2d')
        p = plt.imshow(z, cmap='hot', extent=[10, 90, 3, 61])
        plt.colorbar(p)
        plt.show()
